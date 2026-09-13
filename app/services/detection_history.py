"""SQLite-backed detection history (real detections only — no fake data).

Schema per event: event_id, timestamp, source, defect_class, confidence,
severity, bbox, inference_latency_ms, inspection_status.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.config import resolve_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detection_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    defect_class TEXT,
    confidence REAL,
    severity TEXT,
    bbox TEXT,
    inference_latency_ms REAL,
    inspection_status TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON detection_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_class ON detection_events(defect_class);
"""


class DetectionHistory:
    """Thread-safe wrapper around the SQLite detection-events database."""

    def __init__(self, db_path: str | Path):
        self.db_path = resolve_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def record_inspection(self, result_dict: dict) -> None:
        """Persist one inspection (one row per detected defect; PASS rows get a summary row)."""
        ts = result_dict.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        source = result_dict.get("source", "")
        latency = result_dict.get("performance", {}).get("inference_ms", 0.0)
        status = result_dict.get("inspection_status", "PASS")
        rows = []
        detections = result_dict.get("detections", [])
        if detections:
            for det in detections:
                rows.append(
                    (ts, source, det.get("class"), float(det.get("confidence", 0.0)),
                     det.get("severity"), json.dumps(det.get("bbox", {})), latency, status)
                )
        else:
            rows.append((ts, source, None, None, None, None, latency, status))
        with self._lock, closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO detection_events (timestamp, source, defect_class, confidence, severity, bbox, inference_latency_ms, inspection_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def query(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
        defect_class: str | None = None,
        severity: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        sql = "SELECT * FROM detection_events WHERE 1=1"
        params: list = []
        if status:
            sql += " AND inspection_status = ?"
            params.append(status)
        if defect_class:
            sql += " AND defect_class = ?"
            params.append(defect_class)
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY event_id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self, status: str | None = None, defect_class: str | None = None,
              severity: str | None = None, since: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM detection_events WHERE 1=1"
        params: list = []
        if status:
            sql += " AND inspection_status = ?"
            params.append(status)
        if defect_class:
            sql += " AND defect_class = ?"
            params.append(defect_class)
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        with self._lock, closing(self._connect()) as conn:
            return conn.execute(sql, params).fetchone()[0]

    def summary(self) -> dict:
        """Aggregate stats for the dashboard. Empty-safe: zeros when no data."""
        with self._lock, closing(self._connect()) as conn:
            total = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
            passed = conn.execute("SELECT COUNT(*) FROM detection_events WHERE inspection_status='PASS'").fetchone()[0]
            rejected = conn.execute("SELECT COUNT(*) FROM detection_events WHERE inspection_status='REJECT'").fetchone()[0]
            avg_lat = conn.execute(
                "SELECT AVG(inference_latency_ms) FROM detection_events WHERE inference_latency_ms > 0"
            ).fetchone()[0]
            class_counts = dict(
                conn.execute(
                    "SELECT defect_class, COUNT(*) FROM detection_events WHERE defect_class IS NOT NULL GROUP BY defect_class"
                ).fetchall()
            )
            trend = dict(
                conn.execute(
                    "SELECT substr(timestamp, 1, 16), COUNT(*) FROM detection_events "
                    "WHERE defect_class IS NOT NULL GROUP BY substr(timestamp, 1, 16) ORDER BY substr(timestamp, 1, 16) DESC LIMIT 60"
                ).fetchall()
            )
            avg_conf = conn.execute("SELECT AVG(confidence) FROM detection_events WHERE confidence IS NOT NULL").fetchone()[0]
        defect_rate = (rejected / total * 100.0) if total else 0.0
        return {
            "total_events": total,
            "passed": passed,
            "rejected": rejected,
            "defect_rate_pct": round(defect_rate, 2),
            "avg_inference_ms": round(avg_lat, 2) if avg_lat else 0.0,
            "avg_confidence": round(avg_conf, 4) if avg_conf else 0.0,
            "class_counts": class_counts,
            "trend": trend,
        }

    def clear(self) -> int:
        with self._lock, closing(self._connect()) as conn, conn:
            n = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
            conn.execute("DELETE FROM detection_events")
        return n
