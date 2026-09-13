"""PLC simulator — automated sorting signal generation.

The system talks to an abstract PLCAdapter. The default implementation,
SimulatedPLC, just records the sorting commands (no physical hardware is
required). Real deployments could implement the same interface over
Modbus TCP, OPC UA or MQTT — see adapter docstrings.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from datetime import datetime

from app.services.logging_service import get_logger

logger = get_logger("defect.plc")


class PLCAdapter(ABC):
    """Interface for an industrial controller.

    Implement this with pymodbus / opcua / paho-mqtt for real hardware.
    """

    @abstractmethod
    def send_sort_command(self, action: str, defect: str | None = None, confidence: float | None = None) -> dict:
        """Send a PASS or REJECT sorting command to the line."""


class SimulatedPLC(PLCAdapter):
    """Simulated PLC: logs every command and keeps an in-memory event log."""

    def __init__(self, line_id: str = "LINE-01"):
        self.line_id = line_id
        self._lock = threading.Lock()
        self.events: list[dict] = []
        self.total_pass = 0
        self.total_reject = 0

    def send_sort_command(self, action: str, defect: str | None = None, confidence: float | None = None) -> dict:
        action = action.upper()
        if action not in ("PASS", "REJECT"):
            raise ValueError(f"invalid PLC action: {action!r} (expected PASS or REJECT)")

        command = {
            "action": action,
            "reject_signal": action == "REJECT",
            "defect": defect,
            "confidence": confidence,
            "line_id": self.line_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock:
            self.events.append(command)
            if action == "PASS":
                self.total_pass += 1
            else:
                self.total_reject += 1
        logger.info("PLC %s <- %s (defect=%s, confidence=%s)", self.line_id, action, defect, confidence)
        return command

    def last_event(self) -> dict | None:
        with self._lock:
            return self.events[-1] if self.events else None

    def stats(self) -> dict:
        with self._lock:
            return {
                "line_id": self.line_id,
                "total_pass": self.total_pass,
                "total_reject": self.total_reject,
                "total_commands": len(self.events),
            }


class ModbusTCPPLC(PLCAdapter):
    """Modbus TCP adapter skeleton (requires a real controller + pymodbus).

    Writing to the coil register happens only when pymodbus is installed AND
    a host is configured; otherwise construction raises RuntimeError so the
    caller can fall back to the simulator.
    """

    def __init__(self, host: str, port: int = 502, unit_id: int = 1, coil_address: int = 0):
        try:
            from pymodbus.client import ModbusTcpClient  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pymodbus not installed — Modbus PLC adapter unavailable. "
                "Falling back to SimulatedPLC is recommended."
            ) from exc
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.coil_address = coil_address

    def send_sort_command(self, action: str, defect: str | None = None, confidence: float | None = None) -> dict:
        from pymodbus.client import ModbusTcpClient

        client = ModbusTcpClient(self.host, port=self.port)
        try:
            if not client.connect():
                raise ConnectionError(f"cannot connect to Modbus PLC at {self.host}:{self.port}")
            client.write_coil(self.coil_address, action == "REJECT", slave=self.unit_id)
        finally:
            client.close()
        command = {
            "action": action,
            "reject_signal": action == "REJECT",
            "defect": defect,
            "confidence": confidence,
            "line_id": f"modbus://{self.host}:{self.port}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        logger.info("Modbus PLC command: %s", command)
        return command


# Default PLC instance used by the API
plc = SimulatedPLC()
