"""Industrial-image augmentation pipeline for the NEU defect dataset.

Uses Albumentations with bounding-box-safe transforms only. All probabilities
are configurable. Only realistic manufacturing-image transformations are used
(small rotations, flips, exposure/noise variations, CLAHE) — nothing that
would destroy the physical characteristics of the six defect classes.

Standalone demo (writes augmented samples to outputs/training/augmented):
    python training\\augment.py --count 12

NOTE: Ultralytics YOLO already applies its own on-the-fly augmentation during
training (mosaic, HSV, flips, scale...). This module exists for
1) generating extra offline samples for a small-dataset regime and
2) demonstrating a bbox-safe augmentation pipeline explicitly.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CLASS_NAMES, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR, TRAINING_OUTPUT_DIR  # noqa: E402

BBOX_PARAMS = dict(format="yolo", min_visibility=0.3, min_area=25, clip=True, check_each_transform=True)


def build_pipeline(cfg: dict | None = None) -> A.Compose:
    """Build the augmentation pipeline. All probabilities configurable via cfg."""
    cfg = cfg or {}
    p = {
        "hflip": cfg.get("hflip", 0.5),
        "vflip": cfg.get("vflip", 0.2),
        "rotate": cfg.get("rotate", 0.4),
        "brightness_contrast": cfg.get("brightness_contrast", 0.5),
        "gamma": cfg.get("gamma", 0.3),
        "gauss_noise": cfg.get("gauss_noise", 0.3),
        "blur": cfg.get("blur", 0.2),
        "clahe": cfg.get("clahe", 0.3),
    }
    return A.Compose(
        [
            A.HorizontalFlip(p=p["hflip"]),
            A.VerticalFlip(p=p["vflip"]),
            A.Rotate(limit=7, border_mode=cv2.BORDER_REFLECT_101, p=p["rotate"]),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=p["brightness_contrast"]),
            A.RandomGamma(gamma_limit=(85, 115), p=p["gamma"]),
            A.GaussNoise(std_range=(0.01, 0.05), p=p["gauss_noise"]),
            A.GaussianBlur(blur_limit=(3, 3), p=p["blur"]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p["clahe"]),
        ],
        bbox_params=A.BboxParams(**BBOX_PARAMS),
    )


def load_yolo_labels(label_path: Path) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append([int(parts[0]), *map(float, parts[1:])])
    return boxes


def save_yolo_labels(label_path: Path, boxes: list[list[float]]) -> None:
    with label_path.open("w", encoding="utf-8") as fh:
        for b in boxes:
            fh.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def draw_boxes(img: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cid = int(b[0])
        x1 = int((b[1] - b[3] / 2) * w)
        y1 = int((b[2] - b[4] / 2) * h)
        x2 = int((b[1] + b[3] / 2) * w)
        y2 = int((b[2] + b[4] / 2) * h)
        name = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.putText(out, name, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
    return out


def augment_dataset(n_per_image: int = 1, seed: int = 42, limit: int | None = None) -> int:
    """Generate augmented copies of the training set (images + bbox-safe labels)."""
    random.seed(seed)
    rng = np.random.default_rng(seed)
    pipeline = build_pipeline()
    out_img_dir = IMAGES_TRAIN_DIR.parent.parent / "images" / "train_aug"
    out_lbl_dir = LABELS_TRAIN_DIR.parent.parent / "labels" / "train_aug"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(IMAGES_TRAIN_DIR.glob("*.jpg"))[: limit] if limit else sorted(IMAGES_TRAIN_DIR.glob("*.jpg"))
    total_written = 0
    for img_path in images:
        lbl_path = LABELS_TRAIN_DIR / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # Ultralytics stores labels as class cx cy w h (same layout)
        boxes = load_yolo_labels(lbl_path)
        if not boxes:
            continue
        for i in range(n_per_image):
            # Albumentations 2.x requires numeric labels: [cx, cy, w, h, class_id]
            alb_boxes = [[b[1], b[2], b[3], b[4], int(b[0])] for b in boxes]
            try:
                result = pipeline(image=img, bboxes=alb_boxes)
            except Exception:
                continue
            new_boxes = result["bboxes"]
            if not new_boxes:
                continue
            aug_img = result["image"]
            idx = rng.integers(0, 1_000_000)
            name = f"{img_path.stem}_aug{idx}.jpg"
            cv2.imencode(".jpg", aug_img)[1].tofile(out_img_dir / name)
            rows = []
            for bb in new_boxes:
                rows.append(f"{int(bb[4])} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
            (out_lbl_dir / f"{img_path.stem}_aug{idx}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            total_written += 1
    return total_written


def main() -> int:
    parser = argparse.ArgumentParser(description="Industrial-image augmentation demo / offline generator")
    parser.add_argument("--count", type=int, default=12, help="augmented demo samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-image", type=int, default=0, help="if >0, augment the whole training set n-per-image")
    args = parser.parse_args()

    if args.per_image > 0:
        n = augment_dataset(n_per_image=args.per_image, seed=args.seed)
        print(f"Wrote {n} augmented training samples to data/yolo/images|labels/train_aug")
        return 0

    # Demo: visualise the pipeline on a few real training images
    out_dir = TRAINING_OUTPUT_DIR / "augmented"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline()
    images = sorted(IMAGES_TRAIN_DIR.glob("*.jpg"))
    if not images:
        print("No converted training images found. Run training\\convert_voc_to_yolo.py first.")
        return 1
    rng = random.Random(args.seed)
    picks = rng.sample(images, min(args.count, len(images)))
    for img_path in picks:
        lbl_path = LABELS_TRAIN_DIR / f"{img_path.stem}.txt"
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        boxes = load_yolo_labels(lbl_path)
        alb_boxes = [[b[1], b[2], b[3], b[4], int(b[0])] for b in boxes]
        try:
            result = pipeline(image=img, bboxes=alb_boxes)
        except Exception as exc:
            print(f"skip {img_path.name}: {exc}")
            continue
        vis = draw_boxes(result["image"], [[int(bb[4]), *bb[:4]] for bb in result["bboxes"]])
        orig = draw_boxes(img, boxes)
        side = np.hstack([orig, vis])
        cv2.imencode(".jpg", side)[1].tofile(out_dir / f"{img_path.stem}_augdemo.jpg")
    print(f"Wrote {len(picks)} before/after demo images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
