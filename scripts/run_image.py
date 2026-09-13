"""Single-image defect inspection from the command line.

Usage (from project root):
    python scripts\\run_image.py data\\yolo\\images\\val\\scratches_1.jpg
    python scripts\\run_image.py img.jpg --no-save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Image defect inspection")
    parser.add_argument("image", help="path to the image to inspect")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--no-save", action="store_true", help="do not save annotated output")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / img_path
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}")
        return 1
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: corrupted/unreadable image: {img_path}")
        return 1

    from app.inference.detector import ModelNotTrainedError
    from app.services.detection_service import detection_service

    try:
        result = detection_service.inspect_image(img, source=img_path.name, save_annotated=not args.no_save)
    except ModelNotTrainedError as exc:
        print("CUSTOM DEFECT MODEL NOT TRAINED")
        print(str(exc))
        return 1

    import json

    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
