"""Inspect the NEU-DET dataset and print a full statistics report.

Usage (from project root):
    python training\\inspect_dataset.py
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from app.config import CLASS_NAMES, RAW_DATA_DIR  # noqa: E402

EXPECTED_CLASSES = set(CLASS_NAMES)


def _iter_splits(raw_dir: Path):
    """Yield (split_name, annotations_dir, images_dir) for every detected split."""
    base = raw_dir / "NEU-DET"
    if base.exists():
        for split_dir in sorted(base.iterdir()):
            if not split_dir.is_dir():
                continue
            ann = split_dir / "annotations"
            img = split_dir / "images"
            if ann.exists() and img.exists():
                yield split_dir.name, ann, img
        return
    # Fallback: user may have a flat dataset layout
    ann = raw_dir / "annotations"
    img = raw_dir / "images"
    if ann.exists() and img.exists():
        yield "all", ann, img


def inspect_split(split: str, ann_dir: Path, img_dir: Path) -> dict:
    report: dict = {
        "split": split,
        "images": 0,
        "annotations": 0,
        "class_counts": Counter(),
        "total_boxes": 0,
        "boxes_per_image": Counter(),
        "malformed_xml": [],
        "invalid_boxes": [],
        "missing_pairs": [],
        "image_sizes": Counter(),
        "empty_annotations": [],
        "unknown_classes": Counter(),
    }

    xml_files = sorted(ann_dir.glob("*.xml"))
    image_files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    report["annotations"] = len(xml_files)
    report["images"] = len(image_files)

    img_stems = {p.stem: p for p in image_files}
    xml_stems = {p.stem for p in xml_files}

    for stem in sorted(set(img_stems) - xml_stems):
        report["missing_pairs"].append(f"{stem}.jpg has NO annotation XML")
    for stem in sorted(xml_stems - set(img_stems)):
        report["missing_pairs"].append(f"{stem}.xml has NO image file")

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            report["malformed_xml"].append(f"{xml_path.name}: {exc}")
            continue

        objects = root.findall("object")
        if not objects:
            report["empty_annotations"].append(xml_path.name)

        size = root.find("size")
        if size is not None:
            try:
                w = int(size.findtext("width", "0"))
                h = int(size.findtext("height", "0"))
                report["image_sizes"][(w, h)] += 1
            except ValueError:
                pass

        n_valid_in_file = 0
        for obj in objects:
            name = obj.findtext("name", "").strip()
            if name not in EXPECTED_CLASSES:
                report["unknown_classes"][name] += 1
                continue
            report["class_counts"][name] += 1
            box = obj.find("bndbox")
            try:
                xmin = int(float(box.findtext("xmin", "0")))
                ymin = int(float(box.findtext("ymin", "0")))
                xmax = int(float(box.findtext("xmax", "0")))
                ymax = int(float(box.findtext("ymax", "0")))
            except (TypeError, ValueError):
                report["invalid_boxes"].append(f"{xml_path.name}: non-numeric bbox for '{name}'")
                continue
            if xmax <= xmin or ymax <= ymin or xmin < 0 or ymin < 0:
                report["invalid_boxes"].append(
                    f"{xml_path.name}: degenerate bbox ({xmin},{ymin},{xmax},{ymax}) for '{name}'"
                )
                continue
            n_valid_in_file += 1
        report["total_boxes"] += n_valid_in_file
        report["boxes_per_image"][n_valid_in_file] += 1

    # Verify a sample of image dimensions against the XML claims
    for p in image_files[:50]:
        try:
            with Image.open(p) as im:
                report.setdefault("actual_sample_sizes", Counter())[(im.width, im.height)] += 1
        except OSError:
            report["missing_pairs"].append(f"{p.name}: corrupted/unreadable image")

    return report


def print_report(reports: list[dict]) -> None:
    total_imgs = sum(r["images"] for r in reports)
    total_anns = sum(r["annotations"] for r in reports)
    total_boxes = sum(r["total_boxes"] for r in reports)

    print("=" * 70)
    print("NEU-DET DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"Total images:          {total_imgs}")
    print(f"Total annotation files:{total_anns}")
    print(f"Total bounding boxes:  {total_boxes}")
    print()

    for r in reports:
        print(f"--- Split: {r['split']} " + "-" * (55 - len(r["split"])))
        print(f"Images:       {r['images']}")
        print(f"Annotations:  {r['annotations']}")
        print("Class distribution:")
        grand = sum(r["class_counts"].values()) or 1
        for cls in CLASS_NAMES:
            c = r["class_counts"].get(cls, 0)
            print(f"  {cls:<16} {c:>6}   ({100 * c / grand:.1f}%)")
        for cls, c in r["unknown_classes"].items():
            print(f"  UNKNOWN '{cls}': {c}")
        multi = sum(v for k, v in r["boxes_per_image"].items() if k > 1)
        print(f"Multi-object images: {multi} / {r['annotations']}")
        print(f"Image sizes (from XML): {dict(r['image_sizes'])}")
        if "actual_sample_sizes" in r:
            print(f"Image sizes (sampled files): {dict(r['actual_sample_sizes'])}")
        if r["malformed_xml"]:
            print(f"Malformed XML ({len(r['malformed_xml'])}):")
            for m in r["malformed_xml"][:10]:
                print(f"  {m}")
        if r["invalid_boxes"]:
            print(f"Invalid boxes ({len(r['invalid_boxes'])}):")
            for m in r["invalid_boxes"][:10]:
                print(f"  {m}")
        if r["missing_pairs"]:
            print(f"Missing/mismatched pairs ({len(r['missing_pairs'])}):")
            for m in r["missing_pairs"][:10]:
                print(f"  {m}")
        if r["empty_annotations"]:
            print(f"Empty annotations ({len(r['empty_annotations'])}): {r['empty_annotations'][:5]}")
        if not (r["malformed_xml"] or r["invalid_boxes"] or r["missing_pairs"] or r["empty_annotations"]):
            print("No data integrity issues found in this split.")
        print()

    overall = Counter()
    for r in reports:
        overall.update(r["class_counts"])
    print("=" * 70)
    print("OVERALL CLASS DISTRIBUTION")
    print("=" * 70)
    for cls in CLASS_NAMES:
        print(f"  {cls:<16} {overall.get(cls, 0):>6}")


def main() -> int:
    reports = []
    for split, ann_dir, img_dir in _iter_splits(RAW_DATA_DIR):
        reports.append(inspect_split(split, ann_dir, img_dir))
    if not reports:
        print(f"ERROR: no dataset found under {RAW_DATA_DIR}")
        print("Extract the archive first, e.g. data\\raw\\NEU-DET\\...")
        return 1
    print_report(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
