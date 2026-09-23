"""
Extract single-cell RBC (and platelet, WBC) crops from the TXL-PBC YOLO-format
whole-smear bounding boxes. PBC (Acevedo) has no mature-RBC single-cell images,
so RBC training crops for the 7-class classifier come from here instead.

Provenance is preserved: every crop's filename encodes its source image and
box index, and a metadata CSV records the mapping back to the original
TXL-PBC image + split it came from (spec section 15/16 requirement).

Usage:
    python scripts/extract_rbc_crops_from_txlpbc.py
"""
import csv
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TXL_ROOT = REPO_ROOT / "datasets" / "raw" / "txl_pbc" / "extracted" / "TXL-PBC"
OUT_ROOT = REPO_ROOT / "datasets" / "processed" / "txl_pbc_crops"

CLASS_INDEX_TO_LABEL = {0: "wbc_unclassified", 1: "rbc", 2: "platelet"}
MIN_CROP_SIZE = 8  # px, guards against degenerate boxes


def yolo_box_to_pixels(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))


def process_split(split: str, writer: csv.DictWriter) -> dict:
    images_dir = TXL_ROOT / "images" / split
    labels_dir = TXL_ROOT / "labels" / split
    counts = {label: 0 for label in CLASS_INDEX_TO_LABEL.values()}

    for img_path in sorted(images_dir.glob("*")):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"SKIP unreadable image {img_path.name}: {exc}")
            continue
        img_w, img_h = img.size

        lines = label_path.read_text().strip().splitlines()
        for box_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_idx = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            label = CLASS_INDEX_TO_LABEL.get(cls_idx)
            if label is None:
                continue

            x1, y1, x2, y2 = yolo_box_to_pixels(cx, cy, w, h, img_w, img_h)
            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                continue

            crop = img.crop((x1, y1, x2, y2))
            out_dir = OUT_ROOT / label
            out_dir.mkdir(parents=True, exist_ok=True)
            crop_filename = f"{split}_{img_path.stem}_box{box_idx}.jpg"
            crop.save(out_dir / crop_filename, quality=95)

            writer.writerow({
                "crop_filename": crop_filename,
                "label": label,
                "source_dataset": "txl_pbc",
                "source_split_at_acquisition": split,
                "source_image": img_path.name,
                "box_index": box_idx,
                "bbox_xyxy": f"{x1},{y1},{x2},{y2}",
                "source_image_width": img_w,
                "source_image_height": img_h,
            })
            counts[label] += 1
    return counts


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT_ROOT / "crop_metadata.csv"
    fieldnames = [
        "crop_filename", "label", "source_dataset", "source_split_at_acquisition",
        "source_image", "box_index", "bbox_xyxy", "source_image_width", "source_image_height",
    ]

    total_counts = {label: 0 for label in CLASS_INDEX_TO_LABEL.values()}
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val", "test"):
            split_counts = process_split(split, writer)
            print(f"{split}: {split_counts}")
            for k, v in split_counts.items():
                total_counts[k] += v

    print("TOTAL:", total_counts)
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
