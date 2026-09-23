"""
Unified classification dataset builder (spec section 16 pipeline):
  RAW DATA -> METADATA EXTRACTION -> LABEL NORMALIZATION -> QUALITY CONTROL ->
  DUPLICATE DETECTION -> LEAKAGE-SAFE SPLIT -> PREPROCESSING -> TRAINING DATA

Sources combined (per configs/label_schema.yaml):
  - PBC (Acevedo): neutrophil, eosinophil, basophil, lymphocyte, monocyte, platelet
    (its "ig" and "erythroblast" folders are copied to an out-of-scope-v1 bucket,
    NEVER merged into a target class)
  - TXL-PBC-derived crops (scripts/extract_rbc_crops_from_txlpbc.py output):
    rbc only. This is the sole RBC source (see label_schema.yaml notes) --
    documented domain-shift risk, not hidden.

Leakage-safety note (spec section 17): PBC and its own single-cell images carry
no public per-image patient/donor id (see dataset_feasibility_audit.md section 9),
so PBC images are split with a fixed-seed stratified random split at the image
level -- a documented, weaker-than-ideal fallback, not a silent assumption of
independence. TXL-PBC-derived RBC crops instead KEEP the split their whole-smear
parent image was already assigned to in the original TXL-PBC release, so that
crops from the same source photograph never straddle train/val/test.

Quality control: corrupt/unreadable files are skipped and logged, not silently
dropped without a trace.

Duplicate detection: a simple 8x8 difference-hash (dHash) computed with
Pillow + NumPy only (no extra dependency) flags near-duplicate images within
the PBC pool; flagged duplicates are kept only once (first occurrence).

Run:
    python scripts/extract_rbc_crops_from_txlpbc.py   # must run first
    python scripts/build_classification_dataset.py
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PBC_ROOT = REPO_ROOT / "datasets" / "raw" / "pbc_acevedo" / "extracted" / "PBC_dataset_normal_DIB"
RBC_CROPS_ROOT = REPO_ROOT / "datasets" / "processed" / "txl_pbc_crops" / "rbc"
RBC_CROPS_META = REPO_ROOT / "datasets" / "processed" / "txl_pbc_crops" / "crop_metadata.csv"
OUT_ROOT = REPO_ROOT / "datasets" / "processed" / "classification_v1"

PBC_FOLDER_TO_LABEL = {
    "neutrophil": "neutrophil",
    "eosinophil": "eosinophil",
    "basophil": "basophil",
    "lymphocyte": "lymphocyte",
    "monocyte": "monocyte",
    "platelet": "platelet",
}
PBC_OUT_OF_SCOPE = {"ig": "immature_granulocyte", "erythroblast": "erythroblast"}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
MIN_IMAGE_DIM = 20  # px, quality-control floor


def dhash(image_path: Path, hash_size: int = 8) -> str:
    img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size))
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    return "".join("1" if v else "0" for v in diff.flatten())


def load_pbc_items():
    """Returns list of (path, label, quality_note) for target classes,
    and a separate list for out-of-scope-v1 classes."""
    target_items = []
    out_of_scope_items = []
    qc_skipped = []

    for folder, label in {**PBC_FOLDER_TO_LABEL, **PBC_OUT_OF_SCOPE}.items():
        folder_path = PBC_ROOT / folder
        if not folder_path.exists():
            print(f"WARNING: expected PBC folder missing: {folder_path}")
            continue
        for img_path in sorted(folder_path.glob("*.jpg")):
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    im.verify()
            except Exception as exc:
                qc_skipped.append({"path": str(img_path), "reason": str(exc)})
                continue
            if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                qc_skipped.append({"path": str(img_path), "reason": f"too small ({w}x{h})"})
                continue
            record = {"path": img_path, "label": label, "source_dataset": "pbc_acevedo"}
            if folder in PBC_FOLDER_TO_LABEL:
                target_items.append(record)
            else:
                out_of_scope_items.append(record)
    return target_items, out_of_scope_items, qc_skipped


def dedupe_pbc(items):
    """Drop near-duplicate images (same dHash) within the PBC pool, keep first occurrence."""
    seen_hashes = {}
    deduped = []
    duplicates_dropped = []
    for item in items:
        h = dhash(item["path"])
        if h in seen_hashes:
            duplicates_dropped.append({"path": str(item["path"]), "duplicate_of": str(seen_hashes[h])})
            continue
        seen_hashes[h] = item["path"]
        deduped.append(item)
    return deduped, duplicates_dropped


def split_pbc_stratified(items):
    """Fixed-seed stratified random split per class (documented fallback -- no
    per-image patient id exists in PBC, see dataset_feasibility_audit.md section 9)."""
    rng = random.Random(RANDOM_SEED)
    by_class = defaultdict(list)
    for item in items:
        by_class[item["label"]].append(item)

    split_assignment = {}
    for label, class_items in by_class.items():
        shuffled = class_items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        for i, item in enumerate(shuffled):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_assignment[str(item["path"])] = split
    return split_assignment


def load_rbc_items():
    """RBC crops keep the split already assigned to their TXL-PBC parent image."""
    items = []
    if not RBC_CROPS_META.exists():
        print("WARNING: RBC crop metadata not found -- run extract_rbc_crops_from_txlpbc.py first.")
        return items
    with open(RBC_CROPS_META, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            crop_path = RBC_CROPS_ROOT / row["crop_filename"]
            if not crop_path.exists():
                continue
            items.append({
                "path": crop_path,
                "label": "rbc",
                "source_dataset": "txl_pbc",
                "split": row["source_split_at_acquisition"],
                "source_image": row["source_image"],
            })
    return items


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for label in ["rbc", "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil", "platelet"]:
            (OUT_ROOT / split / label).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "out_of_scope_v1").mkdir(parents=True, exist_ok=True)

    pbc_target, pbc_out_of_scope, pbc_qc_skipped = load_pbc_items()
    print(f"PBC target-class images loaded: {len(pbc_target)}, out-of-scope: {len(pbc_out_of_scope)}, QC-skipped: {len(pbc_qc_skipped)}")

    pbc_deduped, pbc_duplicates = dedupe_pbc(pbc_target)
    print(f"PBC after dedup: {len(pbc_deduped)} (dropped {len(pbc_duplicates)} near-duplicates)")

    pbc_split_map = split_pbc_stratified(pbc_deduped)

    rbc_items = load_rbc_items()
    print(f"RBC crops loaded from TXL-PBC: {len(rbc_items)}")

    metadata_rows = []
    counts = defaultdict(lambda: defaultdict(int))

    for item in pbc_deduped:
        split = pbc_split_map[str(item["path"])]
        dest_dir = OUT_ROOT / split / item["label"]
        dest_path = dest_dir / item["path"].name
        dest_path.write_bytes(item["path"].read_bytes())
        metadata_rows.append({
            "filename": item["path"].name, "label": item["label"], "split": split,
            "source_dataset": item["source_dataset"], "original_path": str(item["path"]),
        })
        counts[split][item["label"]] += 1

    for item in rbc_items:
        split = item["split"]
        dest_dir = OUT_ROOT / split / item["label"]
        dest_path = dest_dir / item["path"].name
        dest_path.write_bytes(item["path"].read_bytes())
        metadata_rows.append({
            "filename": item["path"].name, "label": item["label"], "split": split,
            "source_dataset": item["source_dataset"], "original_path": str(item["path"]),
        })
        counts[split][item["label"]] += 1

    for item in pbc_out_of_scope:
        dest_dir = OUT_ROOT / "out_of_scope_v1" / item["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / item["path"].name
        dest_path.write_bytes(item["path"].read_bytes())

    with open(OUT_ROOT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "split", "source_dataset", "original_path"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    leakage_report = {
        "random_seed": RANDOM_SEED,
        "pbc_qc_skipped_count": len(pbc_qc_skipped),
        "pbc_qc_skipped": pbc_qc_skipped,
        "pbc_duplicates_dropped_count": len(pbc_duplicates),
        "pbc_duplicates_dropped": pbc_duplicates,
        "pbc_split_method": "fixed-seed stratified random split at image level -- NO per-patient id available, documented limitation (see dataset_feasibility_audit.md section 9)",
        "rbc_split_method": "inherits original TXL-PBC train/val/test assignment at the source whole-smear-image level, so multiple RBC crops from one source photo never straddle splits",
        "counts_by_split_and_class": {split: dict(labels) for split, labels in counts.items()},
    }
    with open(OUT_ROOT / "leakage_and_qc_report.json", "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2)

    print(json.dumps(leakage_report["counts_by_split_and_class"], indent=2))
    print(f"Done. Dataset at {OUT_ROOT}")


if __name__ == "__main__":
    main()
