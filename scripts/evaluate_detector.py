"""
Detailed, honest detection evaluation beyond aggregate mAP (spec section 22
and this session's explicit instruction not to hide weak detection
performance behind an aggregate number).

Runs the trained detector on the REAL, held-out TXL-PBC test split at the
SAME confidence threshold the deployed app actually uses
(WholeSmearAnalyzer's default 0.25), not an idealized threshold sweep, then
does real IoU-based ground-truth matching (greedy, per-class, IoU>=0.5) to
report:
  - per-class precision/recall/F1 (WBC/RBC/Platelet) from actual TP/FP/FN
    counts, not just Ultralytics' internal mAP calculation
  - recall stratified by GT box size (small/medium/large tercile) --
    checks whether small objects (platelets, small RBCs) are
    disproportionately missed
  - recall stratified by whether a GT box touches another GT box (IoU>0
    between ground-truth boxes) -- a real, data-derived proxy for
    "touching/overlapping cells", since the real test images already
    contain naturally touching/overlapping cells; no synthetic case needed
  - recall stratified by whether a GT box touches the image boundary
  - recall stratified by per-image cell density (this test split's images
    range from 5 to 28 ground-truth cells)

No number in the output is invented: every count comes from an actual
model prediction matched (or not) against actual ground-truth boxes on
the real held-out test images.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TXL_ROOT = REPO_ROOT / "datasets" / "raw" / "txl_pbc" / "extracted" / "TXL-PBC"
MODEL_PATH = REPO_ROOT / "models" / "detector_v1.pt"
EVAL_DIR = REPO_ROOT / "evaluation"

CLASS_NAMES = ["WBC", "RBC", "Platelet"]
CONF_THRESHOLD = 0.25  # matches WholeSmearAnalyzer's default -- real deployed behavior, not a sweep
IOU_MATCH_THRESHOLD = 0.5
BOUNDARY_MARGIN_FRAC = 0.03  # a box within 3% of image width/height of an edge counts as "boundary"


def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_idx = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append({"cls": cls_idx, "xyxy": (x1, y1, x2, y2)})
    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def touches_boundary(box, img_w, img_h, margin_frac=BOUNDARY_MARGIN_FRAC):
    x1, y1, x2, y2 = box
    mx, my = img_w * margin_frac, img_h * margin_frac
    return x1 <= mx or y1 <= my or x2 >= img_w - mx or y2 >= img_h - my


def main():
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained detector at {MODEL_PATH}. Run scripts/train_detector.py first.")

    model = YOLO(str(MODEL_PATH))

    images_dir = TXL_ROOT / "images" / "test"
    labels_dir = TXL_ROOT / "labels" / "test"

    # Confusion-style counters
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    # cls_true -> cls_pred_or_'missed' misclassification tally (for boxes that
    # matched spatially via IoU but to the WRONG class -- a real detection
    # error mode distinct from pure localization miss)
    class_confusion = defaultdict(lambda: defaultdict(int))

    size_bucket_stats = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}     # [matched, total]
    boundary_stats = {"boundary": [0, 0], "interior": [0, 0]}
    touching_stats = {"touching": [0, 0], "isolated": [0, 0]}
    density_bucket_stats = defaultdict(lambda: [0, 0])  # density tercile -> [matched, total]

    all_gt_areas = []
    per_image_records = []

    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        img_path = images_dir / (label_path.stem + ".png")
        if not img_path.exists():
            img_path = images_dir / (label_path.stem + ".jpg")
        if not img_path.exists():
            continue

        import cv2
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        gt_boxes = load_yolo_labels(label_path, img_w, img_h)
        for b in gt_boxes:
            area = (b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1])
            all_gt_areas.append(area)
        per_image_records.append({"label_path": label_path, "img_path": img_path, "gt_boxes": gt_boxes, "img_w": img_w, "img_h": img_h})

    area_tertiles = np.percentile(all_gt_areas, [33.33, 66.67]) if all_gt_areas else [0, 0]

    def size_bucket_of(area):
        if area <= area_tertiles[0]:
            return "small"
        if area <= area_tertiles[1]:
            return "medium"
        return "large"

    density_values = [len(rec["gt_boxes"]) for rec in per_image_records]
    density_tertiles = np.percentile(density_values, [33.33, 66.67]) if density_values else [0, 0]

    def density_bucket_of(n):
        if n <= density_tertiles[0]:
            return "sparse"
        if n <= density_tertiles[1]:
            return "medium"
        return "dense"

    total_images = 0
    total_gt = 0
    total_pred = 0

    for rec in per_image_records:
        total_images += 1
        import cv2
        img = cv2.imread(str(rec["img_path"]))
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        pred_names = results.names
        preds = []
        for box in results.boxes:
            preds.append({
                "cls": int(box.cls[0].item()),
                "xyxy": tuple(float(v) for v in box.xyxy[0].tolist()),
                "conf": float(box.conf[0].item()),
            })
        total_pred += len(preds)
        total_gt += len(rec["gt_boxes"])

        gt_boxes = rec["gt_boxes"]
        density_bucket = density_bucket_of(len(gt_boxes))

        # Precompute "touching another GT box" per GT box (IoU>0 with any other GT box)
        gt_touching_flags = []
        for i, gb in enumerate(gt_boxes):
            touches_other = any(
                j != i and iou(gb["xyxy"], gt_boxes[j]["xyxy"]) > 0.0
                for j in range(len(gt_boxes))
            )
            gt_touching_flags.append(touches_other)

        matched_pred_indices = set()
        for i, gb in enumerate(gt_boxes):
            gt_cls_name = CLASS_NAMES[gb["cls"]]
            area = (gb["xyxy"][2] - gb["xyxy"][0]) * (gb["xyxy"][3] - gb["xyxy"][1])
            size_bucket = size_bucket_of(area)
            is_boundary = touches_boundary(gb["xyxy"], rec["img_w"], rec["img_h"])
            is_touching = gt_touching_flags[i]

            best_iou, best_idx = 0.0, -1
            for j, pb in enumerate(preds):
                if j in matched_pred_indices:
                    continue
                cur_iou = iou(gb["xyxy"], pb["xyxy"])
                if cur_iou > best_iou:
                    best_iou, best_idx = cur_iou, j

            matched = best_iou >= IOU_MATCH_THRESHOLD
            size_bucket_stats[size_bucket][1] += 1
            boundary_stats["boundary" if is_boundary else "interior"][1] += 1
            touching_stats["touching" if is_touching else "isolated"][1] += 1
            density_bucket_stats[density_bucket][1] += 1

            if matched:
                matched_pred_indices.add(best_idx)
                pred_cls_name = pred_names[preds[best_idx]["cls"]]
                class_confusion[gt_cls_name][pred_cls_name] += 1
                if pred_cls_name == gt_cls_name:
                    per_class_tp[gt_cls_name] += 1
                    size_bucket_stats[size_bucket][0] += 1
                    boundary_stats["boundary" if is_boundary else "interior"][0] += 1
                    touching_stats["touching" if is_touching else "isolated"][0] += 1
                    density_bucket_stats[density_bucket][0] += 1
                else:
                    per_class_fn[gt_cls_name] += 1  # localized but wrong class -> miss for its true class
                    per_class_fp[pred_cls_name] += 1  # and a spurious detection for the predicted class
            else:
                per_class_fn[gt_cls_name] += 1
                class_confusion[gt_cls_name]["missed"] += 1

        for j, pb in enumerate(preds):
            if j not in matched_pred_indices:
                per_class_fp[CLASS_NAMES[pb["cls"]]] += 1

    per_class_metrics = {}
    for cls_name in CLASS_NAMES:
        tp, fp, fn = per_class_tp[cls_name], per_class_fp[cls_name], per_class_fn[cls_name]
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None
        per_class_metrics[cls_name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    def recall_from_stats(stats_dict):
        return {k: (v[0] / v[1] if v[1] > 0 else None) for k, v in stats_dict.items()}

    report = {
        "confidence_threshold": CONF_THRESHOLD,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "total_images": total_images,
        "total_ground_truth_boxes": total_gt,
        "total_predicted_boxes": total_pred,
        "per_class_metrics": per_class_metrics,
        "class_confusion": {k: dict(v) for k, v in class_confusion.items()},
        "recall_by_size_tertile": recall_from_stats(size_bucket_stats),
        "size_tertile_counts": {k: v[1] for k, v in size_bucket_stats.items()},
        "area_tertile_thresholds_px2": list(area_tertiles),
        "recall_by_boundary_proximity": recall_from_stats(boundary_stats),
        "boundary_counts": {k: v[1] for k, v in boundary_stats.items()},
        "recall_by_touching_another_cell": recall_from_stats(touching_stats),
        "touching_counts": {k: v[1] for k, v in touching_stats.items()},
        "recall_by_image_density": recall_from_stats(density_bucket_stats),
        "density_tertile_counts": {k: v[1] for k, v in density_bucket_stats.items()},
        "density_tertile_thresholds": list(density_tertiles),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "detector_v1_detailed_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nDetailed report saved to {out_path}")


if __name__ == "__main__":
    main()
