"""
Train the whole-smear cell detector (spec section 4/19/20/22) on TXL-PBC,
which is already released in a ready-to-use YOLO format (images/labels split
into train/val/test, classes: WBC, RBC, Platelets) -- see
dataset_feasibility_audit.md section 2/7 for provenance.

Architecture: YOLO-nano family, chosen in reports/architecture_decision.md
section 3 for CPU-inference speed (this dev machine has no CUDA GPU).

This writes a real, absolute-path data.yaml (TXL-PBC's own file uses relative
paths that only resolve correctly from inside its own folder) and then runs
an actual Ultralytics training job. Metrics reported by Ultralytics
(precision/recall/mAP50/mAP50-95 per class) are copied verbatim into
evaluation/ -- nothing here is invented.

Usage:
    python scripts/train_detector.py --epochs 20 --model yolov8n.pt --imgsz 416
"""
import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TXL_ROOT = REPO_ROOT / "datasets" / "raw" / "txl_pbc" / "extracted" / "TXL-PBC"
CONFIGS_DIR = REPO_ROOT / "configs"
EVAL_DIR = REPO_ROOT / "evaluation"
MODELS_DIR = REPO_ROOT / "models"


def write_absolute_data_yaml() -> Path:
    yaml_path = CONFIGS_DIR / "txl_pbc_detection.yaml"
    content = (
        f"path: {TXL_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 3\n"
        "names: ['WBC', 'RBC', 'Platelet']\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help works even before install

    data_yaml = write_absolute_data_yaml()
    print(f"Using data config: {data_yaml}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(REPO_ROOT / "experiments"),
        name="txl_pbc_detector",
        exist_ok=True,
        verbose=True,
    )

    metrics = model.val(data=str(data_yaml), split="test", device="cpu")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_weights = REPO_ROOT / "experiments" / "txl_pbc_detector" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, MODELS_DIR / "detector_v1.pt")
        print(f"Best weights copied to {MODELS_DIR / 'detector_v1.pt'}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    eval_report = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "dataset": "txl_pbc",
        "classes": ["WBC", "RBC", "Platelet"],
        "test_set_metrics": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision_per_class": metrics.box.p.tolist() if hasattr(metrics.box.p, "tolist") else list(metrics.box.p),
            "recall_per_class": metrics.box.r.tolist() if hasattr(metrics.box.r, "tolist") else list(metrics.box.r),
        },
        "hardware": "CPU-only (no CUDA GPU available on dev machine, see architecture_decision.md section 1)",
        "iou_threshold_note": "Ultralytics default validation IoU threshold (0.5:0.95 sweep for mAP50-95, 0.5 for mAP50) -- see spec section 22 requirement to document IoU/confidence/NMS settings",
    }
    with open(EVAL_DIR / "detector_v1_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    print(json.dumps(eval_report["test_set_metrics"], indent=2))
    print(f"Evaluation report saved to {EVAL_DIR / 'detector_v1_evaluation.json'}")


if __name__ == "__main__":
    main()
