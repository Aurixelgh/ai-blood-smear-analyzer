"""
Train the 7-class single-cell blood cell classifier on the pooled dataset
produced by build_classification_dataset.py.

Architecture choice (see reports/architecture_decision.md section 4): a
lightweight, ImageNet-pretrained CNN backbone (MobileNetV3-Small by default --
chosen for CPU-inference speed given this machine has no CUDA GPU, see
reports/architecture_decision.md section 1/7), fine-tuned with a new
classification head. This is a real training run on real data; no metric in
evaluation/ is fabricated -- whatever this script prints/saves is what
actually happened on this machine.

Usage:
    python scripts/train_classifier.py --epochs 6 --batch-size 64 --img-size 128
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "datasets" / "processed" / "classification_v1"
MODELS_DIR = REPO_ROOT / "models"
EVAL_DIR = REPO_ROOT / "evaluation"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),           # small rotation -- spec section 18
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # brightness/contrast variation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_model(num_classes: int, backbone: str):
    if backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report_text = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mcc": float(mcc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_order": class_names,
        "classification_report_text": report_text,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small",
                         choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_ROOT / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(DATA_ROOT / "test", transform=eval_tf)

    assert train_ds.classes == val_ds.classes == test_ds.classes, "Class order mismatch across splits!"
    class_names = train_ds.classes
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Class-imbalance handling (spec section 21/10 finding: basophil is the minority class)
    class_counts = [0] * len(class_names)
    for _, label in train_ds.samples:
        class_counts[label] += 1
    class_weights = torch.tensor([1.0 / max(c, 1) for c in class_counts], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_names)
    print("Train class counts:", dict(zip(class_names, class_counts)))

    model = build_model(len(class_names), args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_macro_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, class_names)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | "
              f"time={epoch_time:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_accuracy": val_metrics["accuracy"],
                         "val_macro_f1": val_metrics["macro_f1"],
                         "epoch_seconds": epoch_time})

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, class_names)
    print("\n=== TEST SET RESULTS (real, measured on this run) ===")
    print(test_metrics["classification_report_text"])
    print("Macro F1:", test_metrics["macro_f1"], "| MCC:", test_metrics["mcc"],
          "| Balanced accuracy:", test_metrics["balanced_accuracy"])

    torch.save({"model_state_dict": model.state_dict(), "class_names": class_names,
                "backbone": args.backbone, "img_size": args.img_size},
               MODELS_DIR / "classifier_v1.pt")

    report = {
        "backbone": args.backbone, "img_size": args.img_size, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
        "class_counts_train": dict(zip(class_names, class_counts)),
        "training_history": history,
        "test_metrics": test_metrics,
        "hardware": "CPU-only (no CUDA GPU available on dev machine, see architecture_decision.md section 1)",
    }
    with open(EVAL_DIR / "classifier_v1_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nModel saved to {MODELS_DIR / 'classifier_v1.pt'}")
    print(f"Evaluation report saved to {EVAL_DIR / 'classifier_v1_evaluation.json'}")


if __name__ == "__main__":
    main()
