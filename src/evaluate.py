"""Evaluate a trained checkpoint on the held-out test set.

Outputs (written to the same run_dir as the checkpoint):
  - test_metrics.json  : overall accuracy + macro/weighted P/R/F1
  - confusion_matrix.npy : (num_classes, num_classes)
  - per_class_report.csv : per-class precision/recall/F1/support
  - test_metrics.txt   : human-readable summary
  - test_curves.png    : train/val loss + acc curves from history.json
  - confusion_matrix.png : visualization (downsampled if >50 classes)

Usage
-----
    .venv/bin/python -m src.evaluate --ckpt results/baseline/best.pt
    .venv/bin/python -m src.evaluate --ckpt results/baseline/best.pt --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

from src import config as C
from src import data as D
from src.models.baseline_cnn import BaselineCNN


def build_model_from_args(args_dict: dict, num_classes: int):
    arch = args_dict.get("arch", "baseline_cnn")
    if arch == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, dropout=0.0)
    if arch == "efficientnet_b0":
        import timm
        return timm.create_model("efficientnet_b0", pretrained=False,
                                  num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")


def evaluate_loader(model, loader, device):
    model.eval()
    all_y, all_p, all_topk = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval"):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            topk = logits.topk(5, dim=1).indices
            all_y.append(y.cpu().numpy())
            all_p.append(pred.cpu().numpy())
            all_topk.append(topk.cpu().numpy())
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    topk = np.concatenate(all_topk)
    top5 = float(np.mean([yi in tki for yi, tki in zip(y, topk)]))
    return y, p, top5


def plot_history(run_dir: Path):
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        print("[evaluate] No history.json found, skipping curves plot.")
        return
    hist = json.loads(hist_path.read_text())
    if not hist:
        return
    epochs = [h["epoch"] + 1 for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in hist], label="train")
    axes[0].plot(epochs, [h["val_loss"]   for h in hist], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(epochs, [h["train_acc"] for h in hist], label="train")
    axes[1].plot(epochs, [h["val_acc"]   for h in hist], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[2].plot(epochs, [h["train_f1"] for h in hist], label="train")
    axes[2].plot(epochs, [h["val_f1"]   for h in hist], label="val")
    axes[2].set_title("Macro-F1"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=120)
    plt.close()
    print(f"[evaluate] Wrote {run_dir / 'training_curves.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    args = ap.parse_args()

    C.seed_everything()
    device = C.get_device()
    ckpt_path = Path(args.ckpt).resolve()
    run_dir = ckpt_path.parent
    print(f"[evaluate] ckpt={ckpt_path}")
    print(f"[evaluate] run_dir={run_dir}")
    print(f"[evaluate] split={args.split}, device={device}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = state.get("args", {})
    num_classes = D.get_num_classes()
    model = build_model_from_args(args_dict, num_classes).to(device)
    model.load_state_dict(state["model"])

    # Build eval-only transform that matches training
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    eval_t = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    _, val_loader, test_loader = D.build_loaders(
        batch_size=256, num_workers=0,
        train_transform=eval_t, eval_transform=eval_t,
    )
    if args.split == "val":
        loader = val_loader
    elif args.split == "train":
        train_loader, _, _ = D.build_loaders(
            batch_size=256, num_workers=0,
            train_transform=eval_t, eval_transform=eval_t,
        )
        loader = train_loader
    else:
        loader = test_loader

    y, p, top5 = evaluate_loader(model, loader, device)
    acc = accuracy_score(y, p)
    prec_m = precision_score(y, p, average="macro", zero_division=0)
    rec_m  = recall_score(y, p, average="macro", zero_division=0)
    f1_m   = f1_score(y, p, average="macro", zero_division=0)
    prec_w = precision_score(y, p, average="weighted", zero_division=0)
    rec_w  = recall_score(y, p, average="weighted", zero_division=0)
    f1_w   = f1_score(y, p, average="weighted", zero_division=0)

    metrics = {
        "split": args.split,
        "n_samples": int(len(y)),
        "accuracy": float(acc),
        "top5_accuracy": float(top5),
        "macro_precision": float(prec_m),
        "macro_recall":    float(rec_m),
        "macro_f1":        float(f1_m),
        "weighted_precision": float(prec_w),
        "weighted_recall":    float(rec_w),
        "weighted_f1":        float(f1_w),
    }
    (run_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2))

    print()
    print(f"=== {args.split.upper()} metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")

    # Per-class report
    label_names = D.get_label_names()
    report_dict = classification_report(
        y, p, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    rows = []
    for cls, vals in report_dict.items():
        if isinstance(vals, dict):
            rows.append({
                "label": cls,
                "precision": vals.get("precision", np.nan),
                "recall":    vals.get("recall", np.nan),
                "f1":        vals.get("f1-score", np.nan),
                "support":   vals.get("support", 0),
            })
    pd.DataFrame(rows).to_csv(run_dir / f"{args.split}_per_class.csv", index=False)
    print(f"[evaluate] Wrote per-class CSV.")

    # Confusion matrix
    cm = confusion_matrix(y, p, labels=list(range(num_classes)))
    np.save(run_dir / f"{args.split}_confusion_matrix.npy", cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="rocket_r", xticklabels=False, yticklabels=False, cbar=True)
    plt.title(f"Confusion Matrix ({args.split}, {num_classes} classes)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(run_dir / f"{args.split}_confusion_matrix.png", dpi=120)
    plt.close()
    print(f"[evaluate] Wrote {args.split}_confusion_matrix.png")

    plot_history(run_dir)


if __name__ == "__main__":
    main()
