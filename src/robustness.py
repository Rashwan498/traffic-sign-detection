"""Stage 2 / E3 -- evaluate a checkpoint under each (distortion, severity).

Iterates over the test set; for each distortion/severity, applies the
distortion to every image and records accuracy + macro-F1. Outputs:

  results/<run>/robustness.csv     : long-form (distortion, level, acc, f1)
  results/<run>/robustness.png     : line plot, acc vs severity per distortion

Usage
-----
    .venv/bin/python -m src.robustness --ckpt results/baseline/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src import config as C
from src import data as D
from src.distortions import DISTORTION_NAMES, SEVERITY_LEVELS, apply_distortion
from src.models.baseline_cnn import BaselineCNN


def build_model_from_args(args_dict: dict, num_classes: int):
    arch = args_dict.get("arch", "baseline_cnn")
    if arch == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, dropout=0.0)
    if arch == "efficientnet_b0":
        import timm
        return timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")


def normalize_batch(imgs_uint8_nhwc: np.ndarray) -> torch.Tensor:
    """uint8 NHWC -> normalized float32 tensor NCHW on CPU."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = imgs_uint8_nhwc.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C.seed_everything()
    device = C.get_device()
    ckpt_path = Path(args.ckpt).resolve()
    run_dir = ckpt_path.parent
    print(f"[robustness] ckpt={ckpt_path}")
    print(f"[robustness] device={device}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = state.get("args", {})
    num_classes = D.get_num_classes()
    model = build_model_from_args(args_dict, num_classes).to(device).eval()
    model.load_state_dict(state["model"])

    # Load full test set into RAM (~8.6k * 96*96*3 = ~240 MB uint8)
    splits = json.loads(C.SPLITS_JSON.read_text())
    test_idx = splits["indices"]["test"]
    test_y = np.array(splits["labels"]["test"], dtype=np.int64)
    with h5py.File(C.CROPS_H5, "r") as f:
        test_imgs = np.stack([f["crops"][i] for i in tqdm(test_idx, desc="loading test set")])
    print(f"[robustness] test set: {test_imgs.shape} ({test_imgs.dtype})")

    rows = []
    for d_name in DISTORTION_NAMES:
        for lvl in SEVERITY_LEVELS:
            print(f"[robustness] {d_name} @ level {lvl} ...")
            # Apply distortion
            if lvl == 0:
                imgs_d = test_imgs
            else:
                rng = np.random.default_rng(args.seed + lvl)
                imgs_d = np.stack([
                    apply_distortion(img, d_name, lvl, rng=rng) for img in test_imgs
                ])
            # Predict in batches
            preds = np.empty(len(test_y), dtype=np.int64)
            with torch.no_grad():
                for s in range(0, len(test_y), args.batch_size):
                    e = min(s + args.batch_size, len(test_y))
                    xb = normalize_batch(imgs_d[s:e]).to(device)
                    pb = model(xb).argmax(1).cpu().numpy()
                    preds[s:e] = pb
            acc = accuracy_score(test_y, preds)
            f1m = f1_score(test_y, preds, average="macro", zero_division=0)
            rows.append({"distortion": d_name, "severity": lvl,
                         "accuracy": float(acc), "macro_f1": float(f1m)})
            print(f"     -> acc={acc:.4f}, f1={f1m:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "robustness.csv", index=False)
    print(f"[robustness] Wrote {run_dir / 'robustness.csv'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for d_name in DISTORTION_NAMES:
        sub = df[df["distortion"] == d_name]
        axes[0].plot(sub["severity"], sub["accuracy"], marker="o", label=d_name)
        axes[1].plot(sub["severity"], sub["macro_f1"],  marker="o", label=d_name)
    for ax, title in zip(axes, ["Accuracy", "Macro-F1"]):
        ax.set_xlabel("Severity (0=clean, 3=severe)")
        ax.set_ylabel(title); ax.set_title(f"{title} vs distortion severity")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "robustness.png", dpi=120)
    plt.close()
    print(f"[robustness] Wrote {run_dir / 'robustness.png'}")


if __name__ == "__main__":
    main()
