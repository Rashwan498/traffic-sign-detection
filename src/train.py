"""Resumable training loop for the Vision project.

Features
--------
- Saves a full checkpoint (model+optim+sched+RNG+history) after every epoch.
- `latest.pt` symlink always points at the most recent checkpoint.
- `best.pt` is updated whenever val macro-F1 improves.
- Ctrl+C is intercepted: finishes the current batch, writes a checkpoint,
  then exits cleanly. Resume with `--resume <path-to-latest.pt>`.
- One CLI for multiple experiments (baseline, augmentation, transfer):
  pick architecture and augmentation profile via flags.

Usage
-----
    # Stage 1 baseline (no aug, custom CNN)
    .venv/bin/python -m src.train --name baseline --arch baseline_cnn

    # Stage 2 E1 (augmentation)
    .venv/bin/python -m src.train --name aug --arch baseline_cnn --augment

    # Stage 2 E2 (transfer learning, EfficientNet-B0)
    .venv/bin/python -m src.train --name transfer --arch efficientnet_b0 --augment

    # Resume any run
    .venv/bin/python -m src.train --resume results/baseline/latest.pt
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src import config as C
from src import data as D
from src.models.baseline_cnn import BaselineCNN, count_parameters


# --------------------------------------------------------------------------
# Graceful Ctrl+C
# --------------------------------------------------------------------------
_STOP = False
def _handle_sigint(signum, frame):
    global _STOP
    if _STOP:
        print("\n[train] Second Ctrl+C -- hard exit.", flush=True)
        sys.exit(130)
    _STOP = True
    print("\n[train] Ctrl+C caught. Finishing the current batch, saving "
          "checkpoint, then exiting cleanly. Press Ctrl+C again to force.",
          flush=True)
signal.signal(signal.SIGINT, _handle_sigint)


# --------------------------------------------------------------------------
# Augmentation profiles (Albumentations)
# --------------------------------------------------------------------------
def get_transforms(augment: bool):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if augment:
        train_t = A.Compose([
            A.HorizontalFlip(p=0.0),  # signs are direction-sensitive, no horizontal flip
            A.Rotate(limit=15, border_mode=0, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(max_holes=1, max_height=24, max_width=24,
                            min_holes=1, min_height=8, min_width=8,
                            fill_value=0, p=0.3),
            A.Affine(translate_percent=(-0.06, 0.06), scale=(0.92, 1.08), p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        train_t = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

    eval_t = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return train_t, eval_t


# --------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------
def build_model(arch: str, num_classes: int):
    if arch == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, dropout=0.3)
    if arch == "efficientnet_b0":
        import timm
        return timm.create_model("efficientnet_b0", pretrained=True,
                                  num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")


# --------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------
def save_ckpt(path: Path, model, optimizer, scheduler, scaler, epoch, best_f1,
              history, args_dict):
    state = {
        "epoch": epoch,
        "best_f1": best_f1,
        "history": history,
        "args": args_dict,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------
# Train / eval epochs
# --------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              desc=""):
    train_mode = optimizer is not None
    model.train(train_mode)
    total, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        if _STOP and train_mode:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")
    y = np.concatenate(all_y) if all_y else np.array([])
    p = np.concatenate(all_p) if all_p else np.array([])
    macro_f1 = f1_score(y, p, average="macro", zero_division=0) if len(y) else 0.0
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "macro_f1": float(macro_f1),
        "n": total,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="baseline",
                    help="Run name; results go to results/<name>/")
    ap.add_argument("--arch", type=str, default="baseline_cnn",
                    choices=["baseline_cnn", "efficientnet_b0"])
    ap.add_argument("--augment", action="store_true",
                    help="Enable training-time data augmentation.")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--label-smoothing", type=float, default=C.LABEL_SMOOTHING)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Disable class weighting in the loss.")
    ap.add_argument("--resume", type=str, default="",
                    help="Path to a checkpoint to resume from.")
    args = ap.parse_args()

    C.seed_everything()
    device = C.get_device()
    print(f"[train] Device: {device}")

    # Output dir
    run_dir = C.RESULTS_ROOT / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"
    args_path = run_dir / "args.json"
    args_path.write_text(json.dumps(vars(args), indent=2))

    # Data
    train_t, eval_t = get_transforms(args.augment)
    train_loader, val_loader, _ = D.build_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        train_transform=train_t,
        eval_transform=eval_t,
    )
    num_classes = D.get_num_classes()
    print(f"[train] num_classes={num_classes}, "
          f"train batches={len(train_loader)}, val batches={len(val_loader)}")

    # Model
    model = build_model(args.arch, num_classes).to(device)
    print(f"[train] arch={args.arch}, params={count_parameters(model):,}")

    # Loss
    weights = None if args.no_class_weights else D.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=args.label_smoothing)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           steps_per_epoch=steps_per_epoch,
                           epochs=args.epochs, pct_start=0.1,
                           anneal_strategy="cos")
    scaler = None  # MPS does not currently benefit from torch.amp.GradScaler

    # State
    start_epoch = 0
    best_f1 = -1.0
    history = []

    # Resume
    resume_path = Path(args.resume) if args.resume else None
    if resume_path and resume_path.exists():
        print(f"[train] Resuming from {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if state.get("rng"):
            torch.set_rng_state(state["rng"]["torch"])
            np.random.set_state(state["rng"]["numpy"])
        start_epoch = state["epoch"] + 1
        best_f1 = state["best_f1"]
        history = state.get("history", [])
        print(f"[train] Resumed at epoch {start_epoch}, best_f1={best_f1:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if _STOP:
            print("[train] Stop requested before next epoch. Exiting.")
            break
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device,
                       optimizer=optimizer, scheduler=scheduler,
                       desc=f"E{epoch+1}/{args.epochs} train")
        val = run_epoch(model, val_loader, criterion, device,
                        desc=f"E{epoch+1}/{args.epochs}   val")
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": tr["loss"], "train_acc": tr["acc"], "train_f1": tr["macro_f1"],
            "val_loss":   val["loss"], "val_acc":   val["acc"], "val_f1":   val["macro_f1"],
            "dt_sec": dt,
        }
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2))
        print(f"[E{epoch+1:>2}/{args.epochs}] "
              f"loss {tr['loss']:.3f}/{val['loss']:.3f}  "
              f"acc {tr['acc']:.3f}/{val['acc']:.3f}  "
              f"f1 {tr['macro_f1']:.3f}/{val['macro_f1']:.3f}  "
              f"lr {lr_now:.1e}  {dt:.0f}s")
        # Save checkpoint
        save_ckpt(latest_path, model, optimizer, scheduler, scaler,
                  epoch, max(best_f1, val["macro_f1"]), history, vars(args))
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            save_ckpt(best_path, model, optimizer, scheduler, scaler,
                      epoch, best_f1, history, vars(args))
            print(f"          new best val macro-F1: {best_f1:.4f}  -> saved best.pt")
        if _STOP:
            print("[train] Stop after epoch. Checkpoint saved. Exit.")
            break

    print(f"[train] Done. Best val macro-F1: {best_f1:.4f}")
    print(f"[train] Best ckpt: {best_path}  |  Latest ckpt: {latest_path}")


if __name__ == "__main__":
    main()
