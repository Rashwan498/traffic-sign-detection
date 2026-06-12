"""Re-run the Phase-1 classical baseline (HOG -> SVM) on the SAME splits and
SAME class set used by the CNN, so the paper's comparison table is apples-to-apples.

Phase 1 pipeline (faithful to the colleague's notebook):
    crop -> resize 64x64 -> grayscale -> Gaussian blur 5x5 -> Canny edges
    -> HOG (9 orient, 8x8 cells, 2x2 blocks, L2-Hys) -> StandardScaler -> LinearSVC

Differences from the colleague's notebook (intentional, documented):
    - Uses our merged 326-class set (not 389) and our official splits
    - Uses LinearSVC (liblinear) instead of SVC(linear), ~50x faster on this many features
    - Reports macro/weighted P/R/F1 + top-1 accuracy, not just accuracy
    - No artificial class balancing -- uses class_weight='balanced' inside SVC

Outputs:
    results/phase1_rerun/test_metrics.json
    results/phase1_rerun/test_per_class.csv
    results/phase1_rerun/test_confusion_matrix.{npy,png}
    results/phase1_rerun/training_curves.png  # validation curve over C
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.feature import hog
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from src import config as C
from src import data as D

PHASE1_CROP = 64
RUN_DIR = C.RESULTS_ROOT / "phase1_rerun"


def hog_features(img_uint8_hwc: np.ndarray) -> np.ndarray:
    """Reproduces the colleague's pipeline exactly."""
    img = cv2.resize(img_uint8_hwc, (PHASE1_CROP, PHASE1_CROP), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    feats = hog(edges, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys")
    return feats.astype(np.float32)


def load_split(name: str):
    """Return (X_uint8, y) for the requested split."""
    splits = json.loads(C.SPLITS_JSON.read_text())
    idxs = splits["indices"][name]
    labels = np.array(splits["labels"][name], dtype=np.int64)
    with h5py.File(C.CROPS_H5, "r") as f:
        # batched read for speed
        imgs = np.stack([f["crops"][i] for i in tqdm(idxs, desc=f"loading {name}")])
    return imgs, labels


def extract_hog_batch(imgs: np.ndarray, desc: str) -> np.ndarray:
    return np.stack([hog_features(img) for img in tqdm(imgs, desc=desc)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=1.0, help="LinearSVC regularization (default 1.0).")
    ap.add_argument("--tune", action="store_true", help="Try C in [0.1, 1, 10] on val and pick best.")
    args = ap.parse_args()

    C.seed_everything()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    X_tr_img, y_tr = load_split("train")
    X_va_img, y_va = load_split("val")
    X_te_img, y_te = load_split("test")
    print(f"[phase1] train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}  classes={D.get_num_classes()}")

    # 2. HOG features (cached on disk so re-runs are cheap)
    cache = RUN_DIR / "hog_features.npz"
    if cache.exists():
        d = np.load(cache)
        X_tr, X_va, X_te = d["X_tr"], d["X_va"], d["X_te"]
        print(f"[phase1] Loaded cached HOG features ({X_tr.shape[1]} dims).")
    else:
        t0 = time.time()
        X_tr = extract_hog_batch(X_tr_img, "HOG train")
        X_va = extract_hog_batch(X_va_img, "HOG val")
        X_te = extract_hog_batch(X_te_img, "HOG test")
        print(f"[phase1] HOG extracted in {(time.time()-t0)/60:.1f} min. dim={X_tr.shape[1]}")
        np.savez_compressed(cache, X_tr=X_tr, X_va=X_va, X_te=X_te)

    # 3. Scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # 4. Train
    cv_results = []
    if args.tune:
        candidates = [0.1, 1.0, 10.0]
        best = None
        for c in candidates:
            t0 = time.time()
            clf = LinearSVC(C=c, max_iter=5000, class_weight="balanced", dual="auto")
            clf.fit(X_tr_s, y_tr)
            val_acc = accuracy_score(y_va, clf.predict(X_va_s))
            cv_results.append({"C": c, "val_acc": float(val_acc), "fit_sec": time.time()-t0})
            print(f"[phase1] C={c}: val_acc={val_acc:.4f} ({time.time()-t0:.0f}s)")
            if best is None or val_acc > best[0]:
                best = (val_acc, c, clf)
        val_acc, chosen_C, clf = best
        print(f"[phase1] Chose C={chosen_C} (val acc={val_acc:.4f})")
    else:
        chosen_C = args.C
        t0 = time.time()
        clf = LinearSVC(C=chosen_C, max_iter=5000, class_weight="balanced", dual="auto")
        clf.fit(X_tr_s, y_tr)
        print(f"[phase1] Trained in {time.time()-t0:.0f}s with C={chosen_C}")

    # 5. Test evaluation
    y_pred = clf.predict(X_te_s)
    acc = accuracy_score(y_te, y_pred)
    metrics = {
        "C": chosen_C,
        "accuracy": float(acc),
        "macro_precision": float(precision_score(y_te, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_te, y_pred,    average="macro", zero_division=0)),
        "macro_f1":        float(f1_score(y_te, y_pred,        average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
        "weighted_recall":    float(recall_score(y_te, y_pred,    average="weighted", zero_division=0)),
        "weighted_f1":        float(f1_score(y_te, y_pred,        average="weighted", zero_division=0)),
        "n_samples": int(len(y_te)),
        "n_classes": D.get_num_classes(),
        "cv_results": cv_results,
    }
    (RUN_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== Phase 1 (rerun) TEST metrics ===")
    for k, v in metrics.items():
        if k != "cv_results":
            print(f"  {k:<22} {v}")

    # 6. Per-class report
    label_names = D.get_label_names()
    rep = classification_report(y_te, y_pred, target_names=label_names,
                                output_dict=True, zero_division=0)
    rows = [{"label": k, **{kk: v[kk] for kk in ["precision","recall","f1-score","support"]}}
            for k, v in rep.items() if isinstance(v, dict)]
    pd.DataFrame(rows).to_csv(RUN_DIR / "test_per_class.csv", index=False)
    # 7. Confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=list(range(D.get_num_classes())))
    np.save(RUN_DIR / "test_confusion_matrix.npy", cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="rocket_r", xticklabels=False, yticklabels=False, cbar=True)
    plt.title(f"Phase 1 rerun -- Confusion matrix ({D.get_num_classes()} classes)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "test_confusion_matrix.png", dpi=120)
    plt.close()
    print(f"[phase1] All outputs written to {RUN_DIR}")


if __name__ == "__main__":
    main()
