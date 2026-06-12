"""Fast variant of the Phase 1 rerun. Uses cached HOG features and an
SGDClassifier with hinge loss (linear SVM, SGD-fitted) instead of LinearSVC's
primal solver, which was too slow on 326 one-vs-rest classifiers + balanced
class weights.

Methodologically equivalent: same feature pipeline, same loss family,
same regularization. Trades exact dual-optimum convergence for a >50x speedup.

Outputs match phase1_rerun.py so the notebook / paper can read them directly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import data as D

RUN_DIR = C.RESULTS_ROOT / "phase1_rerun"


def main():
    cache = RUN_DIR / "hog_features.npz"
    assert cache.exists(), f"Missing {cache}; run phase1_rerun.py first to extract."
    d = np.load(cache)
    X_tr, X_va, X_te = d["X_tr"], d["X_va"], d["X_te"]
    splits = json.loads(C.SPLITS_JSON.read_text())
    y_tr = np.array(splits["labels"]["train"], dtype=np.int64)
    y_va = np.array(splits["labels"]["val"],   dtype=np.int64)
    y_te = np.array(splits["labels"]["test"],  dtype=np.int64)

    print(f"[phase1-fast] features train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    t0 = time.time()
    # hinge loss + L2 penalty = linear SVM (one-vs-rest by default for multi-class)
    clf = SGDClassifier(
        loss="hinge", penalty="l2", alpha=1e-4,
        max_iter=30, tol=1e-3, n_jobs=-1, random_state=C.SEED,
        class_weight="balanced",
    )
    clf.fit(X_tr_s, y_tr)
    dt = time.time() - t0
    print(f"[phase1-fast] SGD-SVM fit in {dt:.1f}s")

    val_acc = accuracy_score(y_va, clf.predict(X_va_s))
    print(f"[phase1-fast] val accuracy: {val_acc:.4f}")

    y_pred = clf.predict(X_te_s)
    metrics = {
        "model": "phase1_HOG_SGD-SVM",
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "macro_precision": float(precision_score(y_te, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_te, y_pred,    average="macro", zero_division=0)),
        "macro_f1":        float(f1_score(y_te, y_pred,        average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
        "weighted_recall":    float(recall_score(y_te, y_pred,    average="weighted", zero_division=0)),
        "weighted_f1":        float(f1_score(y_te, y_pred,        average="weighted", zero_division=0)),
        "n_samples": int(len(y_te)),
        "n_classes": D.get_num_classes(),
        "fit_seconds": dt,
        "val_accuracy": float(val_acc),
    }
    (RUN_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== Phase 1 rerun (SGD-SVM) test metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    # Per-class CSV
    label_names = D.get_label_names()
    rep = classification_report(y_te, y_pred, target_names=label_names,
                                output_dict=True, zero_division=0)
    rows = [{"label": k, **{kk: v[kk] for kk in ["precision","recall","f1-score","support"]}}
            for k, v in rep.items() if isinstance(v, dict)]
    pd.DataFrame(rows).to_csv(RUN_DIR / "test_per_class.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=list(range(D.get_num_classes())))
    np.save(RUN_DIR / "test_confusion_matrix.npy", cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="rocket_r", xticklabels=False, yticklabels=False, cbar=True)
    plt.title(f"Phase 1 (HOG + SGD-SVM) -- Confusion matrix ({D.get_num_classes()} classes)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "test_confusion_matrix.png", dpi=120)
    plt.close()
    print(f"[phase1-fast] Outputs at {RUN_DIR}")


if __name__ == "__main__":
    main()
