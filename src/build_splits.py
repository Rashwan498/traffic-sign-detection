"""Stage 0b -- build the clean class set and train/val/test splits.

Reads data/crops_cache.h5 and writes:
  - data/label_map.json  : {label_str -> int, int -> label_str, num_classes, ...}
  - data/splits.json     : {train: [indices], val: [indices], test: [indices]}
  - data/class_weights.npy : float32 vector for CrossEntropyLoss

Policy (Option A, chosen 2026-05-15):
  1. Drop the "other-sign" catch-all class entirely -- it is noise, not a class.
  2. Drop any class with < MIN_SAMPLES_PER_CLASS (default 30) samples in the
     MTSD-train portion. These classes are unlearnable from this little data.
  3. Build splits:
       test  = all surviving crops whose source split is mtsd_val
       train = 90% of surviving crops from mtsd_train (stratified by class)
       val   = 10% of surviving crops from mtsd_train (stratified by class)
     Stratification uses sklearn.model_selection.StratifiedShuffleSplit so
     every class is present in both train and val.
  4. Class weights = (1 / sqrt(count))-normalized so the mean weight is 1.0.

This script is idempotent: re-running overwrites the three output files.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src import config as C

DROP_CLASS = "other-sign"
VAL_FRAC = 0.10  # 10% of mtsd_train -> our validation set


def main():
    C.seed_everything()

    with h5py.File(C.CROPS_H5, "r") as f:
        raw_labels = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in f["raw_label"][:]]
        )
        splits = np.array(
            [s.decode() if isinstance(s, bytes) else s for s in f["split"][:]]
        )
    n_total = len(raw_labels)
    print(f"[build_splits] Loaded {n_total:,} crops from cache.")

    # ---- step 1: drop other-sign ----
    keep_mask = raw_labels != DROP_CLASS
    print(f"[build_splits] Dropping '{DROP_CLASS}': {n_total - keep_mask.sum():,} crops removed.")
    raw_labels = raw_labels[keep_mask]
    splits = splits[keep_mask]
    cache_idx = np.where(keep_mask)[0]  # original index in HDF5 -> we keep this mapping
    print(f"[build_splits] {len(raw_labels):,} crops remain across {len(set(raw_labels)):,} classes.")

    # ---- step 2: drop classes with < MIN_SAMPLES_PER_CLASS in mtsd_train ----
    train_mask = splits == "mtsd_train"
    train_counts = Counter(raw_labels[train_mask].tolist())
    keep_classes = sorted(
        lbl for lbl, c in train_counts.items() if c >= C.MIN_SAMPLES_PER_CLASS
    )
    dropped_classes = [
        (lbl, c) for lbl, c in train_counts.items() if c < C.MIN_SAMPLES_PER_CLASS
    ]
    print(
        f"[build_splits] Keeping {len(keep_classes):,} classes "
        f"(>= {C.MIN_SAMPLES_PER_CLASS} train samples). "
        f"Dropping {len(dropped_classes):,} rare classes "
        f"({sum(c for _, c in dropped_classes):,} train + "
        f"corresponding val crops)."
    )

    keep_set = set(keep_classes)
    class_keep_mask = np.array([l in keep_set for l in raw_labels])
    raw_labels = raw_labels[class_keep_mask]
    splits = splits[class_keep_mask]
    cache_idx = cache_idx[class_keep_mask]
    print(f"[build_splits] Final dataset: {len(raw_labels):,} crops, "
          f"{len(keep_classes):,} classes.")

    # ---- step 3: label map ----
    label_to_int = {lbl: i for i, lbl in enumerate(keep_classes)}
    int_to_label = {i: lbl for lbl, i in label_to_int.items()}
    y = np.array([label_to_int[l] for l in raw_labels], dtype=np.int32)

    # ---- step 4: splits ----
    # test = all mtsd_val crops; train/val split = stratified 90/10 of mtsd_train
    test_pos = np.where(splits == "mtsd_val")[0]
    trval_pos = np.where(splits == "mtsd_train")[0]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=C.SEED)
    trval_y = y[trval_pos]
    (train_in_trval, val_in_trval), = sss.split(np.zeros(len(trval_pos)), trval_y)
    train_pos = trval_pos[train_in_trval]
    val_pos = trval_pos[val_in_trval]

    # Splits are stored as cache (HDF5) indices, so the dataset can read directly.
    split_indices = {
        "train": cache_idx[train_pos].astype(int).tolist(),
        "val":   cache_idx[val_pos].astype(int).tolist(),
        "test":  cache_idx[test_pos].astype(int).tolist(),
    }
    split_labels_int = {
        "train": y[train_pos].astype(int).tolist(),
        "val":   y[val_pos].astype(int).tolist(),
        "test":  y[test_pos].astype(int).tolist(),
    }

    print(f"[build_splits] Split sizes: "
          f"train={len(split_indices['train']):,}, "
          f"val={len(split_indices['val']):,}, "
          f"test={len(split_indices['test']):,}")

    # ---- step 5: class weights (sqrt-inverse-frequency, mean-normalized) ----
    train_class_counts = np.bincount(y[train_pos], minlength=len(keep_classes)).astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(train_class_counts, 1))
    weights = inv_sqrt / inv_sqrt.mean()
    weights = weights.astype(np.float32)
    print(f"[build_splits] Class weights computed. "
          f"min={weights.min():.3f}  max={weights.max():.3f}  mean={weights.mean():.3f}")

    # ---- step 6: write outputs ----
    C.LABEL_MAP_JSON.write_text(json.dumps({
        "num_classes": len(keep_classes),
        "label_to_int": label_to_int,
        "int_to_label": {str(k): v for k, v in int_to_label.items()},
        "dropped_other_sign_count": int((np.array([s.decode() if isinstance(s, bytes) else s
                                                   for s in h5py.File(C.CROPS_H5, 'r')['raw_label'][:]]) == DROP_CLASS).sum()),
        "dropped_rare_classes": [
            {"label": lbl, "train_count": cnt} for lbl, cnt in sorted(dropped_classes, key=lambda x: -x[1])
        ],
        "min_samples_per_class": C.MIN_SAMPLES_PER_CLASS,
    }, indent=2))
    print(f"[build_splits] Wrote {C.LABEL_MAP_JSON}")

    C.SPLITS_JSON.write_text(json.dumps({
        "seed": C.SEED,
        "val_frac_of_train": VAL_FRAC,
        "indices": split_indices,
        "labels": split_labels_int,
    }))
    print(f"[build_splits] Wrote {C.SPLITS_JSON}")

    np.save(C.DATA_ROOT / "class_weights.npy", weights)
    print(f"[build_splits] Wrote {C.DATA_ROOT / 'class_weights.npy'}")

    # ---- summary stats ----
    print()
    print("=== Class-frequency summary (train split) ===")
    counts_sorted = np.sort(train_class_counts)[::-1]
    print(f"  min/median/max samples per class: "
          f"{int(counts_sorted.min())} / "
          f"{int(np.median(counts_sorted))} / "
          f"{int(counts_sorted.max())}")
    print(f"  Top 5 most frequent classes (train):")
    top5 = np.argsort(-train_class_counts)[:5]
    for c in top5:
        print(f"     {int(train_class_counts[c]):>6,}  {int_to_label[int(c)]}")


if __name__ == "__main__":
    main()
