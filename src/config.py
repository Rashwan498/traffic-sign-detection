"""Central configuration for the Vision term project.

All paths, hyperparameters, and seeds live here so every script reads the same
source of truth and experiments stay reproducible.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
ARCHIVE_ROOT = PROJECT_ROOT / "archive"

TRAIN_IMG_DIR = ARCHIVE_ROOT / "mtsd_fully_annotated_train_images" / "images"
VAL_IMG_DIR = ARCHIVE_ROOT / "mtsd_v2_fully_annotated_images.val.zip" / "images"
ANNOT_DIR = (
    ARCHIVE_ROOT
    / "mtsd_v2_fully_annotated_annotation.zip"
    / "mtsd_v2_fully_annotated"
    / "annotations"
)
SPLITS_DIR = (
    ARCHIVE_ROOT
    / "mtsd_v2_fully_annotated_annotation.zip"
    / "mtsd_v2_fully_annotated"
    / "splits"
)

CROPS_H5 = DATA_ROOT / "crops_cache.h5"
LABEL_MAP_JSON = DATA_ROOT / "label_map.json"
SPLITS_JSON = DATA_ROOT / "splits.json"
PREP_PROGRESS = DATA_ROOT / ".prep_progress.json"

# --------------------------------------------------------------------------
# Data parameters
# --------------------------------------------------------------------------
CROP_SIZE = 96
MIN_BBOX_SIDE_PX = 12          # discard sub-12px tiny boxes (unreliable signal)
MIN_SAMPLES_PER_CLASS = 30     # below this -> merged into 'other-sign'
OTHER_CLASS_NAME = "other-sign"
WRITE_CHUNK = 500              # flush HDF5 every N crops (resume granularity)

# --------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 50
LR = 3e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------
def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
