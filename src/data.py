"""PyTorch Dataset + DataLoader factories backed by the HDF5 cache.

The cache (data/crops_cache.h5) stores all 190k raw crops as uint8 HWC. The
splits file (data/splits.json) maps a 'train' / 'val' / 'test' name to the
list of HDF5 indices that belong to that split, plus the matching integer
labels.

This module exposes:
    CropsDataset(split, transform)  - torch Dataset
    build_loaders(...)              - one-call (train, val, test) DataLoaders
    get_num_classes()               - reads label_map.json
    get_class_weights()             - torch.Tensor on CPU
"""
from __future__ import annotations

import json
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import config as C


# --------------------------------------------------------------------------
# ImageNet normalization (used for transfer learning later). For from-scratch
# training we still use these stats -- they're a fine baseline for natural
# RGB images and keep the preprocessing identical across experiments.
# --------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_float_normalized(img_uint8_hwc: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 CHW, ImageNet-normalized."""
    img = img_uint8_hwc.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


class CropsDataset(Dataset):
    """Reads crops on demand from the shared HDF5 cache.

    HDF5 is opened lazily per worker process (h5py is not fork-safe).
    """

    def __init__(self, split: str, transform: Callable | None = None):
        assert split in ("train", "val", "test"), split
        splits = json.loads(C.SPLITS_JSON.read_text())
        self.indices = splits["indices"][split]
        self.labels = splits["labels"][split]
        self.split = split
        self.transform = transform
        self._h5 = None  # opened per worker

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(C.CROPS_H5, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._ensure_open()
        cache_idx = self.indices[idx]
        img = self._h5["crops"][cache_idx]  # uint8 HWC
        label = int(self.labels[idx])

        if self.transform is not None:
            # Albumentations transforms take HWC uint8 and return a dict.
            img = self.transform(image=img)["image"]
            # Albumentations ToTensorV2 returns CHW float; if no transform was
            # used at all we fall back to manual normalize below.
            if isinstance(img, np.ndarray):
                img = to_float_normalized(img)
                img = torch.from_numpy(img)
        else:
            img = torch.from_numpy(to_float_normalized(img))
        return img, label


def build_loaders(
    batch_size: int = C.BATCH_SIZE,
    num_workers: int = C.NUM_WORKERS,
    train_transform: Callable | None = None,
    eval_transform: Callable | None = None,
):
    """Return (train_loader, val_loader, test_loader)."""
    train_ds = CropsDataset("train", transform=train_transform)
    val_ds = CropsDataset("val", transform=eval_transform)
    test_ds = CropsDataset("test", transform=eval_transform)

    # On macOS, multiprocessing workers can be flaky with h5py; we keep
    # num_workers configurable and use persistent_workers when > 0.
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # pin_memory doesn't help on MPS
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


def get_num_classes() -> int:
    return int(json.loads(C.LABEL_MAP_JSON.read_text())["num_classes"])


def get_class_weights() -> torch.Tensor:
    return torch.from_numpy(np.load(C.DATA_ROOT / "class_weights.npy"))


def get_label_names() -> list[str]:
    lm = json.loads(C.LABEL_MAP_JSON.read_text())
    int_to_label = lm["int_to_label"]
    return [int_to_label[str(i)] for i in range(lm["num_classes"])]
