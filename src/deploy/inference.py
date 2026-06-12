"""Shared inference utility used by all deployment targets (Gradio, FastAPI,
ONNX/CoreML smoke tests).

A single class :class:`SignClassifier` that takes a checkpoint path and
exposes ``.predict(image)`` returning the top-k (label, probability) pairs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from src import config as C
from src import data as D
from src.models.baseline_cnn import BaselineCNN


def _build_model_from_args(args_dict: dict, num_classes: int):
    arch = args_dict.get("arch", "baseline_cnn")
    if arch == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, dropout=0.0)
    if arch == "efficientnet_b0":
        import timm
        return timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")


def _preprocess(img: Image.Image) -> torch.Tensor:
    """PIL -> (1, 3, 96, 96) float32 ImageNet-normalized."""
    img = img.convert("RGB").resize((C.CROP_SIZE, C.CROP_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None]  # NCHW
    return torch.from_numpy(arr)


@dataclass
class Prediction:
    label: str
    probability: float

    def asdict(self) -> dict:
        return {"label": self.label, "probability": self.probability}


class SignClassifier:
    """A thin wrapper around a trained checkpoint."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        ckpt_path = Path(ckpt_path).resolve()
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        args_dict = state.get("args", {})
        self.label_names = D.get_label_names()
        self.num_classes = len(self.label_names)
        self.model = _build_model_from_args(args_dict, self.num_classes).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.arch = args_dict.get("arch", "baseline_cnn")

    @torch.no_grad()
    def predict(self, img: Image.Image, top_k: int = 5) -> list[Prediction]:
        x = _preprocess(img).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_idx = np.argsort(-probs)[:top_k]
        return [Prediction(self.label_names[int(i)], float(probs[int(i)])) for i in top_idx]

    @torch.no_grad()
    def predict_batch(self, imgs: Sequence[Image.Image]) -> np.ndarray:
        x = torch.cat([_preprocess(i) for i in imgs]).to(self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=1).cpu().numpy()
