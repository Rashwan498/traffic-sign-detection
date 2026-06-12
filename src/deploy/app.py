"""Gradio demo app -- upload an image, get top-5 traffic sign predictions.

Run locally:
    .venv/bin/python -m src.deploy.app

Or set CKPT env var to point at a specific checkpoint:
    CKPT=results/transfer/best.pt .venv/bin/python -m src.deploy.app

For Hugging Face Spaces deployment, this file is copied to the Space's
app.py along with the chosen checkpoint and the necessary src/ modules.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from src import config as C
from src.deploy.inference import SignClassifier

DEFAULT_CKPT = os.environ.get("CKPT", str(C.RESULTS_ROOT / "best_model.pt"))


def pick_ckpt() -> str:
    """If CKPT env var or default file isn't there, find the best available."""
    if Path(DEFAULT_CKPT).exists():
        return DEFAULT_CKPT
    # fall back: most-improved best.pt by mtime
    candidates = list(C.RESULTS_ROOT.glob("*/best.pt"))
    if not candidates:
        raise FileNotFoundError("No trained checkpoint found in results/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


CKPT = pick_ckpt()
print(f"[app] Loading model from {CKPT}")
CLF = SignClassifier(CKPT)
print(f"[app] arch={CLF.arch}, num_classes={CLF.num_classes}, device={CLF.device}")


def predict(image):
    if image is None:
        return {}
    preds = CLF.predict(image, top_k=5)
    return {p.label: p.probability for p in preds}


with gr.Blocks(title="MTSD Traffic Sign Classifier") as demo:
    gr.Markdown(
        "# 🚦 Traffic Sign Classifier (MTSD)\n"
        f"Upload a cropped traffic sign image. The model returns its top-5 predictions.\n\n"
        f"**Model**: `{CLF.arch}` • **Classes**: {CLF.num_classes} • "
        f"**Device**: `{CLF.device}` • **Checkpoint**: `{Path(CKPT).name}`"
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Input image")
        out = gr.Label(num_top_classes=5, label="Top-5 predictions")
    inp.change(fn=predict, inputs=inp, outputs=out)
    # Examples drawn from the test set are added at build time by deploy script.

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
