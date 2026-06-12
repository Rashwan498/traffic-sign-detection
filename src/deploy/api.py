"""FastAPI service -- POST an image, get JSON predictions.

Run locally:
    .venv/bin/uvicorn src.deploy.api:app --host 0.0.0.0 --port 8000

Or inside Docker via the Dockerfile in this directory.

Endpoints
---------
GET  /                  : service metadata
GET  /healthz           : liveness probe
POST /predict           : multipart/form-data with field 'file', returns top-K JSON
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src import config as C
from src.deploy.inference import SignClassifier

DEFAULT_CKPT = os.environ.get("CKPT", str(C.RESULTS_ROOT / "best_model.pt"))


def pick_ckpt() -> str:
    if Path(DEFAULT_CKPT).exists():
        return DEFAULT_CKPT
    candidates = list(C.RESULTS_ROOT.glob("*/best.pt"))
    if not candidates:
        raise FileNotFoundError("No trained checkpoint found in results/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


app = FastAPI(title="MTSD Traffic Sign Classifier",
              version="1.0",
              description="Classify a cropped traffic-sign image.")

# Lazy-load model at startup so the API responds to /healthz before predict.
@app.on_event("startup")
def _load_model():
    global CLF
    ckpt = pick_ckpt()
    CLF = SignClassifier(ckpt, device="cpu")  # CPU in containers
    app.state.ckpt = ckpt
    app.state.arch = CLF.arch
    app.state.num_classes = CLF.num_classes


class PredItem(BaseModel):
    label: str
    probability: float


class PredResponse(BaseModel):
    predictions: list[PredItem]
    model_arch: str
    num_classes: int


@app.get("/")
def root():
    return {
        "service": "MTSD Traffic Sign Classifier",
        "checkpoint": Path(app.state.ckpt).name,
        "arch": app.state.arch,
        "num_classes": app.state.num_classes,
        "endpoints": {"POST /predict": "multipart file -> top-5", "GET /healthz": "liveness"},
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/predict", response_model=PredResponse)
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")
    preds = CLF.predict(img, top_k=top_k)
    return PredResponse(
        predictions=[PredItem(**p.asdict()) for p in preds],
        model_arch=app.state.arch,
        num_classes=app.state.num_classes,
    )
