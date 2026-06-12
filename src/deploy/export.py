"""Export a trained checkpoint to ONNX and CoreML.

ONNX is the portable format (cloud, mobile, edge). CoreML is Apple-native and
runs at full speed on iOS / macOS via the Neural Engine.

Usage
-----
    .venv/bin/python -m src.deploy.export --ckpt results/baseline/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src import config as C
from src import data as D
from src.deploy.inference import _build_model_from_args


def export_onnx(model: torch.nn.Module, out_path: Path, opset: int = 17):
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, C.CROP_SIZE, C.CROP_SIZE)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    print(f"[export] Wrote {out_path}")


def verify_onnx(onnx_path: Path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, C.CROP_SIZE, C.CROP_SIZE).astype(np.float32)
    out = sess.run(None, {"input": dummy})[0]
    print(f"[export] ONNX inference check: out shape {out.shape}")


def export_coreml(model: torch.nn.Module, out_path: Path):
    import coremltools as ct
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, C.CROP_SIZE, C.CROP_SIZE)
    traced = torch.jit.trace(model, dummy)
    label_names = D.get_label_names()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=dummy.shape)],
        classifier_config=ct.ClassifierConfig(label_names),
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(str(out_path))
    print(f"[export] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-coreml", action="store_true")
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / "export"
    out_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    args_dict = state.get("args", {})
    num_classes = D.get_num_classes()
    model = _build_model_from_args(args_dict, num_classes)
    model.load_state_dict(state["model"])

    if not args.skip_onnx:
        onnx_path = out_dir / "model.onnx"
        export_onnx(model, onnx_path)
        verify_onnx(onnx_path)

    if not args.skip_coreml:
        try:
            export_coreml(model, out_dir / "model.mlpackage")
        except Exception as e:
            print(f"[export] CoreML export failed (this is okay if torch>2.4): {e}")


if __name__ == "__main__":
    main()
