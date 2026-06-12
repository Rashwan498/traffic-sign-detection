# 🚦 Vision Project — Handoff to Colleague

This folder contains everything needed to pick up the **bonus deployment
tasks** (Docker + Hugging Face Spaces) and to read / extend the work.

The big artifacts that are **not** included (and how to regenerate them):

| Not shipped | Size | How to regenerate |
|---|---|---|
| `archive/` (raw MTSD images) | ~33 GB | Download from https://www.mapillary.com/dataset/trafficsign |
| `data/crops_cache.h5` (190 k cropped 96×96 patches) | 3.0 GB | `python -m src.prepare_data` after putting `archive/` in place |
| `.venv/` | a few hundred MB | `python3 -m venv .venv --system-site-packages && .venv/bin/pip install -r requirements.txt` |

**You only need to regenerate them if you want to re-train models.** For
the bonus tasks (Docker, Gradio, Hugging Face), the model checkpoint
shipped here is enough.

---

## What's in the folder

```
HANDOFF.md           ← this file
README.md            ← original reproduction guide
requirements.txt     ← Python deps (assumes Anaconda + Apple Silicon)
run_experiments.sh   ← what was used to train everything

src/                 ← all source code
  config.py
  prepare_data.py    ← Stage 0: cache crops to HDF5
  build_splits.py    ← Stage 0b: clean class set + splits
  data.py            ← PyTorch Dataset
  models/baseline_cnn.py
  train.py
  evaluate.py
  distortions.py
  robustness.py
  phase1_rerun.py / phase1_fast.py
  deploy/
    inference.py     ← SignClassifier wrapper, used by app + api
    app.py           ← Gradio app  (the demo)
    api.py           ← FastAPI service
    Dockerfile       ← CPU-only FastAPI container
    api_requirements.txt
    export.py        ← ONNX + CoreML exporter

data/
  label_map.json     ← 326-class label map (string ↔ int)
  splits.json        ← train / val / test cache indices + labels
  class_weights.npy  ← sqrt-inverse-freq class weights

results/
  best_model.pt          ← STAGED for the Docker build (= aug/best.pt)
  aug/                   ← BEST run (highest macro-F1 = 0.958)
    best.pt
    args.json, history.json
    test_metrics.json, test_per_class.csv
    test_confusion_matrix.{npy,png}
    robustness.{csv,png}
    training_curves.png
    export/
      model.onnx (+ model.onnx.data)
      model.mlpackage/    ← Apple Core ML bundle
  baseline/              ← from-scratch CNN, no aug
  transfer/              ← EfficientNet-B0 transfer learning
  phase1_rerun/          ← classical HOG + SGD-SVM baseline

notebooks/
  visionPhase1.ipynb       ← classical baseline (original)
  visionPhase2.ipynb       ← Phase 2 deliverable (executed, all outputs)
  visionPhase2.html        ← same, as static HTML (no Jupyter needed)

paper/
  paper.md                 ← finished paper with all real numbers
```

---

## Headline results (already in the paper + notebook)

| Model | Top-1 Acc | Macro-F1 |
|---|---:|---:|
| Phase 1 — HOG + SGD-SVM | 55.97 % | 0.524 |
| Phase 2 BaselineCNN (no aug) | 96.55 % | 0.949 |
| Phase 2 + Augmentation 🏆 best macro-F1 | 96.89 % | **0.958** |
| Phase 2 + Transfer (EfficientNet-B0) 🏆 best top-1 | **96.98 %** | 0.956 |

`results/best_model.pt` and `results/aug/best.pt` are the SAME file — the
best macro-F1 checkpoint.

---

## Bonus tasks left for you

### 1. Build & run the Docker image

```bash
docker build -t mtsd-api -f src/deploy/Dockerfile .
docker run --rm -p 8000:8000 mtsd-api
```

Then in another shell:

```bash
curl -s http://localhost:8000/healthz
curl -X POST -F "file=@some_sign.jpg" http://localhost:8000/predict
```

The Dockerfile installs CPU-only torch (it's smaller, faster build) and
copies `results/best_model.pt` into the image. The image is ~600 MB.

⚠️ If the `.dockerignore` ignores `results/best_model.pt`, edit
`src/deploy/.dockerignore` — it currently keeps `best_model.pt` but
excludes other checkpoints/logs to keep the build context small.

### 2. Deploy the Gradio app to Hugging Face Spaces

1. Create a free HF account and a new Space (SDK = Gradio).
2. Clone the Space repo locally.
3. Copy these into the Space repo:
   - `src/deploy/app.py` → rename to `app.py` in the Space root
   - `src/deploy/inference.py` → into `src/deploy/`
   - `src/__init__.py`, `src/config.py`, `src/data.py` (only for label
     names — you can simplify these)
   - `src/models/baseline_cnn.py`, `src/models/__init__.py`
   - `results/best_model.pt`
   - `data/label_map.json`, `data/splits.json` (only `label_map.json`
     is strictly needed — `splits.json` is huge and unnecessary at
     inference time)
4. Add a `requirements.txt` at the Space root with just:
   ```
   torch
   torchvision
   gradio==4.44.1
   numpy
   pillow
   timm   # only if you want to also serve the transfer model
   ```
5. `git push` — the Space builds + serves automatically.

If you want a public URL right now without HF, you can also do:

```bash
.venv/bin/python -m src.deploy.app    # local: http://localhost:7860
# or for a temporary public URL:
.venv/bin/python -c "import gradio as gr; from src.deploy.app import demo; demo.launch(share=True)"
```

### 3. (Optional) Push the Docker image to a public registry

```bash
docker tag mtsd-api <yourusername>/mtsd-api:latest
docker push <yourusername>/mtsd-api:latest
```

---

## How to load the model in your own code (quick reference)

```python
from src.deploy.inference import SignClassifier
from PIL import Image

clf = SignClassifier("results/best_model.pt")  # auto-uses MPS if available
preds = clf.predict(Image.open("some_sign.jpg"), top_k=5)
for p in preds:
    print(f"{p.probability:.3f}  {p.label}")
```

---

Questions? The original notebook (`visionPhase2.ipynb`) walks through the
whole pipeline; the paper (`paper/paper.md`) has all the methodology
written up.
