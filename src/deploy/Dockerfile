# Multi-arch image. Built on linux/amd64 for cloud, linux/arm64 for Apple Silicon dev.
# Build:   docker build -t mtsd-api -f src/deploy/Dockerfile .
# Run  :   docker run --rm -p 8000:8000 mtsd-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for OpenCV + Pillow (image decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch for the container. We pin to a recent stable version that
# matches our training environment behaviorally; ONNX is the preferred path.
RUN pip install --no-cache-dir \
    "torch==2.4.1+cpu" --index-url https://download.pytorch.org/whl/cpu

# Inference deps only (small subset of training requirements.txt)
COPY src/deploy/api_requirements.txt /tmp/api_requirements.txt
RUN pip install --no-cache-dir -r /tmp/api_requirements.txt

# Copy what the API needs
COPY src/ /app/src/
COPY data/label_map.json /app/data/label_map.json
# splits.json and crops_cache.h5 are NOT shipped -- they're for training.
# We bake the chosen checkpoint into the image as best_model.pt.
COPY results/best_model.pt /app/results/best_model.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)" || exit 1

CMD ["uvicorn", "src.deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
