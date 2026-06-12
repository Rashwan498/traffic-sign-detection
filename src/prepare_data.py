"""Stage 0 — extract traffic-sign crops from MTSD into an HDF5 cache.

This is fully resumable. Pause with Ctrl+C at any time and re-run the same
command to continue. State is checkpointed every WRITE_CHUNK crops.

Pipeline
--------
1. Read MTSD's official splits (train.txt, val.txt). The dataset's "test" split
   has no public labels, so we use MTSD-val as our held-out *test* set and
   carve a stratified slice from MTSD-train as our *validation* set later
   (handled by build_splits.py, stage 0b).
2. For every annotated image, open it, walk its objects, crop each bbox,
   resize to 96x96, and append to crops_cache.h5 with metadata.
3. After every WRITE_CHUNK crops, flush HDF5 and update .prep_progress.json.

Resumption logic
----------------
On restart we read .prep_progress.json -> {"done_images": [...], "n_crops": N}.
Any image whose id is already in done_images is skipped. We seek HDF5 to N
and continue appending.

Usage
-----
    .venv/bin/python -m src.prepare_data
    .venv/bin/python -m src.prepare_data --limit 1000   # quick dry-run
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from src import config as C

# h5py string dtype for variable-length UTF-8
H5_STR = h5py.string_dtype(encoding="utf-8")


# --------------------------------------------------------------------------
# Graceful Ctrl+C handling
# --------------------------------------------------------------------------
_INTERRUPTED = False


def _handle_sigint(signum, frame):
    """First Ctrl+C: ask the main loop to stop after the current image.
    Second Ctrl+C: hard exit."""
    global _INTERRUPTED
    if _INTERRUPTED:
        print("\n[prepare_data] Second Ctrl+C -- hard exit.", flush=True)
        sys.exit(130)
    _INTERRUPTED = True
    print(
        "\n[prepare_data] Ctrl+C caught. Finishing current image, then "
        "flushing and exiting cleanly. Press Ctrl+C again to force-quit.",
        flush=True,
    )


signal.signal(signal.SIGINT, _handle_sigint)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def read_split(name: str) -> list[str]:
    """Return image ids (one per line in splits/<name>.txt, no .jpg ext)."""
    path = C.SPLITS_DIR / f"{name}.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def locate_image(img_id: str) -> Path | None:
    """MTSD ships train and val images in two separate folders. Return whichever exists."""
    for d in (C.TRAIN_IMG_DIR, C.VAL_IMG_DIR):
        p = d / f"{img_id}.jpg"
        if p.exists():
            return p
    return None


def load_progress() -> dict:
    if C.PREP_PROGRESS.exists():
        return json.loads(C.PREP_PROGRESS.read_text())
    return {"done_images": [], "n_crops": 0}


def save_progress(state: dict) -> None:
    tmp = C.PREP_PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(C.PREP_PROGRESS)  # atomic on POSIX


# --------------------------------------------------------------------------
# HDF5 setup
# --------------------------------------------------------------------------
def open_h5() -> h5py.File:
    """Open or create the cache file with resizable datasets."""
    new_file = not C.CROPS_H5.exists()
    f = h5py.File(C.CROPS_H5, "a")
    if new_file:
        f.create_dataset(
            "crops",
            shape=(0, C.CROP_SIZE, C.CROP_SIZE, 3),
            maxshape=(None, C.CROP_SIZE, C.CROP_SIZE, 3),
            dtype="uint8",
            chunks=(64, C.CROP_SIZE, C.CROP_SIZE, 3),
            compression="lzf",
        )
        f.create_dataset(
            "raw_label", shape=(0,), maxshape=(None,), dtype=H5_STR
        )
        f.create_dataset(
            "split", shape=(0,), maxshape=(None,), dtype=H5_STR
        )
        f.create_dataset(
            "image_id", shape=(0,), maxshape=(None,), dtype=H5_STR
        )
        # provenance attrs
        f.attrs["crop_size"] = C.CROP_SIZE
        f.attrs["min_bbox_side_px"] = C.MIN_BBOX_SIDE_PX
    return f


def append_batch(f: h5py.File, buf: list[tuple]) -> None:
    """buf items: (crop_uint8_HWC, raw_label, split, image_id)."""
    if not buf:
        return
    n_old = f["crops"].shape[0]
    n_new = n_old + len(buf)
    f["crops"].resize((n_new, C.CROP_SIZE, C.CROP_SIZE, 3))
    f["raw_label"].resize((n_new,))
    f["split"].resize((n_new,))
    f["image_id"].resize((n_new,))
    f["crops"][n_old:n_new] = np.stack([b[0] for b in buf])
    f["raw_label"][n_old:n_new] = [b[1] for b in buf]
    f["split"][n_old:n_new] = [b[2] for b in buf]
    f["image_id"][n_old:n_new] = [b[3] for b in buf]
    f.flush()


# --------------------------------------------------------------------------
# Crop extraction for one image
# --------------------------------------------------------------------------
def crops_from_image(img_id: str, split: str) -> list[tuple]:
    """Return list of (crop_uint8_RGB_HWC, raw_label, split, image_id)."""
    ann_path = C.ANNOT_DIR / f"{img_id}.json"
    img_path = locate_image(img_id)
    if img_path is None or not ann_path.exists():
        return []
    try:
        ann = json.loads(ann_path.read_text())
    except Exception:
        return []
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return []
    H, W = img_bgr.shape[:2]
    out = []
    for obj in ann.get("objects", []):
        label = obj.get("label")
        if not label:
            continue
        bb = obj.get("bbox", {})
        try:
            xmin, ymin = int(round(bb["xmin"])), int(round(bb["ymin"]))
            xmax, ymax = int(round(bb["xmax"])), int(round(bb["ymax"]))
        except (KeyError, TypeError):
            continue
        # clamp to image bounds
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(W, xmax); ymax = min(H, ymax)
        w, h = xmax - xmin, ymax - ymin
        if w < C.MIN_BBOX_SIDE_PX or h < C.MIN_BBOX_SIDE_PX:
            continue
        crop = img_bgr[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (C.CROP_SIZE, C.CROP_SIZE), interpolation=cv2.INTER_AREA)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out.append((crop_rgb, label, split, img_id))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many *additional* images (0 = no limit). Useful for dry-runs.")
    args = parser.parse_args()

    C.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    C.seed_everything()

    train_ids = read_split("train")
    val_ids = read_split("val")
    work: list[tuple[str, str]] = [(i, "mtsd_train") for i in train_ids] + \
                                  [(i, "mtsd_val") for i in val_ids]
    print(f"[prepare_data] Total annotated images: {len(work):,} "
          f"(train={len(train_ids):,}, val={len(val_ids):,})")

    state = load_progress()
    done = set(state["done_images"])
    print(f"[prepare_data] Already processed: {len(done):,} images, "
          f"{state['n_crops']:,} crops cached.")

    f = open_h5()
    pending: list[tuple] = []
    n_crops = state["n_crops"]
    processed_this_run = 0
    t0 = time.time()
    skipped = 0

    pbar = tqdm(work, desc="extracting", unit="img", initial=len(done))
    try:
        for img_id, split in pbar:
            if _INTERRUPTED:
                break
            if img_id in done:
                continue
            if args.limit and processed_this_run >= args.limit:
                print(f"[prepare_data] --limit {args.limit} reached.")
                break

            crops = crops_from_image(img_id, split)
            if not crops:
                skipped += 1
            pending.extend(crops)
            n_crops += len(crops)
            done.add(img_id)
            processed_this_run += 1

            if len(pending) >= C.WRITE_CHUNK:
                append_batch(f, pending)
                pending = []
                state["done_images"] = sorted(done)
                state["n_crops"] = n_crops
                save_progress(state)
                pbar.set_postfix(crops=n_crops, skipped=skipped)
    finally:
        # flush remaining buffer + state, then close cleanly
        append_batch(f, pending)
        state["done_images"] = sorted(done)
        state["n_crops"] = n_crops
        save_progress(state)
        f.close()
        dt = time.time() - t0
        print(f"\n[prepare_data] Flushed and saved. "
              f"This run: {processed_this_run} images, {n_crops - state['n_crops'] + len(pending):,} new crops "
              f"in {dt/60:.1f} min. Total crops on disk: {n_crops:,}. "
              f"Images skipped (no annotations / unreadable): {skipped}.")
        if _INTERRUPTED:
            print("[prepare_data] Interrupted -- safe to resume with the same command.")


if __name__ == "__main__":
    main()
