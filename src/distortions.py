"""Synthetic distortion suite for the robustness experiment (Stage 2 E3).

Six distortions, each parameterized so we can sweep severity:
  - illumination  : brightness shift
  - rotation      : small in-plane rotation
  - occlusion     : square black patch in a random location
  - gaussian_noise: additive isotropic Gaussian noise
  - motion_blur   : directional motion blur
  - jpeg          : JPEG compression artifact at given quality

Why these six (justified in the paper):
  - illumination  models day/night & headlight variation
  - rotation      models camera tilt & sign mounting variance
  - occlusion     models foliage / vehicle / debris partial cover
  - gaussian_noise models low-light sensor noise
  - motion_blur   models moving-vehicle capture
  - jpeg          models transmitted/stored low-quality input

All operate on uint8 HWC numpy arrays and return uint8 HWC. The Albumentations
versions are used everywhere to keep one consistent backend.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np

# Severity levels: 0 = clean, 1 = mild, 2 = moderate, 3 = severe.
# Tuned so level 3 visibly damages the image but stays recognizable to a human.

DISTORTION_SPECS = {
    "illumination": [None,
        A.RandomBrightnessContrast(brightness_limit=(-0.20, 0.20), contrast_limit=0.0, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.40, 0.40), contrast_limit=0.0, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.60, 0.60), contrast_limit=0.0, p=1.0),
    ],
    "rotation": [None,
        A.Rotate(limit=(-8, 8), border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Rotate(limit=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Rotate(limit=(-25, 25), border_mode=cv2.BORDER_CONSTANT, p=1.0),
    ],
    "occlusion": [None,
        A.CoarseDropout(max_holes=1, max_height=18, max_width=18,
                        min_holes=1, min_height=14, min_width=14, fill_value=0, p=1.0),
        A.CoarseDropout(max_holes=1, max_height=28, max_width=28,
                        min_holes=1, min_height=22, min_width=22, fill_value=0, p=1.0),
        A.CoarseDropout(max_holes=1, max_height=40, max_width=40,
                        min_holes=1, min_height=34, min_width=34, fill_value=0, p=1.0),
    ],
    "gaussian_noise": [None,
        A.GaussNoise(var_limit=(10.0, 40.0), mean=0, p=1.0),
        A.GaussNoise(var_limit=(40.0, 100.0), mean=0, p=1.0),
        A.GaussNoise(var_limit=(100.0, 200.0), mean=0, p=1.0),
    ],
    "motion_blur": [None,
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(5, 9), p=1.0),
        A.MotionBlur(blur_limit=(9, 15), p=1.0),
    ],
    "jpeg": [None,
        A.ImageCompression(quality_lower=50, quality_upper=60, p=1.0),
        A.ImageCompression(quality_lower=25, quality_upper=35, p=1.0),
        A.ImageCompression(quality_lower=8,  quality_upper=15, p=1.0),
    ],
}

SEVERITY_LEVELS = [0, 1, 2, 3]
DISTORTION_NAMES = list(DISTORTION_SPECS.keys())


def apply_distortion(img_uint8_hwc: np.ndarray, name: str, level: int,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply one distortion at one severity. Level 0 = clean (no-op)."""
    if level == 0:
        return img_uint8_hwc
    spec = DISTORTION_SPECS[name][level]
    # Albumentations uses its own RNG; we re-seed it for reproducibility.
    if rng is None:
        rng = np.random.default_rng()
    seed = int(rng.integers(0, 2**31 - 1))
    import random
    random.seed(seed); np.random.seed(seed)
    return spec(image=img_uint8_hwc)["image"]


def make_corrupted_test_set(images_uint8_NHWC: np.ndarray, name: str, level: int,
                            seed: int = 0) -> np.ndarray:
    """Apply (name, level) to every image. Returns a new (N, H, W, 3) uint8 array."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(images_uint8_NHWC)
    for i in range(images_uint8_NHWC.shape[0]):
        out[i] = apply_distortion(images_uint8_NHWC[i], name, level, rng=rng)
    return out
