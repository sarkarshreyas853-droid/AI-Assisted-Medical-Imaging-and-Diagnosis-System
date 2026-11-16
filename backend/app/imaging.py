"""Image loading and preprocessing.

Accepts DICOM (.dcm) or common image files (PNG/JPEG). Everything is converted
to a 224x224 RGB float tensor normalised with ImageNet statistics, which is what
the DenseNet classifier expects.
"""
from __future__ import annotations

import io

import numpy as np
import pydicom
import torch
from PIL import Image

IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _dicom_to_uint8(data: bytes) -> np.ndarray:
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[arr.shape[0] // 2]  # take the middle slice of a multi-frame series
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 1.0 - arr
    return (arr * 255).astype(np.uint8)


MAX_SIDE = 512  # large uploads are shrunk so the Grad-CAM overlay fits in memory


def load_image(data: bytes, filename: str) -> Image.Image:
    """Return an RGB PIL image from raw upload bytes, shrunk to at most MAX_SIDE pixels."""
    if filename.lower().endswith((".dcm", ".dicom")):
        gray = _dicom_to_uint8(data)
        img = Image.fromarray(gray).convert("RGB")
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> normalised tensor of shape (1, 3, H, W)."""
    resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
