"""3D U-Net spleen segmentation (MONAI) with Dice scoring."""
from __future__ import annotations

import base64
import io
import os
import warnings

import numpy as np
import torch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

WEIGHTS_PATH = os.environ.get("SEGMENTER_WEIGHTS", "models/segmenter.pt")


def build_unet() -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # background, spleen
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


class Segmenter:
    def __init__(self, weights_path: str = WEIGHTS_PATH):
        self.model = build_unet()
        self.trained = os.path.exists(weights_path)
        if self.trained:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            warnings.warn(f"No segmenter weights at {weights_path}; output is for pipeline testing only.")
        self.model.eval()

    @torch.no_grad()
    def predict_mask(self, volume: torch.Tensor) -> np.ndarray:
        """volume: (1, 1, D, H, W) float tensor, already normalised. Returns (D, H, W) uint8 mask."""
        logits = sliding_window_inference(volume, roi_size=(96, 96, 96), sw_batch_size=1, predictor=self.model)
        return torch.argmax(logits, dim=1)[0].numpy().astype(np.uint8)


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred.astype(bool), truth.astype(bool)
    denom = pred.sum() + truth.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, truth).sum() / denom)
