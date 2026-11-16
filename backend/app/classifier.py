"""DenseNet121 classifier with Grad-CAM explanations."""
from __future__ import annotations

import base64
import io
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models

CLASS_NAMES = ["normal", "pneumonia"]
WEIGHTS_PATH = os.environ.get("CLASSIFIER_WEIGHTS", "models/classifier.pt")


def build_model(pretrained: bool = False) -> torch.nn.Module:
    net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
    net.classifier = torch.nn.Linear(net.classifier.in_features, len(CLASS_NAMES))
    return net


class Classifier:
    def __init__(self, weights_path: str = WEIGHTS_PATH):
        self.model = build_model(pretrained=False)
        self.trained = os.path.exists(weights_path)
        if self.trained:
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state)
        else:
            warnings.warn(
                f"No trained weights at {weights_path}. Predictions are random and for "
                "pipeline testing only. Train with training/train_classifier.py first."
            )
        self.model.eval()

    def predict(self, x: torch.Tensor) -> tuple[str, float, int]:
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)[0]
        idx = int(torch.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx]), idx

    def analyse(self, x: torch.Tensor, original: Image.Image) -> tuple[str, float, int, str]:
        """Classify and build the Grad-CAM heatmap in a single forward pass."""
        target = self.model.features.denseblock4
        activations, gradients = {}, {}

        def fwd_hook(_, __, out):
            activations["value"] = out

        def bwd_hook(_, grad_in, grad_out):
            gradients["value"] = grad_out[0]

        h1 = target.register_forward_hook(fwd_hook)
        h2 = target.register_full_backward_hook(bwd_hook)
        try:
            self.model.zero_grad()
            out = self.model(x)
            probs = F.softmax(out, dim=1)[0].detach()
            idx = int(torch.argmax(probs))
            out[0, idx].backward()
            acts = activations["value"].detach()
            grads = gradients["value"].detach()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=original.size[::-1], mode="bilinear", align_corners=False)
            cam = cam[0, 0].numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            h1.remove()
            h2.remove()
            self.model.zero_grad()

        heat = np.uint8(255 * cam)
        heat_rgb = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=-1)
        base = np.asarray(original.convert("RGB"), dtype=np.float32)
        blended = np.clip(0.55 * base + 0.45 * heat_rgb, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return CLASS_NAMES[idx], float(probs[idx]), idx, heatmap_b64

    def gradcam(self, x: torch.Tensor, class_idx: int, original: Image.Image) -> str:
        """Return a base64 PNG of the Grad-CAM heatmap overlaid on the original image."""
        target = self.model.features.denseblock4  # last dense block
        activations, gradients = {}, {}

        def fwd_hook(_, __, out):
            activations["value"] = out

        def bwd_hook(_, grad_in, grad_out):
            gradients["value"] = grad_out[0]

        h1 = target.register_forward_hook(fwd_hook)
        h2 = target.register_full_backward_hook(bwd_hook)
        try:
            self.model.zero_grad()
            x = x.clone().requires_grad_(False)
            out = self.model(x)
            out[0, class_idx].backward()
            acts = activations["value"].detach()
            grads = gradients["value"].detach()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=original.size[::-1], mode="bilinear", align_corners=False)
            cam = cam[0, 0].numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            h1.remove()
            h2.remove()
            self.model.zero_grad()

        heat = np.uint8(255 * cam)
        heat_rgb = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=-1)
        base = np.asarray(original.convert("RGB"), dtype=np.float32)
        blended = np.clip(0.55 * base + 0.45 * heat_rgb, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
