"""Train a 3D U-Net for spleen segmentation on the MONAI Medical Segmentation Decathlon (Task09_Spleen).

Usage (GPU strongly recommended, e.g. Google Colab):
    python training/train_segmenter.py --root data/raw/msd --epochs 50
Writes models/segmenter.pt and reports mean validation Dice.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.apps import DecathlonDataset
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations, AsDiscrete, Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    NormalizeIntensityd, Orientationd, RandCropByPosNegLabeld, Spacingd, ScaleIntensityRanged,
)
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app.segmenter import build_unet  # noqa: E402


def transforms(train: bool):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True),
    ]
    if train:
        base.append(RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                           spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=2))
    base.append(EnsureTyped(keys=["image", "label"]))
    return Compose(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/msd")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="models/segmenter.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = DecathlonDataset(root_dir=args.root, task="Task09_Spleen", section="training",
                                transform=transforms(True), download=True, cache_rate=0.0)
    val_ds = DecathlonDataset(root_dir=args.root, task="Task09_Spleen", section="validation",
                              transform=transforms(False), download=False, cache_rate=0.0)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    model = build_unet().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["image"].to(device), batch["label"].to(device)
                out = sliding_window_inference(x, (96, 96, 96), 4, model)
                outs = [post_pred(i) for i in decollate_batch(out)]
                labs = [post_label(i) for i in decollate_batch(y)]
                dice_metric(y_pred=outs, y=labs)
        mean_dice = float(dice_metric.aggregate().item())
        dice_metric.reset()
        print(f"epoch {epoch}: val_dice={mean_dice:.3f}")
        if mean_dice > best:
            best = mean_dice
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(model.state_dict(), args.out)
    print(f"best validation Dice {best:.3f}; weights in {args.out}")


if __name__ == "__main__":
    main()
