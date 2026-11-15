"""Fine-tune DenseNet121 (ImageNet weights) on a two-class chest X-ray folder.

Expected layout (create it from NIH ChestX-ray14 or a Kaggle pneumonia set):
    data/raw/chest/normal/*.png|jpg|dcm
    data/raw/chest/pneumonia/*.png|jpg|dcm

Usage:
    python training/train_classifier.py --data data/raw/chest --epochs 5
Writes models/classifier.pt and prints validation accuracy and AUC.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app.classifier import CLASS_NAMES, build_model  # noqa: E402
from backend.app.imaging import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, load_image  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".dcm")


class ScanDataset(Dataset):
    def __init__(self, root: str, train: bool):
        self.items = []
        for idx, name in enumerate(CLASS_NAMES):
            folder = os.path.join(root, name)
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(EXTS):
                    self.items.append((os.path.join(folder, fn), idx))
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)] if train else []
        self.tf = transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), *aug, transforms.ToTensor(),
             transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with open(path, "rb") as f:
            img = load_image(f.read(), path)
        return self.tf(img), label


def evaluate(model, loader, device):
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(y.numpy())
    labels, probs = np.array(labels), np.array(probs)
    acc = float(((probs >= 0.5).astype(int) == labels).mean())
    auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/chest")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="models/classifier.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="skip ImageNet weights (offline smoke tests only)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = ScanDataset(args.data, train=True)
    n_val = max(1, int(0.2 * len(full)))
    train_set, val_set = random_split(full, [len(full) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    val_set.dataset = ScanDataset(args.data, train=False)  # no augmentation for validation
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)

    from torchvision.models import DenseNet121_Weights
    from torchvision import models
    weights = None if args.no_pretrained else DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_auc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        acc, auc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: val_acc={acc:.3f} val_auc={auc:.3f}")
        score = auc if not np.isnan(auc) else acc
        if score > best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(best_state, args.out)
    print(f"saved {args.out} (best validation score {best_auc:.3f})")


if __name__ == "__main__":
    main()
