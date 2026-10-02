"""Compact PyTorch classifier for CORe50 continual recognition.

The continual assignment uses CORe50 for 50-class object-identity
*classification* (see ``data/raw/core50/DATASET_INFO.md``); the Ultralytics
YOLO stack stays reserved for the untouched COCO detection prototype.
``SmallConvNet`` is deliberately small so Phase-5's full NIC experiment is
practical on this CPU-only machine while remaining a real trainable model.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

MODEL_ARCH = "small_cnn"


def seed_everything(seed: int) -> None:
    """Seed python/torch (+numpy when installed) for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:  # noqa: BLE001 - numpy is optional for seeding purposes
        pass


class SmallConvNet(nn.Module):
    """Three conv blocks + adaptive pooling + linear head.

    Fully convolutional up to the head, so any square input >= 8 px works
    (the configured ``image_size`` decides the actual resolution).
    """

    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if width < 4:
            raise ValueError(f"width must be >= 4, got {width}")
        w1, w2, w3 = width, width * 2, width * 4
        self.features = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, padding=1),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w1, w2, kernel_size=3, padding=1),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w2, w3, kernel_size=3, padding=1),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pool = nn.Flatten()
        self.classifier = nn.Linear(w3, num_classes)
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def build_model(num_classes: int, *, width: int = 32, seed: int | None = None) -> SmallConvNet:
    """Build the classifier; with ``seed`` the initialization is isolated
    (the global RNG state is restored afterwards)."""
    if seed is None:
        return SmallConvNet(num_classes, width)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SmallConvNet(num_classes, width)


def model_checksum(model: nn.Module) -> str:
    """Short deterministic checksum of all parameters (for tests/logs)."""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]
