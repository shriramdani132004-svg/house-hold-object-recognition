"""Compact PyTorch classifier for CORe50 continual recognition.

The continual assignment uses CORe50 for 50-class object-identity
*classification* (see ``docs/dataset/core50/DATASET_INFO.md``); the Ultralytics
YOLO stack stays reserved for the untouched COCO detection prototype.
``SmallConvNet`` is deliberately small so Phase-5's full NIC experiment is
practical on this CPU-only machine while remaining a real trainable model.
"""

from __future__ import annotations

import logging
import random

import torch
import torch.nn as nn

MODEL_ARCH = "small_cnn"
SUPPORTED_ARCHS = ("small_cnn", "compact_resnet")
MODEL_CONTRACT = (
    "forward((batch, 3, image_size, image_size)) must return finite "
    "(batch, num_classes) logits; num_classes, image_size, and batch "
    "must be positive ints"
)

logger = logging.getLogger(__name__)


class ModelContractError(ValueError):
    """Raised when a model violates the forward-pass contract."""


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def seed_everything(seed: int) -> None:
    """Seed python/torch (+numpy when installed) for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception as exc:
        logger.warning(
            "numpy seeding was skipped (%s); continuing with the "
            "python and torch seeds",
            exc,
        )


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


class CompactResNet(nn.Module):
    """Small residual CNN using GroupNorm (no BatchNorm running stats).

    Diagnosis-driven design: BatchNorm running statistics drift toward the
    most recent continual experience and mis-normalize older classes at
    inference time. GroupNorm keeps per-sample statistics independent of
    which experience the sample came from.
    """

    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if width < 4:
            raise ValueError(f"width must be >= 4, got {width}")
        groups = 8 if width % 8 == 0 else (4 if width % 4 == 0 else 1)
        w1, w2, w3 = width, width * 2, width * 4

        def block(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, w1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = block(w1)
        self.down1 = nn.Sequential(
            nn.Conv2d(w1, w2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, w2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = block(w2)
        self.down2 = nn.Sequential(
            nn.Conv2d(w2, w3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, w3),
            nn.ReLU(inplace=True),
        )
        self.stage3 = block(w3)
        self.down3 = nn.Sequential(
            nn.Conv2d(w3, w3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, w3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(w3, num_classes)
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = torch.relu(x + self.stage1(x))
        x = self.down1(x)
        x = torch.relu(x + self.stage2(x))
        x = self.down2(x)
        x = torch.relu(x + self.stage3(x))
        x = self.down3(x)
        return self.classifier(self.pool(x).flatten(1))


def build_model(
    num_classes: int,
    *,
    width: int = 32,
    seed: int | None = None,
    arch: str = "small_cnn",
) -> nn.Module:
    """Build the classifier; with ``seed`` the initialization is isolated
    (the global RNG state is restored afterwards)."""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; supported: {', '.join(SUPPORTED_ARCHS)}")
    if not _is_positive_int(num_classes):
        raise ValueError(f"num_classes must be a positive int, got {num_classes!r}")
    if not _is_positive_int(width):
        raise ValueError(f"width must be a positive int, got {width!r}")
    if seed is None:
        if arch == "compact_resnet":
            return CompactResNet(num_classes, width)
        return SmallConvNet(num_classes, width)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arch == "compact_resnet":
            return CompactResNet(num_classes, width)
        return SmallConvNet(num_classes, width)


def assert_model_contract(
    model: nn.Module,
    *,
    num_classes: int,
    image_size: int,
    batch: int = 2,
    device: str = "cpu",
) -> dict[str, object]:
    """Verify the forward-pass contract (:data:`MODEL_CONTRACT`) once.

    Runs a single no-grad forward on a synthetic zeros input of shape
    ``(batch, 3, image_size, image_size)`` with the model in eval mode;
    the model's previous training/eval mode is restored afterwards.
    Raises :class:`ModelContractError` when the requested contract is
    invalid, the output shape is not ``(batch, num_classes)``, or the
    output contains non-finite values. Returns a small summary dict
    (``output_shape``, ``num_classes``, ``param_count``) on success.
    """
    if not _is_positive_int(num_classes):
        raise ModelContractError(
            f"num_classes must be a positive int, got {num_classes!r}"
        )
    if not _is_positive_int(image_size):
        raise ModelContractError(
            f"image_size must be a positive int, got {image_size!r}"
        )
    if not _is_positive_int(batch):
        raise ModelContractError(f"batch must be a positive int, got {batch!r}")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            inputs = torch.zeros(batch, 3, image_size, image_size, device=device)
            output = model(inputs)
    finally:
        model.train(was_training)
    if not isinstance(output, torch.Tensor):
        raise ModelContractError(
            f"Model output must be a tensor, got {type(output).__name__}"
        )
    if tuple(output.shape) != (batch, num_classes):
        raise ModelContractError(
            f"Model output shape {tuple(output.shape)} does not match the "
            f"contract {(batch, num_classes)}: {MODEL_CONTRACT}"
        )
    if not bool(torch.isfinite(output).all()):
        raise ModelContractError(
            f"Model output contains non-finite values (NaN/Inf): {MODEL_CONTRACT}"
        )
    return {
        "output_shape": tuple(output.shape),
        "num_classes": int(num_classes),
        "param_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def model_checksum(model: nn.Module) -> str:
    """Short deterministic checksum of all parameters (for tests/logs)."""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]
