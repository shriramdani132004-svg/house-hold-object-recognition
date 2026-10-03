"""Diagnosis-driven Experience Replay improvements (model-improvement work).

Adds, without touching the Phase-5 defaults:

- :class:`AugmentedCachedDataset` — exact training preprocessing served
  from a decoded :class:`~src.training.tensor_cache.TensorCache`, with
  optional light, semantically safe augmentation (horizontal flip, small
  translation, mild brightness/contrast) applied only during training.
- :class:`ImprovedReplayTrainer` — the standard
  :class:`~src.training.replay.ReplayContinualTrainer` plus
  development-validation early stopping (the only stopping signal) and
  an optional decoded cache for speed.

Experience Replay remains the only anti-forgetting method; only the
memory policy (FIFO vs reservoir), capacity, ratio, duration and the
model configuration vary between candidates.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.continual import SampleRecord
from src.training.base import ProgressCallback
from src.training.replay import ReplayContinualTrainer
from src.training.tensor_cache import TensorCache

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
_MEAN = torch.tensor(MEAN).view(3, 1, 1)
_STD = torch.tensor(STD).view(3, 1, 1)

DevProvider = Callable[[int], Sequence[SampleRecord]]


def _decode_live(path, image_size: int) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return tensor.reshape(image_size, image_size, 3).permute(2, 0, 1)


def _normalize(uint8: torch.Tensor) -> torch.Tensor:
    tensor = uint8.to(torch.float32).div_(255.0)
    return (tensor - _MEAN) / _STD


def _augment(image: torch.Tensor) -> torch.Tensor:
    """Light, semantically safe augmentation on a normalized CHW tensor."""
    size = image.shape[-1]
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=[2])
    dx = int(torch.randint(-2, 3, ()).item())
    dy = int(torch.randint(-2, 3, ()))
    if dx or dy:
        padded = F.pad(image.unsqueeze(0), (2, 2, 2, 2), mode="reflect")
        top = 2 - dy
        left = 2 - dx
        image = padded[0, :, top : top + size, left : left + size]
    contrast = 1.0 + 0.2 * (float(torch.rand(())) - 0.5)
    brightness = 0.1 * (float(torch.rand(())) - 0.5)
    return (image * contrast + brightness).clamp_(-1.0, 1.0)


class AugmentedCachedDataset(Dataset):
    """Official training records with cache-backed decoding + training aug.

    With ``augment=False`` the output is bit-identical to
    :class:`src.training.dataset.ContinualImageDataset`.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        images_root,
        image_size: int,
        *,
        cache: TensorCache | None = None,
        augment: bool = False,
        require_split: str | None = "train",
    ) -> None:
        self.records = tuple(records)
        self.images_root = images_root
        self.image_size = int(image_size)
        self.cache = cache
        self.augment = bool(augment)
        if require_split is not None:
            bad = [r for r in self.records if r.split != require_split]
            if bad:
                raise ValueError(
                    f"Dataset requires split={require_split!r} but contains "
                    f"{len(bad)} record(s) with split={bad[0].split!r}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        cached = self.cache is not None and record.relative_path in self.cache
        if cached:
            uint8 = self.cache.get(record.relative_path)
        else:
            path = self.images_root.joinpath(*record.relative_path.split("/"))
            uint8 = _decode_live(path, self.image_size)
        image = _normalize(uint8)
        if self.augment:
            image = _augment(image)
        return image, int(record.label)


class ImprovedReplayTrainer(ReplayContinualTrainer):
    """Experience Replay trainer with dev-validation early stopping.

    ``dev_provider(experience_id)`` returns that experience's development
    validation records. After every training epoch the model is evaluated
    on them (no augmentation); training for the experience stops once the
    validation accuracy fails to improve for ``early_stop_patience``
    epochs. With ``early_stop_patience=None`` behaviour matches the
    standard trainer exactly.
    """

    def __init__(
        self,
        config=None,
        *,
        on_progress: ProgressCallback | None = None,
        cache: TensorCache | None = None,
        dev_provider: DevProvider | None = None,
    ) -> None:
        super().__init__(config, on_progress=on_progress)
        self._cache = cache
        self._dev_provider = dev_provider
        self._dev_best: dict[int, float] = {}
        self._dev_bad: dict[int, int] = {}
        self.on_dev_metrics: Callable[[dict[str, Any]], None] | None = None

    # ------------------------------------------------------------------
    def _build_train_dataset(self, experience) -> Dataset:
        assert self._scenario is not None
        return AugmentedCachedDataset(
            experience.train_samples,
            self._scenario.images_root,
            self.config.image_size,
            cache=self._cache,
            augment=self.config.augment,
            require_split="train",
        )

    def _on_epoch_end(self, experience, epoch: int, stats: dict[str, Any]) -> bool:
        patience = self.config.early_stop_patience
        if patience is None or self._dev_provider is None:
            return False
        dev_records = self._dev_provider(experience.experience_id)
        if not dev_records:
            return False
        accuracy = self._evaluate_records(dev_records)
        if self.on_dev_metrics is not None:
            self.on_dev_metrics(
                {
                    "event": "dev_epoch",
                    "experience": experience.experience_id,
                    "epoch": epoch + 1,
                    "dev_accuracy": accuracy,
                    "train_loss": stats.get("loss"),
                    "train_accuracy": stats.get("accuracy"),
                }
            )
        best = self._dev_best.get(experience.experience_id, -1.0)
        if accuracy > best + 1e-4:
            self._dev_best[experience.experience_id] = accuracy
            self._dev_bad[experience.experience_id] = 0
            return False
        self._dev_bad[experience.experience_id] = (
            self._dev_bad.get(experience.experience_id, 0) + 1
        )
        return self._dev_bad[experience.experience_id] >= patience

    def _evaluate_records(self, records: Sequence[SampleRecord]) -> float:
        """Top-1 accuracy over ``records`` (evaluation mode, no grad)."""
        assert self._scenario is not None
        assert self._model is not None
        dataset = AugmentedCachedDataset(
            records,
            self._scenario.images_root,
            self.config.image_size,
            cache=self._cache,
            augment=False,
            require_split=None,
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        correct = 0
        total = 0
        self._model.eval()
        with torch.inference_mode():
            for images, labels in loader:
                predictions = self._model(images.to(self.device)).argmax(dim=1).cpu()
                correct += int(predictions.eq(labels).sum())
                total += int(labels.shape[0])
        return correct / total if total else 0.0
