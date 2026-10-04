"""Diagnosis-driven Experience Replay improvements (model-improvement work).

Adds, without touching the Phase-5 defaults:

- :class:`AugmentedCachedDataset` — official training preprocessing served
  from a decoded :class:`~src.training.tensor_cache.TensorCache` (all pixel
  work delegates to the ONE authoritative :mod:`src.data.preprocessing`
  implementation), with optional light, semantically safe augmentation
  (horizontal flip, small rotation/translation, mild brightness/contrast,
  gaussian noise) applied only during training.
- :class:`ImprovedReplayTrainer` — the standard
  :class:`~src.training.replay.ReplayContinualTrainer` plus
  development-validation early stopping with EXPLICIT best-epoch
  selection, and an optional decoded cache for speed.

Early-stopping rule (documented, tested): after every epoch the model is
evaluated on the development subset for the current experience; the best
epoch is any accuracy improving on ``best + 1e-4``; training for the
experience stops after ``early_stop_patience`` consecutive non-improving
epochs; when the experience ends, the BEST epoch's weights are restored
(so the next experience continues from the selected checkpoint, not the
last one). Optimizer moments are NOT rewound — only model weights are
restored. When ``best_checkpoint_path`` is given, every global
improvement of the development signal is written there atomically, and an
existing file re-establishes the threshold across process resumes.

Experience Replay remains the only anti-forgetting method; only the
memory policy (FIFO vs reservoir), capacity, ratio, duration and the
model configuration vary between candidates.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.continual import SampleRecord
from src.data.preprocessing import decode_image_file, normalize_chw
from src.training.base import ProgressCallback
from src.training.early_stop import EarlyStopping
from src.training.replay import ReplayContinualTrainer
from src.training.tensor_cache import TensorCache

DevProvider = Callable[[int], Sequence[SampleRecord]]

BEST_CHECKPOINT_VERSION = 1


def _augment(image: torch.Tensor) -> torch.Tensor:
    """Light, semantically safe augmentation on a normalized CHW tensor.

    Horizontal flip, sub-3-degree rotation (±3°), translation up to 2 px,
    mild brightness/contrast, and additive gaussian noise (sigma 0.02) —
    every transform preserves the object identity of the CORe50 scenes.
    """
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
    if torch.rand(()) < 0.5:
        angle = math.radians(3.0) * (2.0 * float(torch.rand(())) - 1.0)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=image.dtype
        ).unsqueeze(0)
        grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
        image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )[0]
    contrast = 1.0 + 0.2 * (float(torch.rand(())) - 0.5)
    brightness = 0.1 * (float(torch.rand(())) - 0.5)
    image = image * contrast + brightness
    if torch.rand(()) < 0.5:
        image = image + 0.02 * torch.randn_like(image)
    return image.clamp_(-1.0, 1.0)


class AugmentedCachedDataset(Dataset):
    """Official training records with cache-backed decoding + training aug.

    With ``augment=False`` the output is bit-identical to
    :class:`src.training.dataset.ContinualImageDataset` (both use the
    shared :mod:`src.data.preprocessing` pipeline).
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
            uint8 = decode_image_file(path, self.image_size)
        image = normalize_chw(uint8)
        if self.augment:
            image = _augment(image)
        return image, int(record.label)


class ImprovedReplayTrainer(ReplayContinualTrainer):
    """Experience Replay trainer with dev-validation early stopping.

    ``dev_provider(experience_id)`` returns the development validation
    records for the early-stopping signal (development data only, never
    the official held-out sessions). With ``early_stop_patience=None``
    or no provider, behaviour matches the standard trainer exactly.
    """

    def __init__(
        self,
        config=None,
        *,
        on_progress: ProgressCallback | None = None,
        cache: TensorCache | None = None,
        dev_provider: DevProvider | None = None,
        best_checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__(config, on_progress=on_progress, cache=cache)
        self._dev_provider = dev_provider
        self._stoppers: dict[int, EarlyStopping] = {}
        self._exp_best_state: dict[str, torch.Tensor] | None = None
        self._best_checkpoint_path = (
            Path(best_checkpoint_path) if best_checkpoint_path is not None else None
        )
        self._best_dev_value: float | None = None
        self._best_dev_meta: dict[str, Any] = {}
        if self._best_checkpoint_path is not None and self._best_checkpoint_path.is_file():
            previous = torch.load(
                self._best_checkpoint_path, map_location="cpu", weights_only=False
            )
            if isinstance(previous, dict) and "dev_value" in previous:
                self._best_dev_value = float(previous["dev_value"])
                self._best_dev_meta = {
                    key: previous.get(key)
                    for key in ("experience", "epoch", "saved_utc")
                }
        self.on_dev_metrics: Callable[[dict[str, Any]], None] | None = None

    # ------------------------------------------------------------------
    # early stopping / best-checkpoint selection
    # ------------------------------------------------------------------
    @property
    def best_development_value(self) -> float | None:
        return self._best_dev_value

    @property
    def best_development_meta(self) -> dict[str, Any]:
        return dict(self._best_dev_meta)

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

    def _on_experience_start(self, experience) -> None:
        patience = self.config.early_stop_patience
        if patience is not None:
            self._stoppers[experience.experience_id] = EarlyStopping(patience=int(patience))
        self._exp_best_state = None

    def _on_epoch_end(self, experience, epoch: int, stats: dict[str, Any]) -> bool:
        patience = self.config.early_stop_patience
        if patience is None or self._dev_provider is None:
            return False
        dev_records = self._dev_provider(experience.experience_id)
        if not dev_records:
            return False
        accuracy = self._evaluate_records(dev_records)
        stopper = self._stoppers.get(experience.experience_id)
        if stopper is None:
            stopper = EarlyStopping(patience=int(patience))
            self._stoppers[experience.experience_id] = stopper
        stop = stopper.observe(accuracy, epoch)
        is_best = stopper.best_epoch == epoch
        if is_best:
            assert self._model is not None
            self._exp_best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self._model.state_dict().items()
            }
            if (
                self._best_dev_value is None
                or accuracy > self._best_dev_value + stopper.min_delta
            ):
                self._best_dev_value = float(accuracy)
                self._best_dev_meta = {
                    "experience": int(experience.experience_id),
                    "epoch": int(epoch + 1),
                }
                self._save_best_checkpoint(accuracy, experience.experience_id, epoch)
        if self.on_dev_metrics is not None:
            self.on_dev_metrics(
                {
                    "event": "dev_epoch",
                    "experience": experience.experience_id,
                    "epoch": epoch + 1,
                    "dev_accuracy": accuracy,
                    "best_dev_accuracy": stopper.best_value,
                    "non_improving_epochs": stopper.bad_epochs,
                    "is_best": is_best,
                    "train_loss": stats.get("loss"),
                    "train_accuracy": stats.get("accuracy"),
                }
            )
        return stop

    def _finalize_experience_weights(self, experience) -> None:
        if self._exp_best_state is None:
            return
        assert self._model is not None
        self._model.load_state_dict(self._exp_best_state)
        self._exp_best_state = None

    def _save_best_checkpoint(self, dev_value: float, experience_id: int, epoch: int) -> None:
        if self._best_checkpoint_path is None:
            return
        assert self._model is not None
        payload = {
            "format_version": BEST_CHECKPOINT_VERSION,
            "selection_rule": (
                "development-subset top-1 accuracy over classes seen so far; "
                "best epoch per experience, best across experiences"
            ),
            "dev_value": float(dev_value),
            "experience": int(experience_id),
            "epoch": int(epoch + 1),
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_size": int(self.config.image_size),
            "arch": self.config.model_arch,
            "num_classes": int(getattr(self._model, "num_classes", 0)),
            "model_state": {
                name: tensor.detach().cpu()
                for name, tensor in self._model.state_dict().items()
            },
        }
        self._best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._best_checkpoint_path.with_suffix(
            self._best_checkpoint_path.suffix + ".tmp"
        )
        torch.save(payload, tmp)
        tmp.replace(self._best_checkpoint_path)

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
