"""Phase-5 continual evaluation on the official CORe50 test sessions.

Provides:

- :class:`EvalCache` — an in-memory uint8 cache of the FIXED evaluation set
  (sessions 3/7/10). The same cache is reused for every experience and for
  both methods, so comparisons are made on identical inputs.
- :func:`evaluate_model` — one forward pass producing per-class counts.
- :func:`build_metric_record` — the per-experience metric record with the
  documented accuracy conventions (overall = cumulative seen classes,
  full = all classes, old/new = subsets, null when a subset is empty).
- :func:`compute_forgetting` — the Phase-5 forgetting definition:
  ``forgetting(c, t) = max_{t' < t} acc(c, t') - acc(c, t)``, aggregated at
  each experience as the mean over classes that have a prior measurement
  (``null`` at the first experience, where no prior measurement exists).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from src.data.continual import SampleRecord
from src.training.dataset import ContinualImageDataset

EvalProgress = Callable[[int, int], None]


class EvalCacheError(ValueError):
    """Raised when an evaluation cache is malformed or inconsistent."""


class EvalCache:
    """Cached official evaluation tensors (uint8, no file I/O at eval time)."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        paths: Sequence[str],
    ) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise EvalCacheError(
                f"images must have shape (N, 3, H, W), got {tuple(images.shape)}"
            )
        if images.dtype != torch.uint8:
            raise EvalCacheError(f"images must be uint8, got {images.dtype}")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise EvalCacheError("labels must be a 1-D tensor matching images")
        if len(paths) != images.shape[0]:
            raise EvalCacheError("paths must match the number of cached images")
        self.images = images
        self.labels = labels.to(torch.int64)
        self.paths = tuple(paths)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def image_size(self) -> int:
        return int(self.images.shape[2])

    @classmethod
    def from_records(
        cls,
        records: Sequence[SampleRecord],
        images_root: str | Path,
        image_size: int,
        *,
        on_progress: EvalProgress | None = None,
    ) -> "EvalCache":
        """Decode official evaluation records once (identical transform to
        the training pipeline: resize bilinear to ``image_size``, normalize
        by mean/std 0.5, then stored back as uint8 for exact round-trips)."""
        dataset = ContinualImageDataset(
            records, images_root, image_size, require_split="test"
        )
        images = torch.empty((len(dataset), 3, image_size, image_size), dtype=torch.uint8)
        labels = torch.empty(len(dataset), dtype=torch.int64)
        paths: list[str] = []
        for index in range(len(dataset)):
            tensor, label = dataset[index]
            images[index] = (
                ((tensor + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
            )
            labels[index] = label
            paths.append(dataset.records[index].relative_path)
            if on_progress is not None and (index % 2048 == 0 or index == len(dataset) - 1):
                on_progress(index + 1, len(dataset))
        return cls(images, labels, paths)

    def float_batch(self, start: int, end: int) -> torch.Tensor:
        """Normalized float32 batch matching the training preprocessing."""
        return self.images[start:end].to(torch.float32).div_(127.5).sub_(1.0)

    def save(self, path: str | Path) -> Path:
        """Atomically persist the cache (temporary file, then replace)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "format_version": 1,
                "images": self.images,
                "labels": self.labels,
                "paths": list(self.paths),
            },
            tmp,
        )
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "EvalCache":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise EvalCacheError(f"Unsupported eval cache format in {path}")
        return cls(payload["images"], payload["labels"], list(payload["paths"]))


def evaluate_model(
    model: torch.nn.Module,
    cache: EvalCache,
    *,
    num_classes: int,
    device: str,
    batch_size: int = 256,
    on_progress: EvalProgress | None = None,
) -> dict[str, Any]:
    """One forward pass over the fixed evaluation set.

    Returns per-class correct/total counts. The evaluation set is never
    used for training; this function only reads the cache.
    """
    if batch_size < 1:
        raise EvalCacheError(f"batch_size must be >= 1, got {batch_size}")
    correct = [0] * int(num_classes)
    total = [0] * int(num_classes)
    n = len(cache)
    model.eval()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            images = cache.float_batch(start, end).to(device)
            labels = cache.labels[start:end]
            predictions = model(images).argmax(dim=1).cpu()
            total_batch = torch.bincount(labels, minlength=num_classes)
            for class_id in range(num_classes):
                total[class_id] += int(total_batch[class_id])
            hits = predictions.eq(labels)
            correct_batch = torch.bincount(labels[hits], minlength=num_classes)
            for class_id in range(num_classes):
                correct[class_id] += int(correct_batch[class_id])
            if on_progress is not None:
                on_progress(end, n)
    return {
        "correct": correct,
        "total": total,
        "n": n,
        "overall_correct": int(sum(correct)),
    }


def _micro(correct: Sequence[int], total: Sequence[int], classes: Sequence[int]) -> float | None:
    hits = sum(correct[c] for c in classes)
    seen = sum(total[c] for c in classes)
    if seen == 0:
        return None
    return hits / seen


def build_metric_record(
    *,
    experience_id: int,
    train_samples: int,
    classes_introduced: Sequence[int],
    classes_seen: Sequence[int],
    previous_seen: Sequence[int],
    eval_result: dict[str, Any],
    train_stats: dict[str, Any],
    eval_seconds: float,
    replay_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one per-experience metric record.

    Conventions (documented in the Phase-5 report):

    - ``overall``: micro accuracy over evaluation samples whose class has
      been introduced up to this experience (cumulative seen classes).
    - ``full``: micro accuracy over ALL evaluation classes (informational).
    - ``old``: micro accuracy over classes seen BEFORE this experience;
      ``null`` when there are none (experience 0).
    - ``new``: micro accuracy over classes introduced BY this experience;
      ``null`` when this experience introduces none (most NIC experiences).
    - ``per_class``: per-class accuracy for seen classes, ``null`` for
      classes not yet introduced.
    """
    correct: list[int] = list(eval_result["correct"])
    total: list[int] = list(eval_result["total"])
    num_classes = len(correct)
    seen = sorted({int(c) for c in classes_seen})
    introduced = sorted({int(c) for c in classes_introduced})
    old_classes = sorted({int(c) for c in previous_seen})

    def subset_accuracy(classes: Sequence[int]) -> float | None:
        return _micro(correct, total, [c for c in classes if 0 <= c < num_classes])

    per_class: dict[str, float | None] = {}
    for class_id in range(num_classes):
        per_class[str(class_id)] = (
            subset_accuracy([class_id]) if class_id in set(seen) else None
        )

    record: dict[str, Any] = {
        "experience_id": int(experience_id),
        "train_samples": int(train_samples),
        "eval_samples": int(eval_result["n"]),
        "classes_introduced": introduced,
        "classes_seen": seen,
        "old_classes": old_classes,
        "accuracy": {
            "overall": subset_accuracy(seen),
            "full": subset_accuracy(list(range(num_classes))),
            "old": subset_accuracy(old_classes) if old_classes else None,
            "new": subset_accuracy(introduced) if introduced else None,
        },
        "per_class": per_class,
        "train": dict(train_stats),
        "eval_seconds": round(float(eval_seconds), 3),
        "replay": replay_stats,
    }
    return record


def compute_forgetting(records: Sequence[dict[str, Any]]) -> list[float | None]:
    """Per-experience aggregate forgetting (Phase-5 documented definition).

    ``forgetting(c, t) = max_{t' < t} acc(c, t') - acc(c, t)``; the value at
    experience ``t`` is the mean over every class that has a measurement
    from an earlier experience. ``None`` (N/A) when no prior measurement
    exists (the first experience). Negative values are preserved — they
    mean the class improved relative to its previous best.
    """
    previous_best: dict[int, float] = {}
    out: list[float | None] = []
    for record in records:
        per_class = record.get("per_class") or {}
        deltas: list[float] = []
        for class_id, best in previous_best.items():
            current = per_class.get(str(class_id))
            if current is None:
                continue
            deltas.append(best - float(current))
        out.append(sum(deltas) / len(deltas) if deltas else None)
        for class_id, value in per_class.items():
            if value is None:
                continue
            key = int(class_id)
            value = float(value)
            previous_best[key] = max(previous_best.get(key, value), value)
    return out


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON atomically (temporary file, then replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_class_names(mapping_path: str | Path) -> dict[str, str]:
    """Map 0-based label -> class name from the official object mapping."""
    raw = read_json(mapping_path)
    names: dict[str, str] = {}
    objects = raw.get("objects") if isinstance(raw, dict) else None
    if isinstance(objects, list):
        for entry in objects:
            if not isinstance(entry, dict) or "object_id" not in entry:
                continue
            label = int(entry["object_id"]) - 1  # NI/NIC: label == object_id - 1
            name = entry.get("name")
            if name is not None and label >= 0:
                names[str(label)] = str(name)
    return names
