"""Metric primitives: top-k accuracy, confusion matrix, macro P/R/F1.

Conventions (kept consistent with :mod:`src.evaluation.continual`):

- top-1 accuracy = mean of per-sample ``argmax == target``;
- confusion matrix rows are TRUE classes, columns are PREDICTED classes;
- per-class precision/recall/F1 use the confusion matrix rows/columns with
  zero-division mapped to 0.0 (and a class with no true/predicted samples
  contributes 0 to its own metric, never NaN);
- macro metrics are unweighted means over the given class list;
- ``average_incremental_accuracy`` averages per-experience overall
  accuracies — the same definition as ``run_phase5_experiment.py``.

All functions are pure (no I/O, no model access) so every number can be
hand-verified in tests.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def topk_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int] = (1, 5),
) -> dict[int, int]:
    """Count samples whose target appears in the top-``k`` predictions.

    ``logits`` is ``(N, C)``, ``targets`` is ``(N,)``. For ``k >= C`` every
    sample counts (the target is always inside the full ranking), so such
    ``k`` degenerates to ``N``, not to top-1.
    Raises ``ValueError`` on shape mismatch or non-positive ``k``.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C), got shape {tuple(logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(
            f"targets must be shape ({logits.shape[0]},), got {tuple(targets.shape)}"
        )
    ks = tuple(int(k) for k in ks)
    if not ks or any(k < 1 for k in ks):
        raise ValueError(f"ks must be positive integers, got {ks!r}")
    count = logits.shape[1]
    max_k = min(max(ks), count)
    _, top = logits.topk(max_k, dim=1, largest=True, sorted=True)
    hits = top.eq(targets.unsqueeze(1))
    return {k: int(hits[:, : min(k, count)].any(dim=1).sum()) for k in ks}


def confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
) -> torch.Tensor:
    """Confusion matrix with rows = true class, columns = predicted class."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if len(targets) != len(predictions):
        raise ValueError(
            f"targets ({len(targets)}) and predictions ({len(predictions)}) "
            "must have the same length"
        )
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, prediction in zip(targets, predictions, strict=True):
        t = int(target)
        p = int(prediction)
        if not (0 <= t < num_classes) or not (0 <= p < num_classes):
            raise ValueError(
                f"label {t if not 0 <= t < num_classes else p} outside "
                f"[0, {num_classes})"
            )
        matrix[t, p] += 1
    return matrix


def precision_recall_f1(
    matrix: torch.Tensor,
    classes: Sequence[int] | None = None,
) -> dict[str, object]:
    """Per-class and macro precision/recall/F1 from a confusion matrix.

    Returns ``{"per_class": {class: {"precision","recall","f1","support"}},
    "macro": {"precision","recall","f1"}}``. ``classes`` selects which
    rows enter the macro average (default: all).
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion matrix must be square, got {tuple(matrix.shape)}")
    num_classes = matrix.shape[0]
    if classes is None:
        classes = list(range(num_classes))
    classes = [int(c) for c in classes]
    if any(c < 0 or c >= num_classes for c in classes):
        raise ValueError(f"class outside [0, {num_classes}): {classes}")

    per_class: dict[int, dict[str, float | int]] = {}
    for index in range(num_classes):
        tp = float(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class[index] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro = {
        "precision": sum(float(per_class[c]["precision"]) for c in classes) / len(classes),
        "recall": sum(float(per_class[c]["recall"]) for c in classes) / len(classes),
        "f1": sum(float(per_class[c]["f1"]) for c in classes) / len(classes),
    }
    return {"per_class": per_class, "macro": macro}


def accuracy_from_confusion(matrix: torch.Tensor) -> float:
    """Overall top-1 accuracy implied by a confusion matrix."""
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("empty confusion matrix")
    return float(matrix.diag().sum()) / total


def average_incremental_accuracy(per_experience_overalls: Sequence[float | None]) -> float:
    """Mean of per-experience overall accuracies (``None`` entries skipped).

    Same definition as ``run_phase5_experiment.average_incremental``: the
    unweighted mean of each experience's cumulative-classes accuracy.
    """
    values = [float(v) for v in per_experience_overalls if v is not None]
    if not values:
        raise ValueError("no non-None accuracy values to average")
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f"non-finite accuracy in {values!r}")
    return sum(values) / len(values)
