"""Hand-computed tests for src/evaluation/metrics.py.

Reference example (3 classes, 6 samples):

    targets      = [0, 0, 1, 1, 2, 2]
    predictions  = [0, 1, 1, 1, 2, 0]

    confusion (rows = true, cols = predicted):

                 pred 0   pred 1   pred 2
        true 0       1        1        0
        true 1       0        2        0
        true 2       1        0        1

    correct = 1 + 2 + 1 = 4  ->  accuracy = 4/6 = 2/3
    precision: c0 = 1/(1+1) = 1/2,  c1 = 2/(1+2) = 2/3,  c2 = 1/(0+1) = 1
    recall:    c0 = 1/2,             c1 = 2/2 = 1,        c2 = 1/2
    f1:        c0 = 2*(1/2*1/2)/(1/2+1/2) = 1/2
               c1 = 2*(2/3*1)/(2/3+1) = 4/5
               c2 = 2*(1*1/2)/(1+1/2) = 2/3
    macro precision = (1/2 + 2/3 + 1)/3 = 13/18
    macro recall    = (1/2 + 1 + 1/2)/3 = 2/3
    macro f1        = (1/2 + 4/5 + 2/3)/3 = 59/90
"""
from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.metrics import (
    accuracy_from_confusion,
    average_incremental_accuracy,
    confusion_matrix,
    precision_recall_f1,
    topk_correct,
)

TARGETS = [0, 0, 1, 1, 2, 2]
PREDICTIONS = [0, 1, 1, 1, 2, 0]


def test_confusion_matrix_hand_computed():
    matrix = confusion_matrix(TARGETS, PREDICTIONS, num_classes=3)
    expected = torch.tensor([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert torch.equal(matrix, expected)


def test_accuracy_from_confusion_hand_computed():
    matrix = confusion_matrix(TARGETS, PREDICTIONS, num_classes=3)
    assert accuracy_from_confusion(matrix) == pytest.approx(4 / 6)


def test_precision_recall_f1_hand_computed():
    matrix = confusion_matrix(TARGETS, PREDICTIONS, num_classes=3)
    result = precision_recall_f1(matrix)
    macro = result["macro"]
    assert macro["precision"] == pytest.approx(13 / 18)
    assert macro["recall"] == pytest.approx(2 / 3)
    assert macro["f1"] == pytest.approx(59 / 90)
    per_class = result["per_class"]
    assert per_class[0]["precision"] == pytest.approx(1 / 2)
    assert per_class[1]["f1"] == pytest.approx(4 / 5)
    assert per_class[2]["recall"] == pytest.approx(1 / 2)
    assert per_class[1]["support"] == 2


def test_macro_over_selected_classes_only():
    matrix = confusion_matrix(TARGETS, PREDICTIONS, num_classes=3)
    result = precision_recall_f1(matrix, classes=[0, 2])
    expected = (1 / 2 + 1) / 2
    assert result["macro"]["precision"] == pytest.approx(expected)


def test_zero_support_class_is_zero_not_nan():
    matrix = torch.tensor([[2, 0], [0, 0]])
    result = precision_recall_f1(matrix)
    assert result["per_class"][1]["recall"] == 0.0
    assert result["per_class"][1]["precision"] == 0.0
    assert result["per_class"][1]["f1"] == 0.0
    assert all(math.isfinite(v) for c in result["per_class"].values() for v in c.values() if isinstance(v, float))


def test_confusion_matrix_rejects_bad_labels():
    with pytest.raises(ValueError):
        confusion_matrix([0], [3], num_classes=3)
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0], num_classes=3)
    with pytest.raises(ValueError):
        confusion_matrix([0], [0], num_classes=0)


def test_topk_counts_ranked_hits():
    base = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    logits = torch.tensor([base, base, base])
    targets = torch.tensor([0, 2, 6])  # ranks 1, 3, 7
    counts = topk_correct(logits, targets, ks=(1, 3, 5, 10))
    assert counts == {1: 1, 3: 2, 5: 2, 10: 3}


def test_topk_k_at_least_num_classes_counts_everything():
    logits = torch.randn(20, 4)
    targets = torch.randint(0, 4, (20,))
    counts = topk_correct(logits, targets, ks=(1, 4, 99))
    assert counts[4] == counts[99] == 20
    assert 0 <= counts[1] <= 20


def test_topk_rejects_bad_shapes():
    with pytest.raises(ValueError):
        topk_correct(torch.randn(4, 5), torch.randn(4, 1), ks=(1,))
    with pytest.raises(ValueError):
        topk_correct(torch.randn(4, 5), torch.randint(0, 5, (3,)), ks=(1,))
    with pytest.raises(ValueError):
        topk_correct(torch.randn(4, 5), torch.randint(0, 5, (4,)), ks=(0,))


def test_average_incremental_hand_computed():
    assert average_incremental_accuracy([0.5, None, 0.75, 0.25]) == pytest.approx(0.5)
    assert average_incremental_accuracy([1.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        average_incremental_accuracy([])
    with pytest.raises(ValueError):
        average_incremental_accuracy([float("nan")])
