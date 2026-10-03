"""Model-contract tests: ``assert_model_contract`` and ``build_model`` validation.

Fast, synthetic-only checks — no datasets, no checkpoints, no training.
Every test finishes in milliseconds on CPU.
"""

from __future__ import annotations

import logging
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.model import (
    MODEL_CONTRACT,
    ModelContractError,
    assert_model_contract,
    build_model,
    seed_everything,
)

ARCHS = ("small_cnn", "compact_resnet")


class WrongHeadModel(nn.Module):
    """Classifier head that emits the wrong number of classes."""

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, out_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(images).flatten(1))


class NanOutputModel(nn.Module):
    """Correct shape, non-finite values."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.full((images.shape[0], 50), float("nan"))


@pytest.mark.parametrize("arch", ARCHS)
def test_contract_passes_for_built_models(arch: str) -> None:
    model = build_model(50, seed=3, arch=arch)
    model.train()
    summary = assert_model_contract(model, num_classes=50, image_size=64)
    assert summary["output_shape"] == (2, 50)
    assert summary["num_classes"] == 50
    assert summary["param_count"] == sum(p.numel() for p in model.parameters())
    assert int(summary["param_count"]) > 0
    assert model.training is True
    model.eval()
    assert_model_contract(model, num_classes=50, image_size=64)
    assert model.training is False


def test_contract_fails_on_wrong_output_classes() -> None:
    model = WrongHeadModel(10)
    model.train()
    with pytest.raises(ModelContractError, match="shape"):
        assert_model_contract(model, num_classes=50, image_size=64)
    assert model.training is True


def test_contract_fails_on_non_finite_outputs() -> None:
    with pytest.raises(ModelContractError, match="non-finite"):
        assert_model_contract(NanOutputModel(), num_classes=50, image_size=64)


@pytest.mark.parametrize("bad_classes", [0, -3])
def test_contract_rejects_non_positive_num_classes(bad_classes: int) -> None:
    model = build_model(4, width=4, seed=1)
    with pytest.raises(ModelContractError, match="positive"):
        assert_model_contract(model, num_classes=bad_classes, image_size=64)


def test_build_model_rejects_non_positive_num_classes() -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            build_model(bad)


def test_build_model_rejects_non_positive_width() -> None:
    with pytest.raises(ValueError, match="width"):
        build_model(4, width=0)


def test_build_model_rejects_unknown_arch_listing_supported() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_model(10, arch="resnet50")
    message = str(excinfo.value)
    assert "supported" in message
    for arch in ARCHS:
        assert arch in message


def test_model_contract_documented() -> None:
    assert isinstance(MODEL_CONTRACT, str)
    assert "num_classes" in MODEL_CONTRACT


def test_seed_everything_logs_when_numpy_seeding_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="src.training.model"):
        with mock.patch.object(
            np.random, "seed", side_effect=RuntimeError("numpy unavailable")
        ):
            seed_everything(7)
    messages = [record.getMessage() for record in caplog.records]
    assert any("numpy seeding" in message.lower() for message in messages)
