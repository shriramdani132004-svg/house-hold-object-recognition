"""Validation tests for ``src.training.config`` (Phase-4 continual config).

Covers the dead-key cleanup, strict unknown-key rejection with a helpful
message, documented passthrough keys, per-field validation errors, and the
reproducibility defaults of ``ContinualTrainConfig``.
"""

from __future__ import annotations

from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.training.config import (
    ContinualConfigError,
    ContinualTrainConfig,
    _PASSTHROUGH_KEYS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RECIPE_LIKE = {
    "scenario": "NIC",
    "variant": "inc",
    "run": 0,
    "filelist_root": "data/raw/core50/filelists",
    "images_root": "data/raw/core50/dataset/core50_128x128",
    "object_mapping": "data/raw/core50/metadata/object_mapping.json",
    "manifest": "data/splits/nic_inc_run0_dev10_seed42.json",
    "method": "replay",
    "model_arch": "small_cnn",
    "model_width": 64,
    "image_size": 64,
    "epochs": 12,
    "early_stop_patience": 5,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adamw",
    "scheduler": "none",
    "replay_policy": "reservoir",
    "replay_capacity": 10000,
    "replay_batch_size": 32,
    "augment": True,
    "seed": 42,
}


def test_tensor_cache_dir_field_is_gone() -> None:
    field_names = {field.name for field in fields(ContinualTrainConfig)}
    assert "tensor_cache_dir" not in field_names
    assert not hasattr(ContinualTrainConfig(), "tensor_cache_dir")
    assert "tensor_cache_dir" not in ContinualTrainConfig().to_dict()
    with pytest.raises(ContinualConfigError, match="tensor_cache_dir"):
        ContinualTrainConfig.from_dict({"tensor_cache_dir": "models/cache"})


def test_from_yaml_classmethod_is_gone() -> None:
    assert not hasattr(ContinualTrainConfig, "from_yaml")
    assert "from_yaml" not in ContinualTrainConfig.__dict__


def test_from_dict_accepts_recipe_like_payload_with_passthrough_keys() -> None:
    cfg = ContinualTrainConfig.from_dict(dict(RECIPE_LIKE))
    assert cfg.method == "replay"
    assert cfg.model_arch == "small_cnn"
    assert cfg.model_width == 64
    assert cfg.replay_policy == "reservoir"
    assert cfg.replay_capacity == 10000
    assert cfg.augment is True
    stored = cfg.to_dict()
    for key in _PASSTHROUGH_KEYS:
        assert key not in stored


def test_from_dict_rejects_typo_key_and_lists_accepted_keys() -> None:
    with pytest.raises(ContinualConfigError) as excinfo:
        ContinualTrainConfig.from_dict({"learning_rate": 1e-3, "learning_rates": 1e-3})
    message = str(excinfo.value)
    assert "learning_rates" in message
    assert "learning_rate" in message
    assert "epochs" in message
    assert "continual" in message
    assert "object_mapping" in message


def test_from_dict_rejects_unknown_section_settings() -> None:
    with pytest.raises(ContinualConfigError) as excinfo:
        ContinualTrainConfig.from_dict({"training": {"epoch": 3}})
    assert "epoch" in str(excinfo.value)
    assert "epochs" in str(excinfo.value)

    with pytest.raises(ContinualConfigError) as excinfo:
        ContinualTrainConfig.from_dict({"replay": {"ratio": 4}})
    assert "ratio" in str(excinfo.value)
    assert "capacity" in str(excinfo.value)

    with pytest.raises(ContinualConfigError) as excinfo:
        ContinualTrainConfig.from_dict({"continual": {"strategy": "ewc"}})
    assert "strategy" in str(excinfo.value)
    assert "method" in str(excinfo.value)


def test_from_dict_rejects_duplicate_flat_and_section_keys() -> None:
    with pytest.raises(ContinualConfigError, match="specified twice"):
        ContinualTrainConfig.from_dict({"epochs": 2, "training": {"epochs": 3}})


def test_from_dict_requires_mapping() -> None:
    payload: Any = ["epochs", 1]
    with pytest.raises(ContinualConfigError, match="mapping"):
        ContinualTrainConfig.from_dict(payload)


def test_from_dict_builds_from_nested_sections() -> None:
    cfg = ContinualTrainConfig.from_dict(
        {
            "continual": {"method": "replay"},
            "training": {"epochs": 3, "batch_size": 64, "seed": 7},
            "replay": {"enabled": True, "capacity": 512, "batch_size": 8, "policy": "fifo"},
        }
    )
    assert cfg.method == "replay"
    assert cfg.epochs == 3
    assert cfg.seed == 7
    assert cfg.replay_enabled is True
    assert cfg.replay_capacity == 512
    assert cfg.replay_batch_size == 8


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"replay_capacity": -1}, "replay.capacity"),
        ({"replay_capacity": 0}, "replay.capacity"),
        ({"optimizer": "lion"}, "optimizer"),
        ({"method": "ewc"}, "method"),
        ({"model_arch": "vit"}, "model_arch"),
        ({"replay_policy": "ring"}, "replay.policy"),
        ({"epochs": 0}, "epochs"),
        ({"epochs": -3}, "epochs"),
        ({"early_stop_patience": -1}, "early_stop_patience"),
        ({"early_stop_patience": 0}, "early_stop_patience"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": -8}, "batch_size"),
        ({"momentum": 1.5}, "momentum"),
        ({"momentum": -0.1}, "momentum"),
        ({"replay_enabled": "yes"}, "replay.enabled"),
        ({"scheduler": "cosine"}, "scheduler"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"device": "tpu"}, "device"),
        ({"augment": "true"}, "augment"),
        ({"max_steps_per_epoch": -1}, "max_steps_per_epoch"),
    ],
)
def test_invalid_values_raise_with_helpful_message(
    override: dict[str, object], fragment: str
) -> None:
    payload = {**RECIPE_LIKE, **override}
    with pytest.raises(ContinualConfigError) as excinfo:
        ContinualTrainConfig.from_dict(payload)
    assert fragment in str(excinfo.value)


def test_to_dict_round_trips_through_from_dict() -> None:
    cfg = ContinualTrainConfig.from_dict(dict(RECIPE_LIKE))
    assert ContinualTrainConfig.from_dict(cfg.to_dict()) == cfg
    default = ContinualTrainConfig()
    assert ContinualTrainConfig.from_dict(default.to_dict()) == default


def test_documented_reproducibility_defaults_exist() -> None:
    cfg = ContinualTrainConfig()
    assert cfg.seed == 42
    assert cfg.model_arch == "small_cnn"
    assert cfg.augment is False
    assert cfg.early_stop_patience is None
    assert cfg.replay_policy == "fifo"
    assert cfg.method == "naive"


def test_every_field_has_an_explicit_default() -> None:
    for field in fields(ContinualTrainConfig):
        assert field.default is not MISSING, f"{field.name} has no default"


def test_real_phase4_config_still_loads() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "continual.yaml").read_text(encoding="utf-8")
    )
    cfg = ContinualTrainConfig.from_dict(payload)
    assert cfg.method == "naive"
    assert cfg.epochs == 1
    assert cfg.checkpoint_root == "models/continual"


def test_real_phase5_sections_still_load() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase5_nic.yaml").read_text(encoding="utf-8")
    )
    sections = {
        "continual": {"method": "replay"},
        "training": dict(payload["training"]),
        "replay": dict(payload["replay"]),
    }
    cfg = ContinualTrainConfig.from_dict(sections)
    assert cfg.epochs == 3
    assert cfg.replay_enabled is True
    assert cfg.replay_capacity == 2000
    assert cfg.replay_batch_size == 16


def test_every_real_candidate_recipe_still_loads() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "model_improvement_candidates.yaml").read_text(
            encoding="utf-8"
        )
    )
    candidates = payload["candidates"]
    assert candidates
    for entry in candidates:
        flat = {
            key: value
            for key, value in entry.items()
            if key not in {"id", "description"}
        }
        cfg = ContinualTrainConfig.from_dict(flat)
        assert cfg.method == "replay"
        assert cfg.epochs >= 1
        assert cfg.replay_capacity >= 1
