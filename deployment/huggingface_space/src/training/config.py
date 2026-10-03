"""Explicit configuration for continual training (Phase 4).

Every hyper-parameter used by the continual trainers is declared here so an
experiment can be reproduced from configuration alone. Configuration may be
built from the nested sections in ``configs/continual.yaml``
(``continual`` / ``training`` / ``replay``) or from a flat mapping of field
names. Phase-3 data keys are recognised and ignored when the whole YAML
file is loaded.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SUPPORTED_METHODS = ("naive", "replay")
SUPPORTED_OPTIMIZERS = ("adam", "sgd")
SUPPORTED_SCHEDULERS = ("none", "step")
SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")

# Phase-3 data-pipeline keys that may share configs/continual.yaml.
_PHASE3_KEYS = frozenset(
    {
        "scenario",
        "variant",
        "run",
        "filelist_root",
        "images_root",
        "object_mapping",
        "manifest",
        "check_paths_exist",
    }
)

_REPLAY_SECTION_MAP = {
    "enabled": "replay_enabled",
    "capacity": "replay_capacity",
    "batch_size": "replay_batch_size",
    "seed": "replay_seed",
}


class ContinualConfigError(ValueError):
    """Raised when a continual training configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinualConfigError(message)


def resolve_device(device: str) -> str:
    """Resolve a configured device name to a concrete torch device string."""
    import torch

    name = device.strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    _require(name in SUPPORTED_DEVICES, f"Unknown device {device!r}")
    if name == "cuda":
        _require(torch.cuda.is_available(), "device 'cuda' requested but CUDA is unavailable")
    if name == "mps":
        available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        _require(bool(available), "device 'mps' requested but MPS is unavailable")
    return name


@dataclass(frozen=True)
class ContinualTrainConfig:
    """Validated, explicit settings for one continual training run."""

    method: str = "naive"
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.9
    scheduler: str = "none"
    step_size: int = 10
    gamma: float = 0.1
    seed: int = 42
    device: str = "auto"
    workers: int = 0
    image_size: int = 64
    model_width: int = 32
    max_steps_per_epoch: int | None = None
    checkpoint_root: str = "models/continual"
    replay_enabled: bool = False
    replay_capacity: int = 2000
    replay_batch_size: int = 16
    replay_seed: int | None = None

    def __post_init__(self) -> None:
        _require(
            self.method in SUPPORTED_METHODS,
            f"Unknown method {self.method!r}; supported: {', '.join(SUPPORTED_METHODS)}",
        )
        _require(
            isinstance(self.epochs, int) and self.epochs >= 1,
            f"training.epochs must be an integer >= 1, got {self.epochs!r}",
        )
        _require(
            isinstance(self.batch_size, int) and self.batch_size >= 1,
            f"training.batch_size must be an integer >= 1, got {self.batch_size!r}",
        )
        _require(
            isinstance(self.learning_rate, (int, float)) and self.learning_rate > 0,
            f"training.learning_rate must be > 0, got {self.learning_rate!r}",
        )
        _require(
            isinstance(self.weight_decay, (int, float)) and self.weight_decay >= 0,
            f"training.weight_decay must be >= 0, got {self.weight_decay!r}",
        )
        _require(
            self.optimizer in SUPPORTED_OPTIMIZERS,
            f"Unknown optimizer {self.optimizer!r}; supported: {', '.join(SUPPORTED_OPTIMIZERS)}",
        )
        _require(
            self.scheduler in SUPPORTED_SCHEDULERS,
            f"Unknown scheduler {self.scheduler!r}; supported: {', '.join(SUPPORTED_SCHEDULERS)}",
        )
        _require(
            isinstance(self.step_size, int) and self.step_size >= 1,
            f"training.step_size must be an integer >= 1, got {self.step_size!r}",
        )
        _require(
            isinstance(self.gamma, (int, float)) and 0 < self.gamma <= 1,
            f"training.gamma must be in (0, 1], got {self.gamma!r}",
        )
        _require(
            isinstance(self.seed, int),
            f"training.seed must be an integer, got {self.seed!r}",
        )
        _require(
            self.device in SUPPORTED_DEVICES,
            f"Unknown device {self.device!r}; supported: {', '.join(SUPPORTED_DEVICES)}",
        )
        _require(
            isinstance(self.workers, int) and self.workers >= 0,
            f"training.workers must be an integer >= 0, got {self.workers!r}",
        )
        _require(
            isinstance(self.image_size, int) and self.image_size >= 8,
            f"training.image_size must be an integer >= 8, got {self.image_size!r}",
        )
        _require(
            isinstance(self.model_width, int) and self.model_width >= 4,
            f"training.model_width must be an integer >= 4, got {self.model_width!r}",
        )
        _require(
            self.max_steps_per_epoch is None
            or (isinstance(self.max_steps_per_epoch, int) and self.max_steps_per_epoch >= 0),
            f"training.max_steps_per_epoch must be None or an integer >= 0, "
            f"got {self.max_steps_per_epoch!r}",
        )
        _require(
            isinstance(self.checkpoint_root, str) and self.checkpoint_root.strip() != "",
            "training.checkpoint_root must be a non-empty path string",
        )
        _require(
            isinstance(self.replay_capacity, int) and self.replay_capacity >= 1,
            f"replay.capacity must be an integer >= 1, got {self.replay_capacity!r}",
        )
        _require(
            isinstance(self.replay_batch_size, int) and self.replay_batch_size >= 1,
            f"replay.batch_size must be an integer >= 1, got {self.replay_batch_size!r}",
        )
        _require(
            self.replay_seed is None or isinstance(self.replay_seed, int),
            f"replay.seed must be an integer or null, got {self.replay_seed!r}",
        )

    @property
    def resolved_replay_seed(self) -> int:
        return self.seed if self.replay_seed is None else self.replay_seed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContinualTrainConfig":
        """Build a config from nested sections or a flat field mapping."""
        if not isinstance(payload, dict):
            raise ContinualConfigError(f"Configuration must be a mapping, got {type(payload).__name__}")

        values: dict[str, Any] = {}
        field_names = {f.name for f in fields(cls)}

        for key, value in payload.items():
            if key in _PHASE3_KEYS:
                continue
            if key in ("continual", "training", "replay"):
                if not isinstance(value, dict):
                    raise ContinualConfigError(f"Configuration section {key!r} must be a mapping")
                continue
            if key in field_names:
                values[key] = value
            else:
                raise ContinualConfigError(
                    f"Unknown configuration key {key!r} "
                    f"(supported sections: continual, training, replay)"
                )

        for section in ("continual", "training", "replay"):
            block = payload.get(section)
            if block is None:
                continue
            if not isinstance(block, dict):
                raise ContinualConfigError(f"Configuration section {section!r} must be a mapping")
            for key, value in block.items():
                if section == "replay":
                    target = _REPLAY_SECTION_MAP.get(key)
                    if target is None:
                        raise ContinualConfigError(f"Unknown replay setting {key!r}")
                elif section == "continual":
                    if key != "method":
                        raise ContinualConfigError(f"Unknown continual setting {key!r}")
                    target = "method"
                else:
                    if key not in field_names:
                        raise ContinualConfigError(f"Unknown training setting {key!r}")
                    target = key
                if target in values:
                    raise ContinualConfigError(f"Configuration key {target!r} specified twice")
                values[target] = value

        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ContinualTrainConfig":
        import yaml

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ContinualConfigError(f"Expected a mapping in {config_path}")
        return cls.from_dict(payload)


def config_fingerprint(config: ContinualTrainConfig) -> str:
    """Stable short fingerprint used for state compatibility checks."""
    import hashlib
    import json

    blob = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def looks_absolute(value: str) -> bool:
    """True for machine-specific path strings that must not be persisted."""
    if value.startswith("~"):
        return True
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        return True
    return os.path.isabs(value)
