"""Unified continual-training API (Step 6 of Phase 4).

Phase 5 selects a method without touching training code::

    trainer = build_continual_trainer("naive", config)
    trainer = build_continual_trainer("replay", config)

Unknown method names fail immediately with a clear ``ValueError``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.training.base import BaseContinualTrainer
from src.training.config import ContinualTrainConfig
from src.training.naive import NaiveContinualTrainer
from src.training.replay import ReplayContinualTrainer

_REGISTRY: dict[str, type[BaseContinualTrainer]] = {
    "naive": NaiveContinualTrainer,
    "replay": ReplayContinualTrainer,
}


def supported_methods() -> tuple[str, ...]:
    """Method names accepted by :func:`build_continual_trainer`."""
    return tuple(_REGISTRY)


def build_continual_trainer(
    method: str | None = None,
    config: ContinualTrainConfig | dict[str, Any] | None = None,
) -> BaseContinualTrainer:
    """Build the continual trainer for ``method`` (``naive`` or ``replay``).

    ``method`` is authoritative; when omitted, the configuration decides
    (``replay.enabled: true`` selects replay, otherwise naive).
    """
    if config is None:
        cfg = ContinualTrainConfig()
    elif isinstance(config, ContinualTrainConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = ContinualTrainConfig.from_dict(config)
    else:
        raise ValueError(
            f"config must be a ContinualTrainConfig or mapping, got {type(config).__name__}"
        )

    if method is None:
        key = "replay" if cfg.replay_enabled else "naive"
    elif isinstance(method, str):
        key = method.strip().lower()
    else:
        key = ""
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown continual method {method!r}; "
            f"supported methods: {', '.join(_REGISTRY)}"
        )
    return _REGISTRY[key](replace(cfg, method=key))
