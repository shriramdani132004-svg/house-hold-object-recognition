"""Continual training components (Phase 4): naive + experience replay.

Public API::

    from src.training import build_continual_trainer, ContinualTrainConfig

    config = ContinualTrainConfig(method="replay", epochs=1)
    trainer = build_continual_trainer(config.method, config)
    state = trainer.initialize(scenario)          # Phase-3 ContinualScenario
    for experience in scenario.experiences:
        state = trainer.train_experience(state, experience)

Both methods share one interface, persist model/optimizer/scheduler state
under ``models/continual/`` (outside the dataset tree, never committed),
and read experiences exclusively from the Phase-3 official pipeline.
"""

from __future__ import annotations

from src.training.base import BaseContinualTrainer, ContinualTrainingError
from src.training.config import (
    ContinualConfigError,
    ContinualTrainConfig,
    config_fingerprint,
    resolve_device,
)
from src.training.dataset import ContinualImageDataset, make_loader
from src.training.factory import build_continual_trainer, supported_methods
from src.training.model import SmallConvNet, build_model, model_checksum, seed_everything
from src.training.naive import NaiveContinualTrainer
from src.training.replay import (
    ReplayContinualTrainer,
    ReplayMemory,
    ReplayMemoryError,
)
from src.training.state import (
    CheckpointMissingError,
    ContinualStateError,
    ContinualTrainingState,
    StateCompatibilityError,
    checkpoint_exists,
    load_state,
    save_state,
    validate_state,
)

__all__ = [
    "BaseContinualTrainer",
    "CheckpointMissingError",
    "ContinualConfigError",
    "ContinualStateError",
    "ContinualTrainConfig",
    "ContinualTrainingError",
    "ContinualTrainingState",
    "ContinualImageDataset",
    "NaiveContinualTrainer",
    "ReplayContinualTrainer",
    "ReplayMemory",
    "ReplayMemoryError",
    "SmallConvNet",
    "StateCompatibilityError",
    "build_continual_trainer",
    "build_model",
    "checkpoint_exists",
    "config_fingerprint",
    "load_state",
    "make_loader",
    "model_checksum",
    "resolve_device",
    "save_state",
    "seed_everything",
    "supported_methods",
    "validate_state",
]
