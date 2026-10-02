"""Shared sequential-training machinery for the Phase-4 continual trainers.

Both methods follow the same contract:

``initialize(scenario)`` → first state (model created exactly once) and
``train_experience(state, experience)`` → next state. Every experience
*reloads* model/optimizer/scheduler (and replay memory, for the replay
method) from the checkpoint written by the previous experience, so state
genuinely persists across experiences and survives process restarts.
Official experience order is enforced: experience ``i + 1`` may only follow
experience ``i``, experience 0 first, and only the current experience's
official training samples are ever read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, ClassVar

import torch
import torch.nn as nn

from src.data.continual import ContinualExperience, ContinualScenario
from src.training.config import (
    ContinualTrainConfig,
    config_fingerprint,
    resolve_device,
)
from src.training.dataset import ContinualImageDataset, make_loader
from src.training.model import build_model, seed_everything
from src.training.state import (
    PAYLOAD_VERSION,
    ContinualTrainingState,
    StateCompatibilityError,
    load_state,
    save_state,
    validate_state,
)

ProgressCallback = Callable[[str, int, int], None]


class ContinualTrainingError(Exception):
    """Raised when the continual training contract is violated."""


class BaseContinualTrainer:
    """Sequential trainer base; subclasses supply method-specific hooks."""

    method_name: ClassVar[str] = ""

    def __init__(
        self,
        config: ContinualTrainConfig | dict | None = None,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        if config is None:
            config = ContinualTrainConfig()
        elif isinstance(config, dict):
            config = ContinualTrainConfig.from_dict(config)
        elif not isinstance(config, ContinualTrainConfig):
            raise ContinualTrainingError(
                f"config must be a ContinualTrainConfig or mapping, got {type(config).__name__}"
            )
        if config.method != self.method_name:
            raise ContinualTrainingError(
                f"{type(self).__name__} requires method={self.method_name!r}, "
                f"got {config.method!r} (use build_continual_trainer to select a method)"
            )
        self.config = config
        self.on_progress = on_progress
        self.device = resolve_device(config.device)
        self._scenario: ContinualScenario | None = None
        self._run_dir: Path | None = None
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None
        self._fingerprint = config_fingerprint(config)

    # ------------------------------------------------------------------
    # hooks for subclasses (replay)
    # ------------------------------------------------------------------
    def _extra_payload(self) -> dict[str, Any]:
        """Method-specific payload entries persisted with the checkpoint."""
        return {}

    def _restore_extra(self, payload: dict[str, Any]) -> None:
        """Restore method-specific state from a checkpoint payload."""

    def _on_experience_trained(self, experience: ContinualExperience) -> None:
        """Called after the experience's epochs, before the state is saved."""

    def _replay_tensors(
        self, experience_id: int, epoch: int, step: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Optional extra (replay) batch concatenated with the current one."""
        return None

    def _replay_metadata(self) -> dict[str, Any] | None:
        """Serializable replay summary stored in ``state.replay``."""
        return None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def scenario(self) -> ContinualScenario | None:
        return self._scenario

    @staticmethod
    def _num_classes(scenario: ContinualScenario) -> int:
        label_min = int(scenario.metadata.get("label_min", 0))
        label_max = int(scenario.metadata.get("label_max", -1))
        if label_min != 0 or label_max < 0:
            raise ContinualTrainingError(
                f"Scenario metadata must expose a 0-based label range, got "
                f"label_min={label_min} label_max={label_max}"
            )
        return label_max + 1

    def _resolve_run_dir(self, checkpoint_dir: str | Path | None) -> Path:
        if checkpoint_dir is not None:
            return Path(checkpoint_dir)
        scenario = self._scenario
        assert scenario is not None
        name = f"{scenario.scenario_type}_{scenario.variant}_run{scenario.run_id}"
        return Path(self.config.checkpoint_root) / name / self.method_name

    def initialize(
        self,
        scenario: ContinualScenario,
        *,
        checkpoint_dir: str | Path | None = None,
    ) -> ContinualTrainingState:
        """Create the model exactly once and persist the initial state."""
        if not isinstance(scenario, ContinualScenario):
            raise ContinualTrainingError(
                f"scenario must be a ContinualScenario, got {type(scenario).__name__}"
            )
        if not scenario.experiences:
            raise ContinualTrainingError(
                f"Scenario {scenario.name!r} has no experiences to train on"
            )
        self._scenario = scenario
        self._run_dir = self._resolve_run_dir(checkpoint_dir)

        seed_everything(self.config.seed)
        num_classes = self._num_classes(scenario)
        self._model = build_model(
            num_classes, width=self.config.model_width, seed=self.config.seed
        ).to(self.device)
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler()
        self._restore_extra({})

        state = ContinualTrainingState(
            scenario=scenario.scenario_type,
            variant=scenario.variant,
            run_id=scenario.run_id,
            method=self.method_name,
            current_experience=-1,
            seed=self.config.seed,
            config_fingerprint=self._fingerprint,
            training_config=self.config.to_dict(),
            replay=self._replay_metadata(),
        )
        payload = self._build_payload(state)
        save_state(state, self._run_dir, payload=payload)
        self._emit("initialize", 1, 1)
        return state

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        assert self._model is not None
        if cfg.optimizer == "adam":
            return torch.optim.Adam(
                self._model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.SGD(
            self._model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    def _build_scheduler(self) -> Any:
        cfg = self.config
        assert self._optimizer is not None
        if cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self._optimizer, step_size=cfg.step_size, gamma=cfg.gamma
            )
        return None

    # ------------------------------------------------------------------
    # checkpoint payload
    # ------------------------------------------------------------------
    def _build_payload(self, state: ContinualTrainingState) -> dict[str, Any]:
        assert self._model is not None
        assert self._optimizer is not None
        payload: dict[str, Any] = {
            "format_version": PAYLOAD_VERSION,
            "method": self.method_name,
            "current_experience": int(state.current_experience),
            "model_state": self._model.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "scheduler_state": self._scheduler.state_dict() if self._scheduler else None,
            "seed": int(self.config.seed),
            "num_classes": int(getattr(self._model, "num_classes", 0)),
        }
        payload.update(self._extra_payload())
        return payload

    def _restore(self, payload: dict[str, Any]) -> None:
        if payload.get("method") != self.method_name:
            raise StateCompatibilityError(
                f"Checkpoint was written by method {payload.get('method')!r}, "
                f"but this trainer is {self.method_name!r}"
            )
        assert self._model is not None
        self._model.load_state_dict(payload["model_state"])
        assert self._optimizer is not None
        self._optimizer.load_state_dict(payload["optimizer_state"])
        scheduler_state = payload.get("scheduler_state")
        if self._scheduler is not None and scheduler_state is not None:
            self._scheduler.load_state_dict(scheduler_state)
        self._restore_extra(payload)

    # ------------------------------------------------------------------
    # training one experience
    # ------------------------------------------------------------------
    def train_experience(
        self,
        state: ContinualTrainingState,
        experience: ContinualExperience | None = None,
    ) -> ContinualTrainingState:
        """Train the *next* official experience and persist the new state."""
        if state is None:
            raise ContinualTrainingError("state is required; call initialize() first")
        if self._scenario is None or self._run_dir is None:
            raise ContinualTrainingError("Trainer is not initialized; call initialize() first")
        validate_state(
            state,
            scenario=self._scenario.scenario_type,
            variant=self._scenario.variant,
            run_id=self._scenario.run_id,
            method=self.method_name,
            config_fingerprint=self._fingerprint,
        )

        expected_id = state.current_experience + 1
        if experience is None:
            experience = self._scenario.get_experience(expected_id)
        else:
            if not isinstance(experience, ContinualExperience):
                raise ContinualTrainingError(
                    f"experience must be a ContinualExperience, got {type(experience).__name__}"
                )
            expected = self._scenario.get_experience(experience.experience_id)
            if (
                experience.scenario_name != expected.scenario_name
                or experience.run_id != expected.run_id
            ):
                raise ContinualTrainingError(
                    "Experience belongs to a different scenario/run than this trainer"
                )
        if experience.experience_id != expected_id:
            raise ContinualTrainingError(
                f"Official order violated: experience {experience.experience_id} cannot "
                f"follow experience {state.current_experience} "
                f"(expected experience {expected_id}; future experiences must not "
                f"be trained early)"
            )

        # Load previous model/optimizer/scheduler (and replay memory) from disk.
        loaded_state, payload = load_state(self._run_dir)
        validate_state(
            loaded_state,
            scenario=self._scenario.scenario_type,
            variant=self._scenario.variant,
            run_id=self._scenario.run_id,
            method=self.method_name,
            config_fingerprint=self._fingerprint,
        )
        if loaded_state.current_experience != state.current_experience:
            raise StateCompatibilityError(
                f"On-disk state is at experience {loaded_state.current_experience} "
                f"but the caller passed experience {state.current_experience}"
            )
        if payload.get("current_experience") != state.current_experience:
            raise StateCompatibilityError(
                f"Checkpoint payload is at experience "
                f"{payload.get('current_experience')!r} but the caller passed "
                f"experience {state.current_experience}"
            )
        self._restore(payload)
        assert self._model is not None
        self._model.to(self.device)

        experience_id = experience.experience_id
        seed_everything(self.config.seed + 1009 * (experience_id + 1))
        steps = self._train_epochs(experience)
        self._on_experience_trained(experience)

        new_state = ContinualTrainingState(
            scenario=state.scenario,
            variant=state.variant,
            run_id=state.run_id,
            method=state.method,
            current_experience=experience_id,
            experiences_trained=state.experiences_trained + (experience_id,),
            epochs_completed=state.epochs_completed + int(self.config.epochs),
            steps_completed=state.steps_completed + steps,
            seed=state.seed,
            config_fingerprint=state.config_fingerprint,
            training_config=state.training_config,
            replay=self._replay_metadata(),
            created_utc=state.created_utc,
        )
        save_state(new_state, self._run_dir, payload=self._build_payload(new_state))
        return new_state

    def _train_epochs(self, experience: ContinualExperience) -> int:
        cfg = self.config
        assert self._scenario is not None
        dataset = ContinualImageDataset(
            experience.train_samples,
            self._scenario.images_root,
            cfg.image_size,
            require_split="train",
        )
        if len(dataset) == 0:
            raise ContinualTrainingError(
                f"Experience {experience.experience_id} has no training samples"
            )
        total_steps = 0
        epochs = cfg.epochs
        for epoch in range(epochs):
            loader = make_loader(
                dataset,
                batch_size=cfg.batch_size,
                seed=cfg.seed + 1009 * (experience.experience_id + 1) + epoch,
                workers=cfg.workers,
            )
            limit = len(loader)
            if cfg.max_steps_per_epoch is not None:
                limit = min(limit, cfg.max_steps_per_epoch)
            assert self._model is not None
            self._model.train()
            criterion = nn.CrossEntropyLoss()
            for step, (images, labels) in enumerate(loader):
                if cfg.max_steps_per_epoch is not None and step >= cfg.max_steps_per_epoch:
                    break
                replay = self._replay_tensors(experience.experience_id, epoch, step)
                if replay is not None:
                    replay_images, replay_labels = replay
                    images = torch.cat([images, replay_images], dim=0)
                    labels = torch.cat([labels, replay_labels], dim=0)
                images = images.to(self.device)
                labels = labels.to(self.device)
                self._optimizer.zero_grad()
                logits = self._model(images)
                loss = criterion(logits, labels)
                loss.backward()
                self._optimizer.step()
                total_steps += 1
                self._emit("batch", step + 1, limit)
            if self._scheduler is not None:
                self._scheduler.step()
            self._emit("epoch", epoch + 1, epochs)
        return total_steps

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def run(
        self,
        scenario: ContinualScenario,
        *,
        checkpoint_dir: str | Path | None = None,
    ) -> ContinualTrainingState:
        """Initialize once, then train every experience in official order."""
        state = self.initialize(scenario, checkpoint_dir=checkpoint_dir)
        for experience in scenario.iter_experiences():
            state = self.train_experience(state, experience)
        return state

    def _emit(self, event: str, current: int, total: int) -> None:
        if self.on_progress is not None:
            self.on_progress(event, current, total)
