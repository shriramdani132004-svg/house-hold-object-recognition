"""Persistent continual training state (Step 2 of Phase 4).

The state is split into two files inside one run directory:

- ``state.json`` — portable metadata (JSON). It never contains
  machine-specific absolute paths; ``checkpoint`` / ``state_file`` are plain
  file names relative to the state directory itself.
- ``checkpoint.pt`` — torch payload with model, optimizer, scheduler and
  optional replay-memory tensors for resuming exactly where training stopped.

Both files live under the configured checkpoint root (default
``models/continual/``), outside every dataset tree and outside Git.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_VERSION = 1
PAYLOAD_VERSION = 1
CHECKPOINT_NAME = "checkpoint.pt"
STATE_FILE_NAME = "state.json"

_REQUIRED_KEYS = (
    "version",
    "scenario",
    "variant",
    "run_id",
    "method",
    "current_experience",
    "experiences_trained",
    "epochs_completed",
    "steps_completed",
    "seed",
    "config_fingerprint",
)


class ContinualStateError(Exception):
    """Base error for continual training state handling."""


class CheckpointMissingError(ContinualStateError):
    """The checkpoint file referenced by a state does not exist."""


class StateCompatibilityError(ContinualStateError):
    """A state or checkpoint does not match what the caller expects."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Drop values that would embed machine-specific paths into metadata."""
    from src.training.config import looks_absolute

    cleaned: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, str) and looks_absolute(value):
            continue
        cleaned[key] = value
    return cleaned


@dataclass(frozen=True)
class ContinualTrainingState:
    """Serializable snapshot of one continual training run."""

    scenario: str
    variant: str
    run_id: int
    method: str
    current_experience: int = -1
    experiences_trained: tuple[int, ...] = ()
    epochs_completed: int = 0
    steps_completed: int = 0
    seed: int = 0
    config_fingerprint: str = ""
    training_config: dict[str, Any] = field(default_factory=dict)
    replay: dict[str, Any] | None = None
    checkpoint: str = CHECKPOINT_NAME
    state_file: str = STATE_FILE_NAME
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)
    version: int = STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (tuples become lists)."""
        return {
            "version": self.version,
            "scenario": self.scenario,
            "variant": self.variant,
            "run_id": int(self.run_id),
            "method": self.method,
            "current_experience": int(self.current_experience),
            "experiences_trained": [int(i) for i in self.experiences_trained],
            "epochs_completed": int(self.epochs_completed),
            "steps_completed": int(self.steps_completed),
            "seed": int(self.seed),
            "config_fingerprint": self.config_fingerprint,
            "training_config": sanitize_config(dict(self.training_config)),
            "replay": self.replay,
            "checkpoint": self.checkpoint,
            "state_file": self.state_file,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContinualTrainingState":
        if not isinstance(payload, dict):
            raise StateCompatibilityError(
                f"State must be a mapping, got {type(payload).__name__}"
            )
        missing = [key for key in _REQUIRED_KEYS if key not in payload]
        if missing:
            raise StateCompatibilityError(
                f"State is missing required key(s): {', '.join(missing)}"
            )
        version = payload["version"]
        if version != STATE_VERSION:
            raise StateCompatibilityError(
                f"Unsupported state version {version!r} (expected {STATE_VERSION})"
            )
        checkpoint = payload.get("checkpoint", CHECKPOINT_NAME)
        state_file = payload.get("state_file", STATE_FILE_NAME)
        for label, name in (("checkpoint", checkpoint), ("state_file", state_file)):
            if not isinstance(name, str) or not name or Path(name).is_absolute() or len(Path(name).parts) != 1:
                raise StateCompatibilityError(
                    f"State {label} must be a plain file name, got {name!r}"
                )
        experiences = payload.get("experiences_trained", [])
        if not isinstance(experiences, list) or not all(isinstance(i, int) for i in experiences):
            raise StateCompatibilityError("experiences_trained must be a list of integers")
        return cls(
            scenario=str(payload["scenario"]),
            variant=str(payload["variant"]),
            run_id=int(payload["run_id"]),
            method=str(payload["method"]),
            current_experience=int(payload["current_experience"]),
            experiences_trained=tuple(experiences),
            epochs_completed=int(payload["epochs_completed"]),
            steps_completed=int(payload["steps_completed"]),
            seed=int(payload["seed"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            training_config=dict(payload.get("training_config") or {}),
            replay=payload.get("replay"),
            checkpoint=checkpoint,
            state_file=state_file,
            created_utc=str(payload.get("created_utc") or _utc_now()),
            updated_utc=str(payload.get("updated_utc") or _utc_now()),
            version=int(version),
        )


def save_state(
    state: ContinualTrainingState,
    directory: str | Path,
    *,
    payload: dict[str, Any],
) -> Path:
    """Persist state metadata plus the torch checkpoint payload.

    ``payload`` must contain ``format_version`` and ``model_state`` keys; it
    is written atomically enough for single-process training (write to a
    temporary name, then replace).
    """
    import torch

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    if not isinstance(payload, dict) or "format_version" not in payload or "model_state" not in payload:
        raise StateCompatibilityError(
            "Checkpoint payload must be a mapping containing 'format_version' and 'model_state'"
        )

    checkpoint_path = target / CHECKPOINT_NAME
    tmp_path = target / (CHECKPOINT_NAME + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(checkpoint_path)

    updated = replace(
        state,
        checkpoint=CHECKPOINT_NAME,
        state_file=STATE_FILE_NAME,
        updated_utc=_utc_now(),
    )
    state_path = target / STATE_FILE_NAME
    tmp_state = target / (STATE_FILE_NAME + ".tmp")
    tmp_state.write_text(
        json.dumps(updated.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    tmp_state.replace(state_path)
    return state_path


def load_state(
    directory: str | Path,
) -> tuple[ContinualTrainingState, dict[str, Any]]:
    """Load state metadata and its checkpoint payload from a run directory."""
    import torch

    source = Path(directory)
    state_path = source / STATE_FILE_NAME
    if not state_path.is_file():
        raise ContinualStateError(f"State file not found: {STATE_FILE_NAME} in {source}")
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StateCompatibilityError(f"State file is not valid JSON: {state_path}") from exc
    state = ContinualTrainingState.from_dict(raw)

    checkpoint_path = source / state.checkpoint
    if not checkpoint_path.is_file():
        raise CheckpointMissingError(
            f"Checkpoint referenced by state is missing: {state.checkpoint} in {source}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - surface any payload problem as a state error
        raise StateCompatibilityError(
            f"Could not load checkpoint payload from {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise StateCompatibilityError("Checkpoint payload is malformed")
    if payload["format_version"] != PAYLOAD_VERSION:
        raise StateCompatibilityError(
            f"Unsupported checkpoint payload format {payload['format_version']!r} "
            f"(expected {PAYLOAD_VERSION})"
        )
    return state, payload


def validate_state(
    state: ContinualTrainingState,
    *,
    scenario: str | None = None,
    variant: str | None = None,
    run_id: int | None = None,
    method: str | None = None,
    config_fingerprint: str | None = None,
) -> None:
    """Raise :class:`StateCompatibilityError` on any mismatch."""
    checks = (
        ("scenario", scenario, state.scenario),
        ("variant", variant, state.variant),
        ("run_id", run_id, state.run_id),
        ("method", method, state.method),
        ("config_fingerprint", config_fingerprint, state.config_fingerprint),
    )
    mismatches = [
        f"{label}: expected {expected!r}, state has {actual!r}"
        for label, expected, actual in checks
        if expected is not None and expected != actual
    ]
    if mismatches:
        raise StateCompatibilityError(
            "State compatibility check failed: " + "; ".join(mismatches)
        )


def checkpoint_exists(directory: str | Path) -> bool:
    """True when a loadable state + checkpoint pair exists in ``directory``."""
    source = Path(directory)
    if not (source / STATE_FILE_NAME).is_file():
        return False
    try:
        state, _ = load_state(source)
    except ContinualStateError:
        return False
    return (source / state.checkpoint).is_file()
