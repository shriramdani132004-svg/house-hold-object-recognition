"""Method B — Experience Replay: bounded memory and replay trainer (Phase 4).

Exactly ONE anti-forgetting strategy is implemented: uniform Experience
Replay over a bounded, deterministic memory of lightweight sample
*references* (official ``SampleRecord`` paths — never copied image data,
never evaluation/test samples, never future-experience samples).
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import torch

from src.data.continual import SampleRecord
from src.training.base import BaseContinualTrainer
from src.training.dataset import ContinualImageDataset

MEMORY_VERSION = 1


class ReplayMemoryError(Exception):
    """Raised when the replay memory contract is violated."""


class ReplayMemory:
    """Bounded FIFO memory of training-sample references.

    Guarantees:

    - capacity is a fixed maximum;
    - two memory policies, both standard Experience Replay buffers:
      ``"fifo"`` evicts the oldest reference first (the Phase-5 default),
      ``"reservoir"`` performs uniform reservoir sampling over all
      training references seen so far (Algorithm R), which keeps every
      class represented regardless of experience recency;
    - only ``split == "train"`` records are accepted — evaluation/test
      samples can never enter;
    - only records from already-trained experiences are accepted
      (``record.experience_id <= experience_index``), so future experiences
      can never enter;
    - additions follow official sequential order (experience 0, 1, 2, …);
    - sampling with an explicit seed is deterministic;
    - the memory round-trips through ``state_dict`` / ``load_state_dict``.
    """

    def __init__(
        self,
        capacity: int,
        *,
        seed: int = 0,
        scenario: str | None = None,
        variant: str | None = None,
        run_id: int | None = None,
        policy: str = "fifo",
    ) -> None:
        if not isinstance(capacity, int) or capacity < 1:
            raise ReplayMemoryError(f"capacity must be an integer >= 1, got {capacity!r}")
        if policy not in ("fifo", "reservoir"):
            raise ReplayMemoryError(
                f"policy must be 'fifo' or 'reservoir', got {policy!r}"
            )
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.scenario = scenario
        self.variant = variant
        self.run_id = run_id
        self.policy = policy
        self.last_experience = -1
        self._arrivals = 0
        self._items: dict[str, SampleRecord] = {}

    # ------------------------------------------------------------------
    # basic access
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def records(self) -> tuple[SampleRecord, ...]:
        """Current contents in insertion (FIFO eviction) order."""
        return tuple(self._items.values())

    def clear(self) -> None:
        """Explicit reset: empty contents, ready for experience 0 again."""
        self._items.clear()
        self.last_experience = -1
        self._arrivals = 0

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def add(
        self, samples: Sequence[SampleRecord], *, experience_index: int
    ) -> int:
        """Add one experience's official training samples.

        Returns the number of *new* references stored (duplicates are
        skipped). Raises :class:`ReplayMemoryError` for evaluation samples,
        future-experience samples, or out-of-order additions.
        """
        if not isinstance(experience_index, int) or experience_index < 0:
            raise ReplayMemoryError(
                f"experience_index must be a non-negative integer, got {experience_index!r}"
            )
        expected = self.last_experience + 1
        if experience_index != expected:
            raise ReplayMemoryError(
                f"Replay memory additions must follow official order: expected "
                f"experience {expected}, got {experience_index} "
                f"(future or out-of-order experiences cannot enter replay memory)"
            )

        prepared: list[SampleRecord] = []
        for record in samples:
            if not isinstance(record, SampleRecord):
                raise ReplayMemoryError(
                    f"Replay memory accepts SampleRecord references, got {type(record).__name__}"
                )
            if record.split != "train":
                raise ReplayMemoryError(
                    f"Evaluation/test sample {record.relative_path!r} "
                    f"(split={record.split!r}) cannot enter replay memory"
                )
            if record.experience_id > experience_index:
                raise ReplayMemoryError(
                    f"Future-experience sample {record.relative_path!r} "
                    f"(experience {record.experience_id} > {experience_index}) "
                    f"cannot enter replay memory"
                )
            prepared.append(record)

        added = 0
        for record in prepared:
            key = record.relative_path
            if key in self._items:
                continue
            if self.policy == "reservoir":
                self._arrivals += 1
                if len(self._items) < self.capacity:
                    self._items[key] = record
                else:
                    rng = random.Random(self.seed * 1_000_003 + self._arrivals)
                    if rng.randrange(self._arrivals) < self.capacity:
                        victim = rng.choice(list(self._items))
                        del self._items[victim]
                        self._items[key] = record
            else:
                self._arrivals += 1
                self._items[key] = record
            added += 1
        if self.policy == "fifo":
            while len(self._items) > self.capacity:
                self._items.pop(next(iter(self._items)))
        self.last_experience = experience_index
        return added

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def sample(
        self, k: int, *, seed: int | None = None
    ) -> list[SampleRecord]:
        """Uniformly sample ``k`` references without replacement.

        With an explicit ``seed`` the result is fully deterministic; with
        ``seed=None`` the memory's own seed is used.
        """
        if not isinstance(k, int) or k < 0:
            raise ReplayMemoryError(f"k must be a non-negative integer, got {k!r}")
        if not self._items:
            return []
        count = min(k, len(self._items))
        rng = random.Random(self.seed if seed is None else seed)
        return rng.sample(list(self._items.values()), count)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot (paths are project-relative references)."""
        return {
            "format_version": MEMORY_VERSION,
            "capacity": self.capacity,
            "seed": self.seed,
            "scenario": self.scenario,
            "variant": self.variant,
            "run_id": self.run_id,
            "policy": self.policy,
            "arrivals": self._arrivals,
            "last_experience": self.last_experience,
            "items": [record.to_dict() for record in self._items.values()],
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore contents; validates version, capacity and provenance."""
        if not isinstance(payload, dict):
            raise ReplayMemoryError(
                f"Replay memory state must be a mapping, got {type(payload).__name__}"
            )
        version = payload.get("format_version")
        if version != MEMORY_VERSION:
            raise ReplayMemoryError(
                f"Unsupported replay memory format {version!r} (expected {MEMORY_VERSION})"
            )
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raise ReplayMemoryError("Replay memory items must be a list")
        if len(raw_items) > self.capacity:
            raise ReplayMemoryError(
                f"Stored replay memory ({len(raw_items)} items) exceeds the "
                f"configured capacity {self.capacity}"
            )
        last_experience = int(payload.get("last_experience", -1))

        restored: dict[str, SampleRecord] = {}
        for raw in raw_items:
            try:
                record = SampleRecord(**raw)
            except TypeError as exc:
                raise ReplayMemoryError(f"Malformed replay memory record: {exc}") from exc
            if record.split != "train":
                raise ReplayMemoryError(
                    f"Stored evaluation/test sample {record.relative_path!r} is not allowed"
                )
            if record.experience_id > last_experience:
                raise ReplayMemoryError(
                    f"Stored future-experience sample {record.relative_path!r} "
                    f"(experience {record.experience_id} > {last_experience})"
                )
            if record.relative_path in restored:
                raise ReplayMemoryError(
                    f"Duplicate replay reference {record.relative_path!r} in stored state"
                )
            restored[record.relative_path] = record

        scenario = payload.get("scenario", self.scenario)
        variant = payload.get("variant", self.variant)
        run_id = payload.get("run_id", self.run_id)
        if (scenario, variant, run_id) != (self.scenario, self.variant, self.run_id):
            raise ReplayMemoryError(
                f"Replay memory provenance mismatch: stored "
                f"{(scenario, variant, run_id)!r}, expected "
                f"{(self.scenario, self.variant, self.run_id)!r}"
            )
        stored_policy = payload.get("policy", "fifo")
        if stored_policy != self.policy:
            raise ReplayMemoryError(
                f"Replay memory policy mismatch: stored {stored_policy!r}, "
                f"expected {self.policy!r}"
            )
        self._arrivals = int(payload.get("arrivals", len(restored)))

        self._items = restored
        self.last_experience = last_experience
        self.seed = int(payload.get("seed", self.seed))


class ReplayContinualTrainer(BaseContinualTrainer):
    """Experience Replay trainer (the single anti-forgetting method).

    Per experience: load current official training samples, draw replay
    samples from *earlier* experiences only (memory is empty at experience
    0), concatenate current + replay into each training batch, continue the
    SAME persistent model, then update the memory and save model/optimizer/
    scheduler/memory state.
    """

    method_name = "replay"

    def __init__(self, config=None, *, on_progress=None, cache=None) -> None:
        super().__init__(config, on_progress=on_progress)
        self._memory: ReplayMemory | None = None
        self._replay_steps = 0
        self._replay_samples_used = 0
        # Decoded tensor cache shared with the main training dataset so
        # replay samples are read from RAM instead of re-decoded per step.
        self._cache = cache

    # ------------------------------------------------------------------
    # persistent extras
    # ------------------------------------------------------------------
    def _extra_payload(self) -> dict[str, Any]:
        return {
            "replay_memory": self._memory.state_dict() if self._memory else None,
            "replay_stats": {
                "steps_with_replay": self._replay_steps,
                "samples_replayed": self._replay_samples_used,
            },
        }

    def _restore_extra(self, payload: dict[str, Any]) -> None:
        scenario = self._scenario
        memory = ReplayMemory(
            self.config.replay_capacity,
            seed=self.config.resolved_replay_seed,
            scenario=scenario.scenario_type if scenario else None,
            variant=scenario.variant if scenario else None,
            run_id=scenario.run_id if scenario else None,
            policy=self.config.replay_policy,
        )
        stored = payload.get("replay_memory")
        if stored is not None:
            memory.load_state_dict(stored)
        self._memory = memory
        stats = payload.get("replay_stats") or {}
        self._replay_steps = int(stats.get("steps_with_replay", 0))
        self._replay_samples_used = int(stats.get("samples_replayed", 0))

    def _on_experience_trained(self, experience) -> None:
        """Update replay memory AFTER training on the current experience."""
        assert self._memory is not None
        self._memory.add(
            experience.train_samples, experience_index=experience.experience_id
        )

    def _replay_metadata(self) -> dict[str, Any] | None:
        if self._memory is None:
            return None
        return {
            "enabled": True,
            "capacity": self._memory.capacity,
            "size": self._memory.size,
            "last_experience": self._memory.last_experience,
            "seed": self._memory.seed,
            "steps_with_replay": self._replay_steps,
            "samples_replayed": self._replay_samples_used,
        }

    # ------------------------------------------------------------------
    # replay batching
    # ------------------------------------------------------------------
    def _replay_tensors(
        self, experience_id: int, epoch: int, step: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample past-experience images for one training step (None if empty)."""
        if self._memory is None or not self._memory:
            return None
        k = self.config.replay_batch_size
        if k <= 0:
            return None
        records = self._memory.sample(
            k,
            seed=self.config.resolved_replay_seed
            + 7919 * (experience_id + 1)
            + 101 * epoch
            + step,
        )
        if not records:
            return None
        assert self._scenario is not None
        dataset = ContinualImageDataset(
            records,
            self._scenario.images_root,
            self.config.image_size,
            require_split="train",
            cache=self._cache,
        )
        images = torch.stack([dataset[i][0] for i in range(len(dataset))])
        labels = torch.tensor(
            [int(record.label) for record in records], dtype=torch.long
        )
        self._replay_steps += 1
        self._replay_samples_used += len(records)
        return images, labels
