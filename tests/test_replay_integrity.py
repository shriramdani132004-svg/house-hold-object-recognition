"""ReplayMemory integrity tests for both policies (Phase 4/5 contract).

Synthetic SampleRecord references only — no CORe50 data, no checkpoints, no
training runs. Each test pins one documented guarantee of
``src/training/replay.py``: boundedness, FIFO insertion order, reservoir
Algorithm R membership, deterministic sampling, state round-trips,
duplicate skipping, official-order guards, and provenance/policy
validation.
"""

from __future__ import annotations

import collections
import random
from typing import Any, Callable, Sequence

import pytest

from src.data.continual import SampleRecord
from src.training import ReplayMemory, ReplayMemoryError

POLICIES = ("fifo", "reservoir")


def _record(
    index: int,
    *,
    experience: int = 0,
    label: int = 0,
    split: str = "train",
) -> SampleRecord:
    """One synthetic training reference with a unique relative path."""
    return SampleRecord(
        relative_path=f"s1/o1/C_{experience:02d}_{index:03d}.png",
        label=label,
        split=split,
        experience_id=experience,
        source_filelist=f"synthetic/train_batch_{experience:02d}_filelist.txt",
        line_number=index + 1,
        object_id=experience + 1,
        session_id=1,
        category_id=experience,
        category_name="synthetic",
        object_name=f"synthetic_object_{experience + 1}",
    )


def _batch(count: int, *, experience: int = 0, label: int = 0) -> list[SampleRecord]:
    return [_record(i, experience=experience, label=label) for i in range(count)]


def _memory(
    capacity: int,
    policy: str,
    *,
    seed: int = 11,
    scenario: str | None = None,
    variant: str | None = None,
    run_id: int | None = None,
) -> ReplayMemory:
    return ReplayMemory(
        capacity,
        seed=seed,
        scenario=scenario,
        variant=variant,
        run_id=run_id,
        policy=policy,
    )


def _filled_payload() -> dict[str, Any]:
    """A valid stored state with three references at experience 0."""
    memory = _memory(4, "fifo", seed=11)
    memory.add(_batch(3, experience=0), experience_index=0)
    return memory.state_dict()


def _paths(records: Sequence[SampleRecord]) -> list[str]:
    return [record.relative_path for record in records]


def _reservoir_reference(
    batches: Sequence[Sequence[SampleRecord]], capacity: int, seed: int
) -> tuple[dict[str, SampleRecord], int]:
    """Inline reference implementation of Algorithm R.

    Mirrors the reservoir branch of ``ReplayMemory.add`` exactly, including
    the ``random.Random(seed * 1_000_003 + arrivals)`` formula, so membership
    can be asserted without trusting the production code.
    """
    items: dict[str, SampleRecord] = {}
    arrivals = 0
    for batch in batches:
        for record in batch:
            key = record.relative_path
            if key in items:
                continue
            arrivals += 1
            if len(items) < capacity:
                items[key] = record
            else:
                rng = random.Random(seed * 1_000_003 + arrivals)
                if rng.randrange(arrivals) < capacity:
                    victim = rng.choice(list(items))
                    del items[victim]
                    items[key] = record
    return items, arrivals


def _wrong_version(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload, format_version=99)


def _duplicate_item(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload, items=payload["items"] + [payload["items"][0]])


def _stored_test_split(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload, items=[dict(payload["items"][0], split="test")])


def _stored_future_record(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload, items=[dict(payload["items"][0], experience_id=2)], last_experience=1)


def _items_not_a_list(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload, items="not-a-list")


def _malformed_record(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload, items=[{"relative_path": "x"}])


def _not_a_mapping(payload: dict[str, Any]) -> Any:
    return [1, 2, 3]


@pytest.mark.parametrize("policy", POLICIES)
def test_empty_memory_has_size_zero_and_no_samples(policy: str) -> None:
    memory = _memory(5, policy)
    assert memory.size == 0
    assert len(memory) == 0
    assert not memory
    assert memory.records() == ()
    assert memory.sample(5) == []
    assert memory.last_experience == -1
    assert memory.state_dict()["arrivals"] == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_single_record_addition_grows_memory_to_one(policy: str) -> None:
    memory = _memory(5, policy)
    record = _record(0, experience=0)
    added = memory.add([record], experience_index=0)
    assert added == 1
    assert memory.size == 1
    assert memory.records() == (record,)
    assert len(memory.sample(5)) == 1
    assert memory.last_experience == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_capacity_never_exceeded_across_sequential_experiences(policy: str) -> None:
    capacity = 4
    memory = _memory(capacity, policy, seed=5)
    total_added = 0
    for experience in range(3):
        added = memory.add(
            _batch(capacity, experience=experience), experience_index=experience
        )
        assert added == capacity
        total_added += added
        assert memory.size <= capacity, "capacity must bound the memory after every add"
        assert memory.size == capacity
        assert memory.last_experience == experience
    assert total_added == 3 * capacity
    assert memory.state_dict()["arrivals"] == total_added


def test_fifo_retains_newest_references_in_insertion_order() -> None:
    memory = _memory(3, "fifo", seed=11)
    first = _batch(3, experience=0)
    second = _batch(2, experience=1)
    memory.add(first, experience_index=0)
    assert _paths(memory.records()) == _paths(first)
    memory.add(second, experience_index=1)
    assert _paths(memory.records()) == [
        first[2].relative_path,
        second[0].relative_path,
        second[1].relative_path,
    ], "FIFO must evict oldest-first, keeping exactly d, e and c in order"


def test_reservoir_identical_adds_produce_identical_state() -> None:
    batches = [_batch(3, experience=0), _batch(3, experience=1)]
    states: list[dict[str, Any]] = []
    for _ in range(2):
        memory = _memory(3, "reservoir", seed=7)
        for experience, batch in enumerate(batches):
            memory.add(batch, experience_index=experience)
        states.append(memory.state_dict())
    assert states[0] == states[1]


def test_reservoir_matches_reference_and_evicts_an_early_item() -> None:
    capacity, seed = 3, 7
    batches = [_batch(3, experience=0), _batch(3, experience=1)]
    memory = _memory(capacity, "reservoir", seed=seed)
    for experience, batch in enumerate(batches):
        memory.add(batch, experience_index=experience)

    expected, expected_arrivals = _reservoir_reference(batches, capacity, seed)
    kept = {record.relative_path: record for record in memory.records()}
    assert kept == expected
    assert memory.size == capacity
    assert memory.state_dict()["arrivals"] == expected_arrivals == 6

    newest = {record.relative_path for record in batches[1]}
    assert set(kept) != newest, "reservoir must not simply keep the newest references"
    assert batches[0][0].relative_path in kept
    assert batches[0][1].relative_path in kept
    assert batches[0][2].relative_path not in kept, (
        "Algorithm R must evict an early arrival for this seed"
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_sampling_is_deterministic_and_seed_sensitive(policy: str) -> None:
    memory = _memory(12, policy, seed=5)
    memory.add(_batch(12, experience=0), experience_index=0)

    first = memory.sample(6, seed=3)
    second = memory.sample(6, seed=3)
    assert first == second, "same seed must reproduce the same sample list"
    assert first != memory.sample(6, seed=4), "different seeds must reorder the sample"
    assert memory.sample(6) == memory.sample(6, seed=memory.seed)
    assert len(memory.sample(100)) == memory.size
    assert memory.sample(0, seed=1) == []
    with pytest.raises(ReplayMemoryError, match="non-negative integer"):
        memory.sample(-1)


@pytest.mark.parametrize("policy", POLICIES)
def test_state_dict_roundtrip_restores_exact_memory(policy: str) -> None:
    provenance = {"scenario": "SYNTH", "variant": "inc", "run_id": 0}
    source = _memory(4, policy, seed=11, **provenance)
    source.add(_batch(3, experience=0), experience_index=0)
    source.add(_batch(3, experience=1), experience_index=1)

    payload = source.state_dict()
    restored = _memory(4, policy, seed=999, **provenance)
    restored.load_state_dict(payload)
    assert restored.records() == source.records()
    assert restored.policy == source.policy
    assert restored.last_experience == source.last_experience
    assert restored.seed == source.seed
    assert restored.state_dict()["arrivals"] == payload["arrivals"]
    assert restored.state_dict() == payload, "first round-trip must be lossless"

    again = _memory(4, policy, seed=123, **provenance)
    again.load_state_dict(restored.state_dict())
    assert again.state_dict() == payload, "second round-trip must stay stable"
    assert restored.sample(3, seed=9) == source.sample(3, seed=9)


@pytest.mark.parametrize("capacity", [0, -1, 2.5, "4"])
def test_constructor_rejects_invalid_capacity(capacity: Any) -> None:
    with pytest.raises(ReplayMemoryError, match="capacity must be an integer >= 1"):
        ReplayMemory(capacity)


def test_constructor_rejects_unknown_policy() -> None:
    with pytest.raises(ReplayMemoryError, match="policy must be"):
        ReplayMemory(4, policy="lru")


@pytest.mark.parametrize("experience_index", [-1, 1.5, "0"])
def test_add_rejects_invalid_experience_index(experience_index: Any) -> None:
    memory = _memory(4, "fifo")
    with pytest.raises(
        ReplayMemoryError, match="experience_index must be a non-negative integer"
    ):
        memory.add(_batch(2), experience_index=experience_index)
    assert memory.size == 0


def test_add_rejects_out_of_order_experiences() -> None:
    memory = _memory(4, "fifo")
    with pytest.raises(
        ReplayMemoryError, match="expected experience 0, got 1"
    ):
        memory.add(_batch(2, experience=1), experience_index=1)
    memory.add(_batch(2, experience=0), experience_index=0)
    with pytest.raises(
        ReplayMemoryError, match="expected experience 1, got 3"
    ):
        memory.add(_batch(2, experience=3), experience_index=3)
    assert memory.last_experience == 0
    assert memory.size == 2


@pytest.mark.parametrize("split", ["test", "val"])
def test_add_rejects_evaluation_splits(split: str) -> None:
    memory = _memory(4, "fifo")
    with pytest.raises(ReplayMemoryError, match="cannot enter replay memory"):
        memory.add([_record(0, split=split)], experience_index=0)
    assert memory.size == 0
    assert memory.last_experience == -1


def test_add_rejects_future_experience_records() -> None:
    memory = _memory(4, "fifo")
    with pytest.raises(ReplayMemoryError, match="Future-experience"):
        memory.add([_record(9, experience=1)], experience_index=0)
    assert memory.size == 0
    assert memory.last_experience == -1


@pytest.mark.parametrize("value", [42, None, "record", {"relative_path": "x"}])
def test_add_rejects_non_sample_records(value: Any) -> None:
    memory = _memory(4, "fifo")
    with pytest.raises(ReplayMemoryError, match="accepts SampleRecord references"):
        memory.add([value], experience_index=0)
    assert memory.size == 0


@pytest.mark.parametrize(
    ("case_id", "mutate", "target_kwargs", "match"),
    [
        ("format_version", _wrong_version, {}, "Unsupported replay memory format"),
        ("oversized_items", None, {"capacity": 2}, "exceeds the configured capacity"),
        ("duplicate_paths", _duplicate_item, {}, "Duplicate replay reference"),
        ("stored_test_split", _stored_test_split, {}, "is not allowed"),
        ("stored_future_record", _stored_future_record, {}, "Stored future-experience"),
        ("policy_mismatch", None, {"policy": "reservoir"}, "policy mismatch"),
        ("provenance_mismatch", None, {"scenario": "OTHER"}, "provenance mismatch"),
        ("not_a_mapping", _not_a_mapping, {}, "must be a mapping"),
        ("items_not_a_list", _items_not_a_list, {}, "items must be a list"),
        ("malformed_record", _malformed_record, {}, "Malformed replay memory record"),
    ],
)
def test_load_state_dict_rejects_invalid_payloads(
    case_id: str,
    mutate: Callable[[dict[str, Any]], Any] | None,
    target_kwargs: dict[str, Any],
    match: str,
) -> None:
    payload = _filled_payload()
    if mutate is not None:
        payload = mutate(payload)
    target = _memory(
        target_kwargs.get("capacity", 4),
        target_kwargs.get("policy", "fifo"),
        scenario=target_kwargs.get("scenario"),
        variant=target_kwargs.get("variant"),
        run_id=target_kwargs.get("run_id"),
    )
    with pytest.raises(ReplayMemoryError, match=match):
        target.load_state_dict(payload)
    assert target.size == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_duplicate_relative_paths_never_grow_the_memory(policy: str) -> None:
    memory = _memory(4, policy, seed=3)
    record = _record(0)

    added = memory.add([record, record], experience_index=0)
    assert added == 1, "the same call must store the duplicate only once"
    assert memory.size == 1
    assert memory.state_dict()["arrivals"] == 1

    repeated = SampleRecord(**{**record.to_dict(), "experience_id": 1})
    added = memory.add([repeated], experience_index=1)
    assert added == 0, "a later experience repeating a stored path is skipped"
    assert memory.size == 1
    assert memory.last_experience == 1
    assert memory.state_dict()["arrivals"] == 1, (
        "arrivals counts only unique stored arrivals; skipped duplicates add none"
    )
    assert memory.records()[0] == record, "the first-seen record must win"


def test_experience_transitions_and_clear_reset() -> None:
    memory = _memory(4, "fifo", seed=2)
    for experience in range(3):
        memory.add(_batch(2, experience=experience), experience_index=experience)
        assert memory.last_experience == experience
    assert memory.size == 4

    memory.clear()
    assert memory.size == 0
    assert len(memory) == 0
    assert not memory
    assert memory.last_experience == -1
    assert memory.state_dict()["arrivals"] == 0
    assert memory.sample(3) == []

    assert memory.add(_batch(2, experience=0), experience_index=0) == 2
    assert memory.last_experience == 0


def test_label_histogram_is_computable_from_records() -> None:
    memory = _memory(32, "fifo", seed=1)
    records = [_record(i, experience=0, label=i % 10) for i in range(20)]
    memory.add(records, experience_index=0)

    histogram = collections.Counter(record.label for record in memory.records())
    assert dict(histogram) == {label: 2 for label in range(10)}
    assert sum(histogram.values()) == memory.size == 20
    assert {record.label for record in memory.sample(20, seed=6)} == set(range(10))


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_matches_constructor_and_state_dict(policy: str) -> None:
    memory = _memory(4, policy)
    assert memory.policy == policy
    payload = memory.state_dict()
    assert payload["policy"] == policy
    assert payload["format_version"] == 1

    counterpart = "reservoir" if policy == "fifo" else "fifo"
    other = _memory(4, counterpart)
    with pytest.raises(ReplayMemoryError, match="policy mismatch"):
        other.load_state_dict(payload)
