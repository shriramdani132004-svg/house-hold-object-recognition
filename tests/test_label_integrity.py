"""Label-system regression tests (Phases 7-11, 27-30).

Everything here guards THE invariant: every CORe50 reference maps to
exactly one correct identity, label == object_id - 1 for NI/NIC, and
training/evaluation/inference/app all decode predictions through the
same authoritative mapping. Mutation tests prove the guards actually
fire. Dataset-backed tests skip when data/raw is not installed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from src.data import core50
from src.data.class_mapping import (
    MAPPING_VERSION,
    OFFICIAL_LABELS_PATH,
    DEFAULT_MAPPING_PATH,
    ClassMappingError,
    assert_matches_official_names,
    load_class_mapping,
    load_class_names,
    mapping_checksum,
    mapping_from_payload,
)
from src.data.continual.label_audit import audit_references, representative_samples
from src.data.continual.scenarios import validate_reference
from src.training import build_model

MAPPING_SHA_PIN = "5d28556368a38f161160773a0210bdad6a9d73ba839d6e110066421104facad2"
IDENTITY_TABLE = (
    "plug_adapter1", "plug_adapter2", "plug_adapter3", "plug_adapter4", "plug_adapter5",
    "mobile_phone1", "mobile_phone2", "mobile_phone3", "mobile_phone4", "mobile_phone5",
    "scissor1", "scissor2", "scissor3", "scissor4", "scissor5",
    "light_bulb1", "light_bulb2", "light_bulb3", "light_bulb4", "light_bulb5",
    "can1", "can2", "can3", "can4", "can5",
    "glass1", "glass2", "glass3", "glass4", "glass5",
    "ball1", "ball2", "ball3", "ball4", "ball5",
    "marker1", "marker2", "marker3", "marker4", "marker5",
    "cup1", "cup2", "cup3", "cup4", "cup5",
    "remote_control1", "remote_control2", "remote_control3", "remote_control4", "remote_control5",
)

DATA_AVAILABLE = DEFAULT_MAPPING_PATH.is_file() and (
    core50.FILELIST_DIR / "NIC_inc" / "run0"
).is_dir()
requires_data = pytest.mark.skipif(
    not DATA_AVAILABLE, reason="official CORe50 filelists/mapping not installed"
)


# ---------------------------------------------------------------------------
# Phase 6/7: mapping identity, version, checksum
# ---------------------------------------------------------------------------
def test_identity_table_is_stable_and_official():
    mapping = load_class_mapping()
    assert tuple(mapping.names) == IDENTITY_TABLE
    assert mapping_checksum() == MAPPING_SHA_PIN
    assert MAPPING_VERSION == "core50-object-mapping-v1"
    official = [
        line.strip()
        for line in OFFICIAL_LABELS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert official == list(IDENTITY_TABLE)


def test_mapping_file_bytes_are_pinned():
    digest = hashlib.sha256(DEFAULT_MAPPING_PATH.read_bytes()).hexdigest()
    assert digest == MAPPING_SHA_PIN


def test_name_mutation_is_caught():
    import json

    payload = json.loads(DEFAULT_MAPPING_PATH.read_text(encoding="utf-8"))
    payload["objects"][0]["name"] = "not_the_real_identity"
    mutated = mapping_from_payload(payload, source="mutated")
    with pytest.raises(ClassMappingError, match="official"):
        assert_matches_official_names(mutated)


def test_category_order_permutation_is_caught():
    import json

    payload = json.loads(DEFAULT_MAPPING_PATH.read_text(encoding="utf-8"))
    payload["category_order"][0], payload["category_order"][1] = (
        payload["category_order"][1],
        payload["category_order"][0],
    )
    with pytest.raises(ClassMappingError, match="category_order"):
        mapping_from_payload(payload, source="permuted")


# ---------------------------------------------------------------------------
# Phase 7/8: label immutability + training target validation
# ---------------------------------------------------------------------------
def test_official_label_rule_rejects_off_by_one():
    mapping = {1: {"name": "x", "category_id": 0, "category_name": "c"}}
    validate_reference(
        "s1/o1/a.png", 1, 1, 0, mapping, source=Path("f.txt"), line_number=1,
        require_official_label_rule=True,
    )
    with pytest.raises(Exception, match="official identity mapping is broken"):
        validate_reference(
            "s1/o1/a.png", 1, 1, 1, mapping, source=Path("f.txt"), line_number=1,
            require_official_label_rule=True,
        )


@requires_data
def test_every_official_reference_obeys_the_rule():
    result = audit_references(check_files_exist=False)
    assert result.passed, [f"{f.where}:{f.line_number} {f.problem}" for f in result.failures[:5]]
    assert result.train_references == 119_894
    assert result.eval_references == 44_972
    assert len(result.identity_counts) == 50
    assert result.mapping_checksum == MAPPING_SHA_PIN


@requires_data
def test_training_target_matches_authoritative_label():
    from src.data.continual import load_scenario
    from src.training.dataset import ContinualImageDataset

    scenario = load_scenario("NIC", variant="inc", run=0)
    mapping = load_class_mapping()
    experience = scenario.get_experience(0)
    record = experience.train_samples[0]
    assert record.label == record.object_id - 1
    assert mapping.name_for(record.label) == record.object_name
    dataset = ContinualImageDataset(
        experience.train_samples[:1], scenario.images_root, image_size=16,
        require_split="train",
    )
    _image, label = dataset[0]
    assert int(label) == record.label
    torch.nn.functional.cross_entropy(
        torch.zeros(1, 50, requires_grad=True), torch.tensor([int(label)])
    ).backward()


@requires_data
def test_representative_label_proof():
    samples = representative_samples(per_identity=1, check_files_exist=True)
    assert len(samples) == 50
    assert {int(s["label"]) for s in samples} == set(range(50))
    assert all(s["matches"] for s in samples)
    assert all(s["file_exists"] for s in samples)


# ---------------------------------------------------------------------------
# Phases 9/10/11: evaluation, inference, app decode through ONE mapping
# ---------------------------------------------------------------------------
def test_index_to_identity_decoding_is_exact():
    names = load_class_names()
    assert names == {str(i): name for i, name in enumerate(IDENTITY_TABLE)}
    for arch in ("small_cnn", "compact_resnet"):
        model = build_model(50, arch=arch)
        logits = model(torch.zeros(2, 3, 16, 16))
        assert logits.shape == (2, 50)
        predicted = int(logits[0].argmax())
        assert predicted in range(50)
        assert names[str(predicted)] == IDENTITY_TABLE[predicted]


# ---------------------------------------------------------------------------
# Phase 28: gradient sanity (parameters change, no NaN/Inf)
# ---------------------------------------------------------------------------
def test_one_batch_changes_parameters_without_nan():
    torch.manual_seed(0)
    model = build_model(50)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    images = torch.randn(4, 3, 16, 16)
    targets = torch.tensor([0, 20, 49, 7])
    optimizer.zero_grad()
    loss = criterion(model(images), targets)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    optimizer.step()
    changed = sum(
        not torch.equal(a, b.detach())
        for a, b in zip(before, model.parameters(), strict=True)
    )
    assert changed == len(before)
    assert all(torch.isfinite(p).all() for p in model.parameters())


# ---------------------------------------------------------------------------
# Phase 19: loss receives RAW LOGITS (no softmax in between)
# ---------------------------------------------------------------------------
def test_cross_entropy_consumes_raw_logits():
    from src.training import base as training_base
    import inspect

    source = inspect.getsource(training_base)
    assert "criterion(logits, labels)" in source
    assert "softmax" not in source.lower().replace("log_softmax", "")
