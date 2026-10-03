"""Focused Phase-6 tests: error analysis, final model selection, freeze.

Uses the committed Phase-5 artifacts as the source of truth plus tiny
synthetic fixtures for the targeted-inference helpers. No training, no
full-dataset scans — the whole file finishes in seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
import torch

from src.data.continual import SampleRecord
from src.evaluation.continual import load_class_names
from src.evaluation.error_analysis import (
    ENV_FALLBACK_SENTENCE,
    PRIMARY_CRITERIA,
    build_class_analysis,
    build_environment_analysis,
    build_experiment_placeholder,
    build_experience_analysis,
    build_final_model_metadata,
    build_forgetting_analysis,
    build_representative_errors,
    final_class_stats,
    load_phase5_results,
    render_error_analysis_md,
    render_selection_md,
    render_summary_md,
    select_candidate_records,
    select_final_model,
    select_target_classes,
    sha256_file,
)
from src.training import build_model
from src.utils.progress import PhaseProgress

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE5_REPORTS = PROJECT_ROOT / "reports" / "phase5_nic"
ANALYSIS_DIR = PROJECT_ROOT / "reports" / "phase6_analysis"
PHASE5_MODELS = PROJECT_ROOT / "models" / "continual" / "phase5_nic"
FINAL_CKPT = PROJECT_ROOT / "models" / "continual" / "final_model.pt"
FINAL_META = PROJECT_ROOT / "models" / "continual" / "final_model.json"
OBJECT_MAPPING = PROJECT_ROOT / "data" / "raw" / "core50" / "metadata" / "object_mapping.json"
FILELIST_DIR = PROJECT_ROOT / "data" / "raw" / "core50" / "filelists"
ABS_MARKERS = ("C:\\", "C:/", "/home/")

PHASE5_PRESENT = all(
    (PHASE5_REPORTS / name).is_file()
    for name in (
        "naive_metrics.json",
        "replay_metrics.json",
        "experiment_summary.json",
        "per_class_metrics.json",
    )
)
CHECKPOINTS_PRESENT = all(
    (PHASE5_MODELS / method / "checkpoint.pt").is_file()
    for method in ("naive", "replay")
)
FINAL_PRESENT = FINAL_CKPT.is_file() and FINAL_META.is_file()
ANALYSIS_PRESENT = ANALYSIS_DIR.is_dir()

needs_phase5 = pytest.mark.skipif(
    not PHASE5_PRESENT, reason="committed Phase-5 reports not present"
)
needs_final = pytest.mark.skipif(not FINAL_PRESENT, reason="final model not frozen yet")
needs_analysis = pytest.mark.skipif(
    not ANALYSIS_PRESENT, reason="reports/phase6_analysis not generated yet"
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_record(
    label: int, session: int, path_tag: str, *, split: str = "test"
) -> SampleRecord:
    return SampleRecord(
        relative_path=f"s{session}/o1/C_{session:02d}_01_{path_tag}.png",
        label=label,
        split=split,
        experience_id=0,
        source_filelist="synthetic/test_filelist.txt",
        line_number=1,
        object_id=label + 1,
        session_id=session,
        category_id=0,
        category_name="synthetic",
        object_name=f"object_{label}",
    )


def representative_fixture() -> tuple[list, dict, dict]:
    """18 test records (6 labels x 3 sessions), all errors.

    Row 0 is a naive/replay disagreement; every other row is a
    both-method error. Session-balanced picks fill 12 examples with the
    per-class cap of 2 while still covering all three sessions.
    """
    labels = range(6)
    candidates = [
        make_record(label, session, f"{label}_{session}")
        for label in labels
        for session in (3, 7, 10)
    ]
    naive, replay = [], []
    for index, record in enumerate(candidates):
        true = record.label
        naive_pred = true if index == 0 else (true + 1) % 6
        replay_pred = (true + 1) % 6
        naive.append(
            {"index": index, "true": true, "predicted": naive_pred, "confidence": 0.9}
        )
        replay.append(
            {"index": index, "true": true, "predicted": replay_pred, "confidence": 0.8}
        )
    names = {str(label): f"class_{label}" for label in labels}
    return candidates, {"naive": naive, "replay": replay}, names


@needs_phase5
def results_fixture() -> dict:
    return load_phase5_results(PHASE5_REPORTS)


# ---------------------------------------------------------------------------
# 1. loading + validation of committed Phase-5 artifacts
# ---------------------------------------------------------------------------


@needs_phase5
def test_load_phase5_results_reads_committed_artifacts() -> None:
    results = results_fixture()
    assert results["num_experiences"] == 79
    assert len(results["records"]["naive"]) == 79
    assert len(results["records"]["replay"]) == 79
    assert len(results["class_names"]) == 50
    assert results["forgetting_definition"].strip()
    assert results["accuracy_definition"].strip()
    assert results["average_incremental_definition"].strip()
    for key in (
        "final_accuracy",
        "final_forgetting",
        "average_incremental_accuracy",
    ):
        assert set(results["comparison"][key]) == {"naive", "replay"}


def test_load_phase5_results_missing_artifacts_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing Phase-5 artifact"):
        load_phase5_results(tmp_path / "empty")


# ---------------------------------------------------------------------------
# 2. forgetting analysis: deterministic, fabrication-free
# ---------------------------------------------------------------------------


@needs_phase5
def test_forgetting_analysis_deterministic_and_tied_to_measurements() -> None:
    results = results_fixture()
    first = build_forgetting_analysis(results)
    second = build_forgetting_analysis(results)
    assert first == second

    summary = results["summary"]["comparison"]
    agg = first["aggregate"]
    assert agg["naive"]["final_overall_accuracy"] == summary["final_accuracy"]["naive"]
    assert agg["replay"]["final_overall_accuracy"] == summary["final_accuracy"]["replay"]
    assert agg["naive"]["final_mean_forgetting"] == summary["final_forgetting"]["naive"]
    delta = agg["delta_replay_minus_naive"]
    assert delta["final_accuracy"] == pytest.approx(
        summary["final_accuracy"]["replay"] - summary["final_accuracy"]["naive"]
    )
    assert delta["final_forgetting"] == pytest.approx(
        summary["final_forgetting"]["replay"] - summary["final_forgetting"]["naive"]
    )
    naive_forgetting = [
        row["forgetting"] for row in first["top_forgetting_classes"]["naive"]
    ]
    assert naive_forgetting == sorted(naive_forgetting, reverse=True)
    assert len(first["experiences_with_largest_forgetting_increase"]) <= 8


def test_final_class_stats_missing_values_stay_none() -> None:
    empty = final_class_stats([None, None, None])
    assert empty["introduced_experience"] is None
    assert empty["final_accuracy"] is None
    assert empty["forgetting"] is None
    assert empty["n_measurements"] == 0

    single = final_class_stats([0.5])
    assert single["forgetting"] is None  # no prior measurement to forget from
    assert single["final_accuracy"] == 0.5

    drop = final_class_stats([0.4, 0.5, 0.35])
    assert drop["forgetting"] == pytest.approx(0.15)
    assert drop["peak_accuracy"] == pytest.approx(0.5)

    improvement = final_class_stats([0.3, 0.4])
    assert improvement["forgetting"] == pytest.approx(-0.1)  # negative preserved


@needs_phase5
def test_class_and_experience_analyses_cover_everything() -> None:
    results = results_fixture()
    class_analysis = build_class_analysis(results)
    experience_analysis = build_experience_analysis(results)
    assert len(class_analysis["classes"]) == 50
    assert len(experience_analysis["experiences"]) == 79
    row = class_analysis["classes"][0]
    assert {"label", "name", "naive", "replay", "difference_replay_minus_naive"} <= set(
        row
    )
    exp = experience_analysis["experiences"][-1]
    assert exp["experience_id"] == 78
    assert {"naive_overall", "replay_overall", "naive_forgetting"} <= set(exp)


# ---------------------------------------------------------------------------
# 3. final model selection: deterministic, evidence-driven
# ---------------------------------------------------------------------------


@needs_phase5
def test_selection_deterministic_and_replay_dominates() -> None:
    results = results_fixture()
    first = select_final_model(results)
    second = select_final_model(results)
    assert first == second

    comparison = results["summary"]["comparison"]
    replay_all = (
        comparison["final_accuracy"]["replay"] > comparison["final_accuracy"]["naive"]
        and comparison["final_forgetting"]["replay"]
        < comparison["final_forgetting"]["naive"]
        and comparison["average_incremental_accuracy"]["replay"]
        > comparison["average_incremental_accuracy"]["naive"]
        and results["records"]["replay"][-1]["accuracy"]["old"]
        > results["records"]["naive"][-1]["accuracy"]["old"]
    )
    assert replay_all, "committed Phase-5 measurements must justify the verdict"
    assert first["selected_method"] == "replay"
    assert [row["criterion"] for row in first["criteria"]] == PRIMARY_CRITERIA
    assert first["rationale"].strip()
    assert first["source_experiment"]["artifacts"] == "reports/phase5_nic/"
    assert first["naive_metrics"]["final_accuracy"] == comparison["final_accuracy"][
        "naive"
    ]
    assert first["replay_metrics"]["final_forgetting"] == comparison["final_forgetting"][
        "replay"
    ]


@needs_phase5
def test_selection_refuses_missing_records() -> None:
    results = results_fixture()
    results["records"] = {"naive": [], "replay": []}
    with pytest.raises(ValueError, match="at least one record"):
        select_final_model(results)


# ---------------------------------------------------------------------------
# 4. targeted candidate + representative-error selection
# ---------------------------------------------------------------------------


def test_candidate_records_cover_sessions_and_reject_train() -> None:
    records = [
        make_record(label, session, f"{label}_{session}_{i}")
        for label in (0, 1, 2)
        for session in (3, 7, 10)
        for i in range(4)
    ]
    picked = select_candidate_records([0, 1, 2], records, max_total=9)
    assert len(picked) == 9
    assert select_candidate_records([0, 1, 2], records, max_total=9) == picked
    for label in (0, 1, 2):
        sessions = {r.session_id for r in picked if r.label == label}
        assert sessions == {3, 7, 10}
    assert select_candidate_records([], records) == []

    with pytest.raises(ValueError, match="split="):
        select_candidate_records(
            [0], records + [make_record(9, 1, "bad", split="train")]
        )


def test_representative_errors_deterministic_and_well_formed() -> None:
    candidates, predictions, names = representative_fixture()
    first = build_representative_errors(candidates, predictions, names)
    second = build_representative_errors(candidates, predictions, names)
    assert first == second

    assert first["candidates_evaluated"] == 18
    assert first["candidates_per_method"] == {"naive": 17, "replay": 18}
    assert first["total_errors"] == {"disagreement": 1, "both_wrong": 17}

    examples = first["examples"]
    assert first["n_examples"] == len(examples) == 12  # session quota fills to cap
    assert examples[0]["disagreement"] is True  # disagreements rank first
    assert {ex["session"] for ex in examples} == {3, 7, 10}  # session balance
    per_class: dict[int, int] = {}
    for ex in examples:
        per_class[ex["true_label"]] = per_class.get(ex["true_label"], 0) + 1
        assert ex["split"] == "test"
        assert ex["checkpoint_experience"]
        assert ex["showcase_method"] in ("naive", "replay", "naive+replay")
        for pred in ex["predictions"].values():
            assert 0.0 <= pred["confidence"] <= 1.0
            assert isinstance(pred["correct"], bool)
    assert max(per_class.values()) <= 2


def test_environment_analysis_avoids_causal_claims() -> None:
    candidates, predictions, names = representative_fixture()
    representative = build_representative_errors(candidates, predictions, names)
    environment = build_environment_analysis(representative)

    assert set(environment["errors_by_session"]) == {"3", "7", "10"}
    assert sum(environment["errors_by_session"].values()) == representative[
        "n_examples"
    ]
    assert ENV_FALLBACK_SENTENCE in environment["possible_factors"]
    assert "No causal environmental claim" in environment["causal_claim_policy"]
    assert environment["observed_evidence"]
    assert environment["observations"]
    for observation in environment["observations"]:
        assert "cannot be established" in observation["possible_factor"]
        assert observation["session"] in (3, 7, 10)
    assert set(environment["session_facts"]["test_sessions"]) == {3, 7, 10}


def test_target_classes_deterministic_and_capped() -> None:
    bucket_row = {"label": 7, "forgetting": 0.9, "final_accuracy": 0.1, "difference": 0.5}
    forgetting = {
        "top_forgetting_classes": {"naive": [bucket_row]},
        "lowest_final_accuracy_classes": {
            "naive": [bucket_row],
            "replay": [{"label": 9, "final_accuracy": 0.2}],
        },
        "largest_naive_replay_differences": [
            {"label": 3, "difference": -0.7}
        ],
    }
    targets = select_target_classes(forgetting)
    assert targets == sorted(set(targets))  # unique + sorted by label
    assert set(targets) == {3, 7, 9}
    assert select_target_classes(forgetting) == targets
    assert len(select_target_classes(forgetting, per_bucket=1, max_classes=2)) <= 2


# ---------------------------------------------------------------------------
# 5. final checkpoint metadata + freeze integrity
# ---------------------------------------------------------------------------


def test_final_model_metadata_validates_portable_paths_and_hash() -> None:
    good_hash = "a" * 64
    payload = build_final_model_metadata(
        method="replay",
        source_checkpoint="models/continual/phase5_nic/replay/checkpoint.pt",
        final_checkpoint="models/continual/final_model.pt",
        sha256=good_hash,
        scenario="NIC",
        variant="inc",
        run=0,
        seed=42,
        num_classes=50,
        arch="small_cnn",
        width=32,
        image_size=64,
        class_mapping="data/raw/core50/metadata/object_mapping.json",
    )
    assert payload["sha256"] == good_hash
    assert payload["num_classes"] == 50
    assert not Path(payload["final_checkpoint"]).is_absolute()

    with pytest.raises(ValueError, match="project-relative"):
        build_final_model_metadata(
            method="replay",
            source_checkpoint=r"C:\abs\source.pt",
            final_checkpoint="models/continual/final_model.pt",
            sha256=good_hash,
            scenario="NIC",
            variant="inc",
            run=0,
            seed=42,
        )
    with pytest.raises(ValueError, match="64 lowercase hex"):
        build_final_model_metadata(
            method="replay",
            source_checkpoint="models/a.pt",
            final_checkpoint="models/continual/final_model.pt",
            sha256="nope",
            scenario="NIC",
            variant="inc",
            run=0,
            seed=42,
        )


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"abc")
    assert sha256_file(target) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@needs_phase5
@needs_final
def test_frozen_checkpoint_matches_source_and_metadata() -> None:
    metadata = json.loads(FINAL_META.read_text(encoding="utf-8"))
    source = PROJECT_ROOT / metadata["source_checkpoint"]
    assert not Path(metadata["final_checkpoint"]).is_absolute()
    assert metadata["phase"] == 6
    assert metadata["method"] in ("naive", "replay")
    assert sha256_file(FINAL_CKPT) == metadata["sha256"] == sha256_file(source)
    assert metadata["num_classes"] == 50


@needs_final
def test_final_model_loads_with_full_class_mapping() -> None:
    metadata = json.loads(FINAL_META.read_text(encoding="utf-8"))
    payload = torch.load(FINAL_CKPT, map_location="cpu", weights_only=False)
    model = build_model(int(payload["num_classes"]), width=int(metadata["width"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, int(metadata["image_size"]), int(metadata["image_size"])))
    assert logits.shape == (1, int(metadata["num_classes"]))

    if OBJECT_MAPPING.is_file():
        mapping = load_class_names(OBJECT_MAPPING)
        assert len(mapping) == int(payload["num_classes"])


# ---------------------------------------------------------------------------
# 6. generated reports: present + free of absolute personal paths
# ---------------------------------------------------------------------------


@needs_analysis
def test_analysis_reports_exist_and_are_portable() -> None:
    required = [
        "forgetting_analysis.json",
        "class_analysis.json",
        "experience_analysis.json",
        "phase6_error_analysis.md",
        "representative_errors.json",
        "environment_analysis.json",
        "final_model_selection.json",
        "final_model_selection.md",
        "phase6_summary.md",
        "phase6_analysis.json",
    ]
    for name in required:
        assert (ANALYSIS_DIR / name).is_file(), f"missing {name}"

    offenders = []
    for path in sorted(ANALYSIS_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".md"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in ABS_MARKERS:
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, f"absolute paths leaked: {offenders}"


@needs_analysis
def test_representative_errors_artifact_metadata_valid() -> None:
    payload = json.loads(
        (ANALYSIS_DIR / "representative_errors.json").read_text(encoding="utf-8")
    )
    assert 6 <= payload["n_examples"] <= 12
    assert payload["candidates_evaluated"] <= 100
    assert set(payload["candidates_per_method"]) == {"naive", "replay"}
    sessions = set()
    for example in payload["examples"]:
        sessions.add(example["session"])
        assert example["split"] == "test"
        assert example["true_name"]
        assert set(example["predictions"]) == {"naive", "replay"}
        for pred in example["predictions"].values():
            assert 0.0 <= pred["confidence"] <= 1.0
    assert sessions <= {3, 7, 10}


# ---------------------------------------------------------------------------
# 7. markdown renderers over committed artifacts
# ---------------------------------------------------------------------------


@needs_phase5
def test_renderers_emit_readable_reports_without_absolute_paths() -> None:
    results = results_fixture()
    forgetting = build_forgetting_analysis(results)
    public = {k: v for k, v in forgetting.items() if not k.startswith("_")}
    class_analysis = build_class_analysis(results)
    selection = select_final_model(results)
    placeholder = build_experiment_placeholder()

    analysis_md = render_error_analysis_md(
        public, class_analysis, placeholder, placeholder
    )
    selection_md = render_selection_md(selection)
    summary_md = render_summary_md(
        results=results,
        selection=selection,
        forgetting=public,
        representative=placeholder,
        environment=placeholder,
        sha256="b" * 64,
        final_checkpoint="models/continual/final_model.pt",
    )

    assert "# Phase 6" in analysis_md
    assert "Selected method" in selection_md
    assert "REPLAY" in selection_md
    assert "PHASE 6" in summary_md
    assert "models/continual/final_model.pt" in summary_md
    for text in (analysis_md, selection_md, summary_md):
        for marker in ABS_MARKERS:
            assert marker not in text


# ---------------------------------------------------------------------------
# 8. progress display additions stay additive
# ---------------------------------------------------------------------------


def test_progress_eta_own_line_and_sample_wording() -> None:
    stream = StringIO()
    progress = PhaseProgress(
        "PHASE 6 OVERALL",
        ["step one"],
        stream=stream,
        eta_own_line=True,
        sample_wording=True,
    )
    progress.update(5, 10, force=True)
    lines = progress.build_lines()
    assert "Current sample: 5 / 10" in lines
    assert "Progress: 50.0%" in lines
    elapsed_index = next(i for i, l in enumerate(lines) if l.startswith("Elapsed:"))
    assert lines[elapsed_index + 1].startswith("ETA: ")

    plain = PhaseProgress("PHASE 6 OVERALL", ["step one"], stream=StringIO())
    plain.update(5, 10, force=True)
    plain_lines = plain.build_lines()
    assert not any(line.startswith("Current sample") for line in plain_lines)
    assert any(line.startswith("Elapsed: ") for line in plain_lines)
    assert not any(line.startswith("ETA:") for line in plain_lines)


# ---------------------------------------------------------------------------
# 9. driver smoke test (only when every input exists locally)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PHASE5_PRESENT and CHECKPOINTS_PRESENT and FILELIST_DIR.is_dir()),
    reason="Phase-5 reports/checkpoints or CORe50 filelists not present",
)
def test_phase6_driver_runs_end_to_end() -> None:
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_phase6_analysis.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "PHASE 6 — ERROR ANALYSIS + FINAL MODEL SELECTION COMPLETE" in proc.stdout
    assert "PHASE 6 BLOCKED" not in proc.stdout

    summary = json.loads(
        (ANALYSIS_DIR / "phase6_analysis.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "complete"
    assert summary["selected_method"] == "replay"
    assert all(
        result["status"] in ("PASS", "PRESERVED")
        for result in summary["integrity"].values()
    )
