"""Phase 3 tests — official continual scenario loading, ordering and leakage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.data import core50
from src.data.continual import (
    ExperienceNotFoundError,
    MalformedFilelistError,
    RunNotFoundError,
    ScenarioNotFoundError,
    VariantNotFoundError,
    build_manifest,
    build_scenario_manifest,
    list_scenarios,
    load_scenario,
    load_scenario_cached,
    validate_scenario,
    write_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = ["plug adapter", "mobile phone"]


def _mapping_payload() -> dict[str, Any]:
    return {
        "category_order": CATEGORIES,
        "objects": [
            {
                "object_id": 1,
                "directory": "o1",
                "name": "plug_adapter1",
                "category": CATEGORIES[0],
            },
            {
                "object_id": 2,
                "directory": "o2",
                "name": "plug_adapter2",
                "category": CATEGORIES[0],
            },
        ],
    }


@pytest.fixture(scope="module")
def ni_scenario():
    return load_scenario("NI", variant="inc", run=0)


@pytest.fixture(scope="module")
def nc_scenario():
    return load_scenario("NC", variant="inc", run=0)


@pytest.fixture(scope="module")
def nic_scenario():
    return load_scenario("NIC", variant="inc", run=0)


@pytest.fixture
def synth(tmp_path):
    """Factory building an official-style filelist tree with tiny fake images."""

    def build(
        *,
        batches: dict[int, list[str]],
        test_lines: list[str],
        scenario: str = "NI",
        variant: str = "inc",
        run: int = 0,
        missing_images: tuple[str, ...] = (),
        omit_test_filelist: bool = False,
    ) -> dict[str, Path]:
        root = tmp_path / "core50_synth"
        run_dir = root / "filelists" / f"{scenario}_{variant}" / f"run{run}"
        run_dir.mkdir(parents=True, exist_ok=True)
        images_root = root / "images"
        mapping_path = root / "object_mapping.json"
        mapping_path.write_text(
            json.dumps(_mapping_payload()), encoding="utf-8"
        )

        for index, lines in sorted(batches.items()):
            (run_dir / f"train_batch_{index:02d}_filelist.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        if not omit_test_filelist:
            (run_dir / "test_filelist.txt").write_text(
                "\n".join(test_lines) + "\n", encoding="utf-8"
            )

        referenced = {
            line.split(" ", 1)[0]
            for lines in list(batches.values()) + [test_lines]
            for line in lines
        }
        for rel_path in sorted(referenced):
            if rel_path in missing_images:
                continue
            target = images_root.joinpath(*rel_path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake")

        return {
            "filelist_root": root / "filelists",
            "images_root": images_root,
            "object_mapping": mapping_path,
            "run_dir": run_dir,
        }

    return build


def _load(paths: dict[str, Path], scenario: str = "NI", **kwargs):
    return load_scenario(
        scenario,
        filelist_root=paths["filelist_root"],
        images_root=paths["images_root"],
        object_mapping=paths["object_mapping"],
        **kwargs,
    )


# --- 1. project path resolution --------------------------------------------


def test_project_path_resolution(ni_scenario) -> None:
    assert core50.PROJECT_ROOT == PROJECT_ROOT
    assert core50.FILELIST_DIR == PROJECT_ROOT / "data" / "raw" / "core50" / "filelists"
    metadata = ni_scenario.metadata
    assert metadata["filelist_root"] == "data/raw/core50/filelists"
    assert metadata["images_root"] == "data/raw/core50/dataset/core50_128x128"
    assert metadata["object_mapping"] == "data/raw/core50/metadata/object_mapping.json"
    assert metadata["batch_files"]
    for entry in metadata["batch_files"] + [metadata["evaluation_source"]]:
        assert not Path(entry).is_absolute()
        assert "\\" not in entry
        assert ":" not in entry
    assert str(PROJECT_ROOT) not in json.dumps(metadata)


# --- 2. scenario discovery --------------------------------------------------


def test_scenario_discovery() -> None:
    variants = list_scenarios()
    names = [variant.name for variant in variants]
    assert names == [
        "NI_inc",
        "NI_cum",
        "NC_inc",
        "NC_cum",
        "NIC_inc",
        "NIC_cum",
        "NIC_v2_196",
        "NIC_v2_391",
        "NIC_v2_79",
    ]
    by_name = {variant.name: variant for variant in variants}
    assert by_name["NI_inc"].runs == tuple(range(10))
    assert by_name["NC_cum"].runs == tuple(range(5))
    assert by_name["NIC_cum"].runs == tuple(range(3))
    assert by_name["NIC_v2_391"].runs == tuple(range(10))
    assert by_name["NIC_inc"].scenario_type == "NIC"
    assert by_name["NIC_inc"].variant == "inc"


# --- 3/4/5. NI / NC / NIC loading ------------------------------------------


def test_ni_scenario_loading(ni_scenario) -> None:
    assert ni_scenario.scenario_type == "NI"
    assert ni_scenario.variant == "inc"
    assert len(ni_scenario.experiences) == 8
    assert ni_scenario.metadata["train_samples_total"] == 119_894
    assert ni_scenario.metadata["evaluation_samples"] == 44_972
    assert ni_scenario.metadata["train_unique_paths"] == 119_894
    assert ni_scenario.metadata["train_sessions"] == [1, 2, 4, 5, 6, 8, 9, 11]
    assert ni_scenario.metadata["evaluation_sessions"] == [3, 7, 10]
    assert ni_scenario.metadata["label_equals_object_minus_one"] is True
    assert ni_scenario.metadata["label_min"] == 0
    assert ni_scenario.metadata["label_max"] == 49


def test_nc_scenario_loading(nc_scenario) -> None:
    assert nc_scenario.scenario_type == "NC"
    assert len(nc_scenario.experiences) == 9
    assert nc_scenario.metadata["train_samples_total"] == 119_894
    assert nc_scenario.metadata["evaluation_samples"] == 44_972
    # Official NC filelists use per-run remapped labels (create_sII_filelist.py
    # change_labels=True) — NOT label == object_id - 1 and NOT category ids.
    assert nc_scenario.metadata["label_equals_object_minus_one"] is False
    first = nc_scenario.get_experience(0)
    assert len(first.classes_introduced) == 10
    assert len(first.objects_introduced) == 10
    # official NC batches span every training session in each batch
    assert first.sessions_present == (1, 2, 4, 5, 6, 8, 9, 11)
    last = nc_scenario.get_experience(-1)
    assert len(last.objects_introduced) == 5
    assert last.objects_seen == tuple(range(1, 51))


def test_nic_scenario_loading(nic_scenario) -> None:
    assert nic_scenario.scenario_type == "NIC"
    assert len(nic_scenario.experiences) == 79
    assert nic_scenario.metadata["train_samples_total"] == 119_894
    assert nic_scenario.metadata["evaluation_samples"] == 44_972
    assert nic_scenario.metadata["label_equals_object_minus_one"] is True
    first = nic_scenario.get_experience(0)
    assert first.objects_introduced == (1, 6, 11, 16, 21, 26, 31, 36, 41, 46)
    assert first.sessions_present == (1,)
    last = nic_scenario.get_experience(-1)
    assert last.objects_seen == tuple(range(1, 51))
    assert last.classes_seen == tuple(range(50))


# --- 6. explicit run / variant selection -----------------------------------


def test_explicit_run_selection() -> None:
    default = load_scenario("NI")
    assert default.variant == "inc"
    assert default.run_id == 0
    other_run = load_scenario("NI", variant="inc", run=3)
    assert other_run.run_id == 3
    assert len(other_run.experiences) == 8
    assert other_run.get_experience(0).train_source.endswith(
        "NI_inc/run3/train_batch_00_filelist.txt"
    )
    with pytest.raises(RunNotFoundError):
        load_scenario("NI", variant="inc", run=99)


def test_explicit_variant_selection() -> None:
    cumulative = load_scenario("NI", variant="cum", run=0)
    assert cumulative.variant == "cum"
    assert len(cumulative.experiences) == 8
    assert cumulative.metadata["train_samples_total"] == 539_618
    report = validate_scenario(cumulative, check_paths_exist=False)
    assert report.all_passed
    # cumulative growth: last experience is the largest
    counts = [exp.train_count for exp in cumulative.experiences]
    assert counts == sorted(counts)
    assert counts[-1] == 119_894


# --- 7. experience ordering -------------------------------------------------


def test_experience_ordering(ni_scenario) -> None:
    ids = [exp.experience_id for exp in ni_scenario.iter_experiences()]
    assert ids == list(range(8))
    sources = [exp.train_source for exp in ni_scenario.iter_experiences()]
    for index, source in enumerate(sources):
        assert source.endswith(f"train_batch_{index:02d}_filelist.txt")
    report = validate_scenario(ni_scenario, check_paths_exist=False)
    ordering = next(c for c in report.checks if c.name == "experience_ordering")
    assert ordering.passed


# --- 8. first/last experience identification --------------------------------


def test_first_last_experience_identification(ni_scenario) -> None:
    first = ni_scenario.get_experience(0)
    last = ni_scenario.get_experience(-1)
    assert first.experience_id == 0
    assert first.train_source.endswith("train_batch_00_filelist.txt")
    assert last.experience_id == 7
    assert last.train_source.endswith("train_batch_07_filelist.txt")
    with pytest.raises(ExperienceNotFoundError):
        ni_scenario.get_experience(8)
    with pytest.raises(IndexError):
        ni_scenario.get_experience(-9)


# --- 9. train / evaluation separation ---------------------------------------


def test_train_evaluation_separation(ni_scenario) -> None:
    eval_paths = {
        record.relative_path for record in ni_scenario.iter_evaluation_samples()
    }
    train_paths = {
        record.relative_path for record in ni_scenario.iter_train_samples()
    }
    assert eval_paths and train_paths
    assert eval_paths.isdisjoint(train_paths)
    eval_sessions = {record.session_id for record in ni_scenario.iter_evaluation_samples()}
    train_sessions = {record.session_id for record in ni_scenario.iter_train_samples()}
    assert eval_sessions == {3, 7, 10}
    assert train_sessions == {1, 2, 4, 5, 6, 8, 9, 11}
    assert eval_sessions.isdisjoint(train_sessions)
    report = validate_scenario(ni_scenario, check_paths_exist=False)
    by_name = {check.name: check for check in report.checks}
    assert by_name["train_evaluation_overlap"].passed
    assert by_name["evaluation_stable"].passed
    assert by_name["session_integrity"].passed


# --- 10. no future-training leakage ----------------------------------------


def test_no_future_training_leakage(ni_scenario, nc_scenario) -> None:
    for scenario in (ni_scenario, nc_scenario):
        report = validate_scenario(scenario, check_paths_exist=False)
        leakage = next(
            c for c in report.checks if c.name == "future_training_leakage"
        )
        assert leakage.passed, leakage.detail


def test_future_leakage_is_detected(synth) -> None:
    paths = synth(
        batches={
            0: ["s1/o1/C_01_01_000.png 0", "s1/o2/C_01_02_000.png 1"],
            1: ["s1/o1/C_01_01_000.png 0", "s4/o1/C_04_01_000.png 0"],
        },
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    scenario = _load(paths)
    report = validate_scenario(scenario, check_paths_exist=False)
    leakage = next(c for c in report.checks if c.name == "future_training_leakage")
    assert not leakage.passed
    assert not report.all_passed


def test_cumulative_chain_is_checked(synth) -> None:
    good = synth(
        scenario="NI",
        variant="cum",
        batches={
            0: ["s1/o1/C_01_01_000.png 0"],
            1: ["s1/o1/C_01_01_000.png 0", "s4/o1/C_04_01_000.png 0"],
        },
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    scenario = _load(good, variant="cum")
    report = validate_scenario(scenario, check_paths_exist=False)
    assert report.all_passed

    broken = synth(
        scenario="NC",
        variant="cum",
        batches={
            0: ["s1/o1/C_01_01_000.png 0", "s4/o1/C_04_01_000.png 0"],
            1: ["s4/o1/C_04_01_000.png 0"],
        },
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    scenario = _load(broken, scenario="NC", variant="cum")
    report = validate_scenario(scenario, check_paths_exist=False)
    leakage = next(c for c in report.checks if c.name == "future_training_leakage")
    assert not leakage.passed


# --- 11. no random reordering -----------------------------------------------


def test_no_random_reordering(ni_scenario) -> None:
    official_first_line = (
        PROJECT_ROOT
        / "data/raw/core50/filelists/NI_inc/run0/train_batch_00_filelist.txt"
    ).read_text(encoding="utf-8").splitlines()[0].split(" ", 1)[0]
    first_sample = ni_scenario.get_experience(0).train_samples[0]
    assert first_sample.relative_path == official_first_line
    streamed = list(ni_scenario.iter_train_samples())
    assert streamed[0] is ni_scenario.get_experience(0).train_samples[0]
    assert streamed[-1] is ni_scenario.get_experience(-1).train_samples[-1]


# --- 12. object / session / class metadata extraction -----------------------


def test_metadata_extraction(ni_scenario) -> None:
    record = ni_scenario.get_experience(0).train_samples[0]
    assert record.relative_path == "s11/o1/C_11_01_000.png"
    assert record.label == 0
    assert record.split == "train"
    assert record.experience_id == 0
    assert record.object_id == 1
    assert record.session_id == 11
    assert record.category_id == 0
    assert record.category_name == "plug adapter"
    assert record.object_name == "plug_adapter1"
    assert ni_scenario.get_experience(-1).categories_seen == tuple(range(10))
    resolved = ni_scenario.image_path(record)
    assert resolved.is_file()
    payload = record.to_dict()
    assert payload["relative_path"] == record.relative_path
    assert "image_path" not in payload


# --- 13. unresolved path detection ------------------------------------------


def test_unresolved_path_detection(synth) -> None:
    paths = synth(
        batches={0: ["s1/o1/C_01_01_000.png 0"]},
        test_lines=["s3/o1/C_03_01_000.png 0"],
        missing_images=("s1/o1/C_01_01_000.png",),
    )
    scenario = _load(paths)
    report = validate_scenario(scenario, check_paths_exist=True)
    resolution = next(c for c in report.checks if c.name == "path_resolution")
    assert not resolution.passed
    assert "C_01_01_000.png" in resolution.detail
    report_without = validate_scenario(scenario, check_paths_exist=False)
    assert report_without.all_passed


# --- 14. duplicate reference detection --------------------------------------


def test_duplicate_reference_detection(synth) -> None:
    paths = synth(
        batches={
            0: [
                "s1/o1/C_01_01_000.png 0",
                "s1/o1/C_01_01_000.png 0",
            ],
        },
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    scenario = _load(paths)
    report = validate_scenario(scenario, check_paths_exist=False)
    duplicates = next(
        c for c in report.checks if c.name == "duplicate_references"
    )
    assert not duplicates.passed

    inconsistent = synth(
        scenario="NC",
        variant="inc",
        batches={
            0: [
                "s1/o1/C_01_01_000.png 0",
                "s1/o1/C_01_01_000.png 7",
            ],
        },
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    with pytest.raises(MalformedFilelistError):
        _load(inconsistent, scenario="NC")

    eval_dup = synth(
        scenario="NIC",
        variant="inc",
        batches={0: ["s1/o1/C_01_01_000.png 0"]},
        test_lines=[
            "s3/o1/C_03_01_000.png 0",
            "s3/o1/C_03_01_000.png 0",
        ],
    )
    with pytest.raises(MalformedFilelistError):
        _load(eval_dup, scenario="NIC")


# --- 15. relative-path serialization ----------------------------------------


def test_relative_path_serialization(ni_scenario) -> None:
    report = validate_scenario(ni_scenario, check_paths_exist=False)
    manifest = build_scenario_manifest(ni_scenario, report)
    text = json.dumps(manifest)
    assert str(PROJECT_ROOT) not in text
    assert "C:" not in text
    assert "/home/" not in text
    assert "\\" not in text


# --- 16. scenario manifest generation ---------------------------------------


def test_scenario_manifest_generation(tmp_path, ni_scenario) -> None:
    report = validate_scenario(ni_scenario, check_paths_exist=False)
    entry = build_scenario_manifest(ni_scenario, report)
    assert entry["scenario"] == "NI"
    assert entry["variant"] == "inc"
    assert entry["run"] == 0
    assert entry["experience_count"] == 8
    assert len(entry["experiences"]) == 8
    first = entry["experiences"][0]
    for key in (
        "train_count",
        "evaluation_count",
        "classes_introduced",
        "classes_seen",
        "objects_introduced",
        "objects_seen",
        "sessions_present",
        "sessions_seen",
        "train_source",
    ):
        assert key in first
    assert entry["validation"]["all_passed"] is True
    assert entry["unresolved_references"] == []
    assert entry["leakage_check_results"]
    assert all(item["passed"] for item in entry["leakage_check_results"])

    aggregate = build_manifest([ni_scenario], {ni_scenario.name: report})
    assert aggregate["phase"] == 3
    assert aggregate["dataset"] == "CORe50"
    assert aggregate["dataset_root"] == "data/raw/core50/dataset/core50_128x128"
    assert aggregate["image_copies_created"] == 0
    assert aggregate["dataset_modified"] is False
    assert aggregate["primary_future_scenario"] == "NIC"
    assert aggregate["run_selection"]["all_checks_passed"] is True
    out = write_manifest(aggregate, tmp_path / "manifest.json")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == aggregate


# --- 17. deterministic repeated loading --------------------------------------


def test_deterministic_repeated_loading(ni_scenario) -> None:
    again = load_scenario("NI", variant="inc", run=0)
    assert again.experiences == ni_scenario.experiences
    assert again.metadata == ni_scenario.metadata
    cached = load_scenario_cached("NI", "inc", 0)
    assert cached.experiences == ni_scenario.experiences


# --- 18. empty / malformed input handling -----------------------------------


def test_empty_and_malformed_input_handling(synth) -> None:
    empty = synth(
        variant="empty",
        batches={0: []},
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    with pytest.raises(MalformedFilelistError):
        _load(empty, variant="empty")

    no_separator = synth(
        variant="noseparator",
        batches={0: ["s1/o1/C_01_01_000.png"]},
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    with pytest.raises(MalformedFilelistError):
        _load(no_separator, variant="noseparator")

    bad_label = synth(
        variant="badlabel",
        batches={0: ["s1/o1/C_01_01_000.png oops"]},
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    with pytest.raises(MalformedFilelistError):
        _load(bad_label, variant="badlabel")

    bad_path = synth(
        variant="badpath",
        batches={0: ["not/a/path.png 0"]},
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    with pytest.raises(MalformedFilelistError):
        _load(bad_path, variant="badpath")

    no_test = synth(
        variant="notest",
        batches={0: ["s1/o1/C_01_01_000.png 0"]},
        test_lines=[],
        omit_test_filelist=True,
    )
    with pytest.raises(MalformedFilelistError):
        _load(no_test, variant="notest")

    gapped = synth(
        variant="gap",
        batches={0: ["s1/o1/C_01_01_000.png 0"], 2: ["s4/o1/C_04_01_000.png 0"]},
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    with pytest.raises(MalformedFilelistError):
        _load(gapped, variant="gap")

    unknown_object = synth(
        variant="unknownobject",
        batches={0: ["s1/o9/C_01_09_000.png 0"]},
        test_lines=["s3/o1/C_03_01_000.png 0"],
    )
    with pytest.raises(Exception) as excinfo:
        _load(unknown_object, variant="unknownobject")
    assert "unknown object id 9" in str(excinfo.value)


# --- 19. invalid scenario ----------------------------------------------------


def test_invalid_scenario_handling() -> None:
    with pytest.raises(ScenarioNotFoundError) as excinfo:
        load_scenario("XYZ")
    assert "NI_inc" in str(excinfo.value)


# --- 20. invalid run / variant ----------------------------------------------


def test_invalid_run_and_variant_handling() -> None:
    with pytest.raises(VariantNotFoundError) as excinfo:
        load_scenario("NIC", variant="bogus")
    assert "inc" in str(excinfo.value)
    with pytest.raises(RunNotFoundError) as excinfo:
        load_scenario("NIC", variant="cum", run=7)
    assert "0, 1, 2" in str(excinfo.value)
