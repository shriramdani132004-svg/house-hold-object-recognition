"""Phase 2 tests — CORe50 dataset acquisition, structure, docs and tooling."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.data import core50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_IMAGES = core50.DATASET_DIR / "core50_128x128"
MANIFEST_PATH = core50.DOWNLOAD_DIR / "MANIFEST.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "phase2_core50_summary.json"
MAPPING_PATH = core50.METADATA_DIR / "object_mapping.json"
SAMPLES_DIR = PROJECT_ROOT / "reports" / "phase2_core50_samples"

DATASET_PRESENT = DATASET_IMAGES.is_dir()


# --- pure constants (no dataset needed) -------------------------------------


def test_session_and_object_constants() -> None:
    assert core50.SESSION_IDS == tuple(range(1, 12))
    assert core50.TEST_SESSIONS == (3, 7, 10)
    assert set(core50.TRAIN_SESSIONS) == set(core50.SESSION_IDS) - {3, 7, 10}
    assert core50.OBJECT_IDS == tuple(range(1, 51))
    assert len(core50.OFFICIAL_CATEGORY_ORDER) == 10


def test_object_category_mapping_is_official_order() -> None:
    expected = {
        **{i: "plug adapter" for i in range(1, 6)},
        **{i: "mobile phone" for i in range(6, 11)},
        **{i: "scissors" for i in range(11, 16)},
        **{i: "light bulb" for i in range(16, 21)},
        **{i: "can" for i in range(21, 26)},
        **{i: "glass" for i in range(26, 31)},
        **{i: "ball" for i in range(31, 36)},
        **{i: "marker" for i in range(36, 41)},
        **{i: "cup" for i in range(41, 46)},
        **{i: "remote control" for i in range(46, 51)},
    }
    assert core50.OBJECT_CATEGORY == expected
    assert core50.OBJECT_SLUG[1] == "plug_adapter"
    assert core50.OBJECT_SLUG[50] == "remote_control"


def test_name_helpers() -> None:
    assert core50.session_name(3) == "s3"
    assert core50.object_name(46) == "o46"
    assert core50.frame_name(1, 7, 150) == "C_01_07_150.png"


def test_scenarios_declared() -> None:
    assert core50.SCENARIOS == ("NI", "NC", "NIC")


def test_phase2_sources_have_no_hardcoded_personal_paths() -> None:
    files = [
        PROJECT_ROOT / "scripts" / "download_core50.py",
        PROJECT_ROOT / "scripts" / "extract_core50.py",
        PROJECT_ROOT / "scripts" / "validate_core50.py",
        PROJECT_ROOT / "src" / "data" / "core50.py",
        PROJECT_ROOT / "src" / "utils" / "progress.py",
    ]
    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "C:\\Users" in text or "C:/Users" in text or "/home/" in text:
            offenders.append(path.name)
    assert not offenders, f"Hardcoded personal paths in: {offenders}"


# --- documentation ----------------------------------------------------------


def test_core50_documentation_present() -> None:
    info = PROJECT_ROOT / "docs" / "dataset" / "core50" / "DATASET_INFO.md"
    readme = PROJECT_ROOT / "docs" / "dataset" / "core50" / "README.md"
    dataset_md = PROJECT_ROOT / "DATASET.md"
    for path in (info, readme, dataset_md):
        assert path.is_file(), f"Missing {path}"
    text = info.read_text(encoding="utf-8")
    for heading in (
        "# CORe50 Dataset",
        "## Official Source",
        "## Object Identities",
        "## Categories",
        "## Sessions",
        "## Dataset Layout",
        "## Continual Scenarios",
        "## Filelists",
        "## Acquisition Method",
        "## License / Usage",
        "## Reproducibility",
        "## Integrity Verification",
    ):
        assert heading in text, f"DATASET_INFO.md missing {heading!r}"
    assert "vlomonaco.github.io/core50" in text
    assert "creativecommons.org/licenses/by/4.0" in text
    assert "truncat" in text.lower(), "truncated-host issue must be documented"


def test_dataset_md_marks_core50_as_primary() -> None:
    text = (PROJECT_ROOT / "DATASET.md").read_text(encoding="utf-8")
    assert "CORe50" in text
    # original headings kept for the legacy prototype tests
    for heading in (
        "# Dataset",
        "## Selected Dataset",
        "## Official Source",
        "## Dataset Version",
        "## License",
        "## Why This Dataset Was Selected",
        "## Dataset Size",
        "## Relevant Classes",
        "## Annotation Format",
        "## Acquisition Method",
        "## Citation / Attribution",
        "## Limitations",
    ):
        assert heading in text, f"DATASET.md missing heading {heading!r}"


# --- validation report ------------------------------------------------------


def test_validation_summary_passed() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["phase"] == 2
    assert summary["dataset"] == "CORe50"
    assert summary["status"] == "PASS"
    assert summary["failures"] == []
    assert summary["total_images"] == 164_866
    assert summary["object_count"] == 50
    assert summary["category_count"] == 10
    assert summary["session_count"] == 11
    assert summary["test_sessions"] == ["s3", "s7", "s10"]
    assert summary["filelist_unresolved_paths"] == 0
    assert summary["integrity"]["zero_byte_files"] == 0
    assert summary["integrity"]["png_signature_failures"] == 0
    assert summary["integrity"]["official_paths_match_disk"] is True
    assert summary["samples"]["count"] == 16
    assert summary["samples"]["categories"] == 10


def test_object_mapping_consistent() -> None:
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    assert mapping["category_order"] == list(core50.OFFICIAL_CATEGORY_ORDER)
    objects = mapping["objects"]
    assert len(objects) == 50
    assert [o["object_id"] for o in objects] == list(range(1, 51))
    for obj in objects:
        assert obj["category"] == core50.OBJECT_CATEGORY[obj["object_id"]]
        assert obj["directory"] == f"o{obj['object_id']}"
        assert sum(obj["images_per_session"].values()) == obj["images"]
    assert sum(o["images"] for o in objects) == 164_866


def test_sample_outputs_present() -> None:
    pngs = sorted(SAMPLES_DIR.glob("*.png"))
    assert len(pngs) == 16, f"Expected 16 sample images, found {len(pngs)}"
    for png in pngs:
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", png.name
    assert (SAMPLES_DIR / "README.md").is_file()


# --- dataset on disk (skipped when the dataset is absent) -------------------


@pytest.mark.skipif(not DATASET_PRESENT, reason="CORe50 dataset not downloaded")
def test_dataset_session_structure() -> None:
    for session_id in core50.SESSION_IDS:
        session_dir = DATASET_IMAGES / f"s{session_id}"
        assert session_dir.is_dir(), f"Missing {session_dir.name}"
        object_dirs = sorted(
            (p.name for p in session_dir.iterdir() if p.is_dir()),
            key=lambda name: int(name[1:]),
        )
        assert object_dirs == [f"o{i}" for i in core50.OBJECT_IDS]


@pytest.mark.skipif(not DATASET_PRESENT, reason="CORe50 dataset not downloaded")
def test_sample_frame_resolves_on_disk() -> None:
    frame = DATASET_IMAGES / "s1" / "o1" / "C_01_01_150.png"
    assert frame.is_file()
    assert frame.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="No download manifest")
def test_download_manifest_has_hashes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest.get("resources", manifest)
    assert isinstance(entries, (list, dict))
    text = json.dumps(manifest)
    assert "sha256" in text
    assert "core50_128x128.zip" in text


@pytest.mark.skipif(not core50.FILELIST_DIR.is_dir(), reason="No filelists")
def test_official_filelist_scenarios_present() -> None:
    scenarios = sorted(p.name for p in core50.FILELIST_DIR.iterdir() if p.is_dir())
    for expected in ("NI_inc", "NI_cum", "NC_inc", "NC_cum", "NIC_inc", "NIC_cum"):
        assert expected in scenarios, f"Missing filelist scenario {expected}"
    run_dir = core50.FILELIST_DIR / "NI_inc" / "run0"
    assert (run_dir / "test_filelist.txt").is_file()
    first_line = (run_dir / "test_filelist.txt").read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("s")


# --- git hygiene ------------------------------------------------------------


def test_core50_dataset_is_git_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/raw/core50/dataset"],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    if result.returncode == 128:  # git unavailable / not a repo
        pytest.skip("git not available")
    assert result.returncode == 0, "data/raw/core50/dataset must be git-ignored"


def test_core50_docs_are_trackable() -> None:
    for rel in ("docs/dataset/core50/DATASET_INFO.md", "docs/dataset/core50/README.md"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
        if result.returncode == 128:
            pytest.skip("git not available")
        assert result.returncode == 1, f"{rel} must NOT be git-ignored"
