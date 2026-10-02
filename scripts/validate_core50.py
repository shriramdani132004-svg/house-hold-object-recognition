"""Validate the acquired CORe50 dataset (Phase 2, steps 5 and 6).

Run from the project root:

    .\\.venv\\Scripts\\python.exe scripts/validate_core50.py

Checks performed (all against official CORe50 metadata, no network):

Structure (step 5)
  * dataset root, 11 sessions, 50 object directories per session
  * official frame file naming (C_ss_oo_fff.png) matches its directories
  * total image count vs the official Color128x128.tsv dims table
  * per session/object counts vs the official dims table
  * no zero-byte files, no unexpected extensions
  * PNG signature check on every image
  * official paths.pkl cross-check (set equality with disk contents)

Sessions / objects / filelists (step 6)
  * representative image decode across sessions, objects and categories
  * NI / NC / NIC filelists present with official run and batch counts
  * every filelist path resolves to a real dataset file
  * fixed test set is identical across runs of a scenario
  * object identity mapping (50 objects -> name, category) written out
  * bounding-box metadata parses
  * small deterministic human-inspection sample + summary JSON written
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.data.core50 import (  # noqa: E402
    DATASET_DIR,
    DOWNLOAD_DIR,
    FILELIST_DIR,
    METADATA_DIR,
    OBJECT_CATEGORY,
    OFFICIAL_CATEGORY_ORDER,
    OFFICIAL_PAGE,
    OFFICIAL_REPO,
    PROJECT_ROOT as ROOT,
    SESSION_IDS,
    TEST_SESSIONS,
    TRAIN_SESSIONS,
)
from src.utils.progress import PhaseProgress, format_duration  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FRAME_RE = re.compile(r"^C_(\d{2})_(\d{2})_(\d{3})\.png$")
EXPECTED_IMAGES = 164_866
EXPECTED_SESSIONS = 11
EXPECTED_OBJECTS = 50
# Official train-batch counts per scenario directory (official data loader
# nbatch table: ni=8, nc=9, nic=79; NICv2 variants are named after their
# experience counts).
EXPECTED_BATCHES: dict[str, int] = {
    "NI_inc": 8,
    "NI_cum": 8,
    "NC_inc": 9,
    "NC_cum": 9,
    "NIC_inc": 79,
    "NIC_cum": 79,
    "NIC_v2_79": 79,
    "NIC_v2_196": 196,
    "NIC_v2_391": 391,
}
DOCUMENTED_RUNS = 10
MAIN_SCENARIOS = tuple(
    f"{scenario}_{mode}" for scenario in ("NI", "NC", "NIC") for mode in ("inc", "cum")
)
SAMPLE_COUNT = 16

STEP_LABELS = (
    "Project / environment check",
    "Official CORe50 source inspection",
    "Downloading CORe50",
    "Extracting / organizing",
    "Validating dataset structure",
    "Validating sessions / objects / filelists",
    "Documentation + tests",
    "Final verification + git commit",
)


class ValidationError(RuntimeError):
    """Raised when the acquired dataset fails validation."""


def _human(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} GiB"


def load_official_dims(path: Path) -> tuple[dict[tuple[int, int], int], dict[str, int]]:
    """Parse the official Color128x128.tsv into (obj, session) -> count."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    sessions = [cell for cell in header[1:] if cell.startswith("s")]
    per_pair: dict[tuple[int, int], int] = {}
    per_session: dict[str, int] = {}
    for line in lines[1:]:
        cells = line.split("\t")
        if not cells or not cells[0]:
            continue
        if cells[0] == "totale":
            for session, value in zip(sessions, cells[1:]):
                per_session[session] = int(value)
            continue
        object_id = int(cells[0][1:])
        for session, value in zip(sessions, cells[1:]):
            per_pair[(object_id, int(session[1:]))] = int(value)
    return per_pair, per_session


def scan_structure(
    dataset_root: Path, progress: PhaseProgress
) -> dict[str, object]:
    errors: list[str] = []
    session_dirs = sorted(
        (d for d in dataset_root.iterdir() if d.is_dir()),
        key=lambda d: int(d.name[1:]),
    )
    if len(session_dirs) != EXPECTED_SESSIONS:
        errors.append(f"expected {EXPECTED_SESSIONS} sessions, found {len(session_dirs)}")
    if [d.name for d in session_dirs] != [f"s{i}" for i in SESSION_IDS]:
        errors.append("session directories are not exactly s1..s11")

    object_dirs_total = 0
    files: list[Path] = []
    counts: Counter[tuple[int, int]] = Counter()
    empty_files: list[str] = []
    bad_names: list[str] = []
    bad_ext: list[str] = []

    for session_dir in session_dirs:
        session_id = int(session_dir.name[1:])
        object_dirs = sorted(
            (d for d in session_dir.iterdir() if d.is_dir()),
            key=lambda d: int(d.name[1:]),
        )
        object_dirs_total += len(object_dirs)
        if [d.name for d in object_dirs] != [f"o{i}" for i in range(1, 51)]:
            errors.append(f"{session_dir.name}: does not contain exactly o1..o50")
        for object_dir in object_dirs:
            object_id = int(object_dir.name[1:])
            for path in object_dir.iterdir():
                if not path.is_file():
                    continue
                files.append(path)
                if path.suffix.lower() != ".png":
                    bad_ext.append(str(path.relative_to(dataset_root)))
                if path.stat().st_size == 0:
                    empty_files.append(str(path.relative_to(dataset_root)))
                match = FRAME_RE.match(path.name)
                if not match:
                    bad_names.append(str(path.relative_to(dataset_root)))
                    continue
                file_session, file_object = int(match.group(1)), int(match.group(2))
                if file_session != session_id or file_object != object_id:
                    bad_names.append(str(path.relative_to(dataset_root)))
                    continue
                counts[(object_id, session_id)] += 1

    total_files = len(files)
    if total_files != EXPECTED_IMAGES:
        errors.append(f"expected {EXPECTED_IMAGES} images, found {total_files}")
    if object_dirs_total != EXPECTED_SESSIONS * EXPECTED_OBJECTS:
        errors.append(
            f"expected {EXPECTED_SESSIONS * EXPECTED_OBJECTS} object dirs, "
            f"found {object_dirs_total}"
        )
    if empty_files:
        errors.append(f"{len(empty_files)} zero-byte files (first: {empty_files[:3]})")
    if bad_names:
        errors.append(
            f"{len(bad_names)} files with illegal names (first: {bad_names[:3]})"
        )
    if bad_ext:
        errors.append(f"{len(bad_ext)} non-png files (first: {bad_ext[:3]})")

    progress.log(
        f"[chk ] structure: {total_files:,} files, {object_dirs_total} object dirs, "
        f"{len(empty_files)} zero-byte, {len(bad_names)} bad names"
    )
    return {
        "errors": errors,
        "total_files": total_files,
        "session_dirs": [d.name for d in session_dirs],
        "object_dirs": object_dirs_total,
        "counts": counts,
        "empty_files": empty_files,
        "bad_names": bad_names,
        "relative_paths": {
            str(p.relative_to(dataset_root)).replace("\\", "/") for p in files
        },
    }


def check_png_signatures(files_root: Path, progress: PhaseProgress) -> list[str]:
    paths = sorted(files_root.rglob("*.png"))
    broken: list[str] = []
    started = time.monotonic()
    last = started
    for index, path in enumerate(paths, start=1):
        try:
            with path.open("rb") as handle:
                if handle.read(8) != PNG_MAGIC:
                    broken.append(str(path.relative_to(files_root)))
        except OSError as exc:
            broken.append(f"{path.relative_to(files_root)} ({exc})")
        now = time.monotonic()
        if now - last >= 0.3:
            last = now
            elapsed = max(now - started, 1e-6)
            rate = index / elapsed
            remaining = (len(paths) - index) / rate if rate > 0 else 0
            progress.update(
                index,
                len(paths),
                detail=(
                    f"PNG signature scan  ETA {format_duration(remaining)}  "
                    f"broken={len(broken)}"
                ),
            )
    progress.update(len(paths), len(paths), force=True)
    return broken


def representative_sample(dataset_root: Path, progress: PhaseProgress) -> dict[str, object]:
    """Decode a deterministic sample spanning sessions/objects/categories."""
    picked: list[Path] = []
    for session_id in SESSION_IDS:
        session_dir = dataset_root / f"s{session_id}"
        object_ids = list(range(1, EXPECTED_OBJECTS + 1))
        # 5 objects per session, rotating so all 50 objects appear across sessions
        stride_ids = object_ids[(session_id - 1) * 5 % 50 :: 7]
        for object_id in stride_ids[:5]:
            object_dir = session_dir / f"o{object_id}"
            frames = sorted(object_dir.glob("C_*.png"))
            if frames:
                picked.append(frames[len(frames) // 2])
    decoded = 0
    failures: list[str] = []
    started = time.monotonic()
    for index, path in enumerate(picked, start=1):
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (128, 128):
                    failures.append(f"{path.name}: size {image.size}")
                decoded += 1
        except Exception as exc:  # noqa: BLE001 - report any decode failure
            failures.append(f"{path.name}: {exc}")
        if index % 10 == 0:
            progress.update(
                index,
                len(picked),
                detail=f"representative decode  elapsed "
                f"{format_duration(time.monotonic() - started)}",
            )
    return {
        "picked": len(picked),
        "decoded": decoded,
        "failures": failures,
        "sessions_covered": len({p.parents[1].name for p in picked}),
        "objects_covered": len({p.parent.name for p in picked}),
        "categories_covered": len(
            {OBJECT_CATEGORY[int(p.parent.name[1:])] for p in picked}
        ),
    }


def validate_official_paths(dataset_root: Path, structure: dict[str, object]) -> dict[str, object]:
    paths_pkl = METADATA_DIR / "core50-official" / "extras" / "paths.pkl"
    official = pickle.loads(paths_pkl.read_bytes())
    official_set = set(official)
    disk_set: set[str] = structure["relative_paths"]  # type: ignore[assignment]
    missing = official_set - disk_set
    extra = disk_set - official_set
    return {
        "official_paths": len(official),
        "official_unique": len(official_set),
        "missing_on_disk": sorted(missing)[:10],
        "extra_on_disk": sorted(extra)[:10],
        "identical": not missing and not extra,
    }


def parse_filelist(path: Path) -> tuple[list[tuple[str, int]], int]:
    entries: list[tuple[str, int]] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                bad += 1
                continue
            entries.append((parts[0].replace("\\", "/"), int(parts[1])))
    return entries, bad


def validate_filelists(
    progress: PhaseProgress, universe: set[str]
) -> dict[str, object]:
    errors: list[str] = []
    scenario_stats: dict[str, object] = {}
    all_paths: set[str] = set()
    total_lines = 0
    total_bad = 0

    scenarios = sorted(
        d.name for d in FILELIST_DIR.iterdir() if d.is_dir()
    ) if FILELIST_DIR.is_dir() else []
    missing_scenarios = [s for s in MAIN_SCENARIOS if s not in scenarios]
    if missing_scenarios:
        errors.append(f"missing official scenarios: {missing_scenarios}")

    scenario_dirs = [
        d for d in FILELIST_DIR.iterdir() if d.is_dir()
    ]
    started = time.monotonic()
    for scenario_dir in scenario_dirs:
        runs = sorted(d for d in scenario_dir.iterdir() if d.is_dir())
        run_stats: dict[str, object] = {}
        test_hashes: dict[str, str] = {}
        unresolved = 0
        lines = 0
        bad = 0
        for run_dir in runs:
            filelists = sorted(run_dir.glob("*.txt"))
            train_lists = sorted(
                p for p in filelists if p.name.startswith("train_batch_")
            )
            test_lists = [p for p in filelists if p.name.startswith("test_")]
            for filelist in train_lists + test_lists:
                entries, file_bad = parse_filelist(filelist)
                bad += file_bad
                lines += len(entries)
                for rel_path, _label in entries:
                    all_paths.add(rel_path)
                    if rel_path not in universe:
                        unresolved += 1
                if filelist.name.startswith("test_"):
                    digest = hashlib.sha1(
                        "\n".join(sorted(rel for rel, _ in entries)).encode()
                    ).hexdigest()
                    test_hashes[run_dir.name] = digest
            run_stats[run_dir.name] = {
                "train_batches": len(train_lists),
                "test_filelists": len(test_lists),
            }
        total_lines += lines
        total_bad += bad
        if unresolved:
            errors.append(
                f"{scenario_dir.name}: {unresolved} filelist paths do not resolve"
            )
        expected_batches = EXPECTED_BATCHES.get(scenario_dir.name)
        if expected_batches is not None:
            for run_name, stats in run_stats.items():
                if stats["train_batches"] != expected_batches:  # type: ignore[index]
                    errors.append(
                        f"{scenario_dir.name}/{run_name}: "
                        f"{stats['train_batches']} train batches "
                        f"(official: {expected_batches})"
                    )
                if stats["test_filelists"] != 1:  # type: ignore[index]
                    errors.append(
                        f"{scenario_dir.name}/{run_name}: "
                        f"{stats['test_filelists']} test filelists (expected 1)"
                    )
            for index, run in enumerate(runs):
                if run.name != f"run{index}":
                    errors.append(
                        f"{scenario_dir.name}: runs are not contiguous run0..runN "
                        f"(found {run.name} at index {index})"
                    )
        # Official documentation describes 10 runs per scenario; the published
        # batches_filelists.zip ships fewer for some cumulative variants.
        # Record the difference instead of failing on an official resource.
        runs_match_documented = (
            len(runs) == DOCUMENTED_RUNS
            if scenario_dir.name in MAIN_SCENARIOS
            else None
        )
        scenario_stats[scenario_dir.name] = {
            "runs": len(runs),
            "run_names": [r.name for r in runs],
            "documented_runs": (
                DOCUMENTED_RUNS if scenario_dir.name in MAIN_SCENARIOS else None
            ),
            "runs_match_documentation": runs_match_documented,
            "filelists": sum(
                len(list(r.glob('*.txt'))) for r in runs
            ),
            "lines": lines,
            "unresolved_paths": unresolved,
            "train_batches_per_run": expected_batches,
            "test_set_fixed_across_runs": len(
                set(test_hashes.values())
            ) <= 1,
        }
        note = ""
        if runs_match_documented is False:
            note = f" [official archive ships {len(runs)} of {DOCUMENTED_RUNS} runs]"
        progress.log(
            f"[chk ] {scenario_dir.name}: {len(runs)} runs, {lines:,} lines, "
            f"{unresolved} unresolved{note}"
        )

    if total_bad:
        errors.append(f"{total_bad} malformed filelist lines")
    return {
        "errors": errors,
        "scenarios": scenario_stats,
        "total_lines": total_lines,
        "unique_referenced_paths": len(all_paths),
        "malformed_lines": total_bad,
        "elapsed": time.monotonic() - started,
    }


def load_object_names() -> list[str]:
    names_path = METADATA_DIR / "core50-official" / "extras" / "core50_labels.txt"
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    if len(names) != EXPECTED_OBJECTS:
        raise ValidationError(
            f"official object-name list has {len(names)} entries, expected 50"
        )
    return names


def build_object_mapping(
    counts: dict[tuple[int, int], int]
) -> dict[str, object]:
    names = load_object_names()
    objects = []
    for object_id in range(1, EXPECTED_OBJECTS + 1):
        per_session = {
            f"s{session_id}": counts.get((object_id, session_id), 0)
            for session_id in SESSION_IDS
        }
        objects.append(
            {
                "object_id": object_id,
                "directory": f"o{object_id}",
                "name": names[object_id - 1],
                "category": OBJECT_CATEGORY[object_id],
                "images": sum(per_session.values()),
                "images_per_session": per_session,
                "available_in_sessions": [
                    f"s{s}" for s in SESSION_IDS if per_session[f"s{s}"] > 0
                ],
            }
        )
    return {
        "source_object_names": "official repo extras/core50_labels.txt",
        "source_categories": OFFICIAL_PAGE,
        "category_order": list(OFFICIAL_CATEGORY_ORDER),
        "objects": objects,
        "categories": sorted({o["category"] for o in objects}),
    }


def validate_bbox(metadata_root: Path) -> dict[str, object]:
    bbox_root = metadata_root / "bbox"
    files = sorted(bbox_root.rglob("CropC_o*.txt"))
    parsed = 0
    bad: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not text:
            bad.append(f"{path.relative_to(bbox_root)}: empty")
            continue
        ok = True
        for line in text[:5]:
            parts = line.split(":")
            if len(parts) != 2 or len(parts[1].split()) != 4:
                ok = False
                break
        if not ok:
            bad.append(f"{path.relative_to(bbox_root)}: unparsable header")
        else:
            parsed += 1
    return {
        "files": len(files),
        "parsed": parsed,
        "bad": bad[:5],
        "expected": EXPECTED_SESSIONS * EXPECTED_OBJECTS,
    }


def write_samples(
    dataset_root: Path, destination: Path, progress: PhaseProgress
) -> dict[str, object]:
    """Copy a small deterministic cross-section of images for inspection."""
    # Deterministic selection covering every session and every category:
    #  pick i -> session SESSION_IDS[i % 11], category i % 10,
    #            object instance 1 (i < 10) else instance 2 of that category.
    selected: list[tuple[int, int]] = []
    category_count = len(OFFICIAL_CATEGORY_ORDER)
    for index in range(SAMPLE_COUNT):
        session_id = SESSION_IDS[index % EXPECTED_SESSIONS]
        instance = 1 if index < category_count else 2
        object_id = (index % category_count) * 5 + instance
        selected.append((session_id, object_id))

    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*.png"):
        stale.unlink()
    names = load_object_names()
    manifest = []
    for session_id, object_id in selected:
        object_dir = dataset_root / f"s{session_id}" / f"o{object_id}"
        frames = sorted(object_dir.glob("C_*.png"))
        if not frames:
            continue
        source = frames[len(frames) // 2]
        target = destination / f"s{session_id}_o{object_id}_{source.name}"
        shutil.copyfile(source, target)
        manifest.append(
            {
                "file": target.name,
                "session": f"s{session_id}",
                "object": f"o{object_id}",
                "object_name": names[object_id - 1],
                "category": OBJECT_CATEGORY[object_id],
                "split": "test" if session_id in TEST_SESSIONS else "train",
            }
        )
        progress.update(
            len(manifest), len(selected), detail="copying inspection samples"
        )
    return {
        "count": len(manifest),
        "sessions": sorted({m["session"] for m in manifest}),
        "objects": len({m["object"] for m in manifest}),
        "categories": len({m["category"] for m in manifest}),
        "images": manifest,
    }


def read_dims(datasets_root: Path) -> dict[str, object]:
    dims_path = datasets_root / "dataset_dims" / "Color128x128.tsv"
    per_pair, per_session = load_official_dims(dims_path)
    return {
        "per_pair": per_pair,
        "per_session": per_session,
        "grand_total": sum(per_pair.values()),
    }


def run(*, with_samples: bool = True) -> int:
    progress = PhaseProgress("PHASE 2 OVERALL", STEP_LABELS)
    dataset_root = DATASET_DIR / "core50_128x128"
    failures: list[str] = []

    if not dataset_root.is_dir():
        raise ValidationError(f"dataset root missing: {dataset_root}")

    # ---------------- Step 5: structure ----------------
    progress.set_step(5, "Validating dataset structure")
    phase_start = time.monotonic()
    structure = scan_structure(dataset_root, progress)
    progress.log(f"[time] structure scan: {time.monotonic() - phase_start:.1f}s")

    dims = read_dims(METADATA_DIR)
    counts: dict[tuple[int, int], int] = structure["counts"]  # type: ignore[assignment]
    mismatches = []
    for (object_id, session_id), expected in dims["per_pair"].items():  # type: ignore[union-attr]
        observed = counts.get((object_id, session_id), 0)
        if observed != expected:
            mismatches.append(
                f"o{object_id}/s{session_id}: {observed} != official {expected}"
            )
    if mismatches:
        structure["errors"].append(  # type: ignore[union-attr]
            f"{len(mismatches)} count mismatches vs official dims (first: {mismatches[0]})"
        )
    if dims["grand_total"] != EXPECTED_IMAGES:  # type: ignore[union-attr]
        structure["errors"].append(  # type: ignore[union-attr]
            f"official dims table totals {dims['grand_total']}, expected {EXPECTED_IMAGES}"
        )
    progress.log(
        f"[chk ] official dims cross-check: {len(dims['per_pair'])} object/session "  # type: ignore[union-attr]
        f"cells, {len(mismatches)} mismatches"
    )

    broken = check_png_signatures(dataset_root, progress)
    progress.log(
        f"[time] PNG signature scan: {len(broken)} failures, "
        f"step 5 total {time.monotonic() - phase_start:.1f}s"
    )
    if broken:
        structure["errors"].append(  # type: ignore[union-attr]
            f"{len(broken)} PNG signature failures (first: {broken[:3]})"
        )

    failures.extend(structure["errors"])  # type: ignore[arg-type]

    # ---------------- Step 6: sessions / objects / filelists ----------------
    progress.set_step(6, "Validating sessions / objects / filelists")
    step6_start = time.monotonic()
    paths_check = validate_official_paths(dataset_root, structure)
    progress.log(
        f"[time] official paths.pkl cross-check: {time.monotonic() - step6_start:.1f}s"
    )
    if not paths_check["identical"]:
        failures.append(
            "official paths.pkl does not match disk contents: "
            f"missing={len(paths_check['missing_on_disk'])}, "
            f"extra={len(paths_check['extra_on_disk'])}"
        )

    sample_check = representative_sample(dataset_root, progress)
    if sample_check["failures"]:
        failures.append(f"representative decode failures: {sample_check['failures'][:3]}")

    filelist_check = validate_filelists(progress, structure['relative_paths'])
    progress.log(
        f"[time] filelist scan: {filelist_check['elapsed']:.1f}s for "  # type: ignore[union-attr]
        f"{filelist_check['total_lines']:,} lines; step 6 so far "  # type: ignore[union-attr]
        f"{time.monotonic() - step6_start:.1f}s"
    )
    failures.extend(filelist_check["errors"])  # type: ignore[arg-type]

    bbox_check = validate_bbox(METADATA_DIR)
    if bbox_check["files"] != bbox_check["expected"] or bbox_check["bad"]:
        failures.append(
            f"bbox metadata: {bbox_check['files']} files "
            f"(expected {bbox_check['expected']}), bad={bbox_check['bad'][:3]}"
        )

    mapping = build_object_mapping(counts)
    mapping_path = METADATA_DIR / "object_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    progress.log(f"[ ok ] object mapping written: {mapping_path.relative_to(ROOT)}")

    samples: dict[str, object] = {"count": 0}
    if with_samples:
        samples = write_samples(
            dataset_root, ROOT / "reports" / "phase2_core50_samples", progress
        )
        if samples["count"] != SAMPLE_COUNT:  # type: ignore[index]
            failures.append(f"expected {SAMPLE_COUNT} samples, wrote {samples['count']}")
        if len(samples["sessions"]) != EXPECTED_SESSIONS:  # type: ignore[arg-type]
            failures.append(
                f"samples cover {len(samples['sessions'])} of {EXPECTED_SESSIONS} sessions"  # type: ignore[arg-type]
            )
        if samples["categories"] != len(OFFICIAL_CATEGORY_ORDER):  # type: ignore[index]
            failures.append(f"samples cover {samples['categories']} categories")  # type: ignore[index]

    # ---------------- Summary ----------------
    per_session_counts = {
        f"s{session_id}": sum(
            counts.get((object_id, session_id), 0)
            for object_id in range(1, EXPECTED_OBJECTS + 1)
        )
        for session_id in SESSION_IDS
    }
    per_category_counts = Counter()
    per_object_counts = Counter()
    for (object_id, _session), count in counts.items():
        per_object_counts[object_id] += count
        per_category_counts[OBJECT_CATEGORY[object_id]] += count

    manifest = json.loads(
        (DOWNLOAD_DIR / "MANIFEST.json").read_text(encoding="utf-8")
    )
    summary = {
        "phase": 2,
        "dataset": "CORe50",
        "dataset_root": str(dataset_root.relative_to(ROOT)).replace("\\", "/"),
        "official_source": OFFICIAL_PAGE,
        "repository": OFFICIAL_REPO,
        "sessions": [f"s{i}" for i in SESSION_IDS],
        "train_sessions": [f"s{i}" for i in TRAIN_SESSIONS],
        "test_sessions": [f"s{i}" for i in TEST_SESSIONS],
        "session_count": len(structure["session_dirs"]),  # type: ignore[arg-type]
        "object_identities": [
            {"object_id": o["object_id"], "name": o["name"], "category": o["category"]}
            for o in mapping["objects"]  # type: ignore[index]
        ],
        "categories": mapping["categories"],  # type: ignore[arg-type]
        "object_count": len(mapping["objects"]),  # type: ignore[arg-type]
        "category_count": len(mapping["categories"]),  # type: ignore[arg-type]
        "total_files": structure["total_files"],  # type: ignore[arg-type]
        "total_images": structure["total_files"],  # type: ignore[arg-type]
        "images_per_session": per_session_counts,
        "images_per_category": dict(sorted(per_category_counts.items())),
        "images_per_object": {
            f"o{oid}": per_object_counts[oid]
            for oid in sorted(per_object_counts)
        },
        "official_dims_crosscheck": {
            "source": "metadata/dataset_dims/Color128x128.tsv",
            "cells": len(dims["per_pair"]),  # type: ignore[arg-type]
            "mismatches": len(mismatches),
        },
        "filelists": {
            name: stats for name, stats in filelist_check["scenarios"].items()  # type: ignore[union-attr]
        },
        "filelist_total_lines": filelist_check["total_lines"],  # type: ignore[union-attr]
        "filelist_unique_paths": filelist_check["unique_referenced_paths"],  # type: ignore[union-attr]
        "filelist_unresolved_paths": sum(
            s["unresolved_paths"] for s in filelist_check["scenarios"].values()  # type: ignore[union-attr, union-attr]
        ),
        "scenarios": {
            "NI": "New Instances",
            "NC": "New Classes",
            "NIC": "New Instances + Classes",
        },
        "integrity": {
            "zero_byte_files": len(structure["empty_files"]),  # type: ignore[arg-type]
            "bad_file_names": len(structure["bad_names"]),  # type: ignore[arg-type]
            "png_signature_checked": structure["total_files"],  # type: ignore[arg-type]
            "png_signature_failures": len(broken),
            "representative_decode": {
                "images": sample_check["decoded"],
                "sessions": sample_check["sessions_covered"],
                "objects": sample_check["objects_covered"],
                "categories": sample_check["categories_covered"],
                "failures": len(sample_check["failures"]),
            },
            "official_paths_match_disk": paths_check["identical"],
            "bbox": bbox_check,
        },
        "acquisition": {
            "resources": len(manifest["resources"]),
            "bytes": sum(r["size_bytes"] for r in manifest["resources"]),
            "acquired_utc": manifest["acquired_utc"],
            "manifest": "data/raw/core50/downloads/MANIFEST.json",
        },
        "samples": samples,
        "license": "CC BY 4.0",
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary_path = ROOT / "reports" / "phase2_core50_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    progress.update(EXPECTED_IMAGES, EXPECTED_IMAGES, force=True)
    if failures:
        progress.finish("VALIDATION FAILED")
        for failure in failures:
            print(f"  FAIL: {failure}")
        return 1
    progress.finish(
        f"Validation PASS: {structure['total_files']:,} images, "  # type: ignore[arg-type]
        f"{len(mapping['objects'])} objects, "  # type: ignore[arg-type]
        f"{filelist_check['total_lines']:,} filelist lines resolved"  # type: ignore[union-attr]
    )
    print(f"Summary: {summary_path.relative_to(ROOT).as_posix()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="skip writing the human-inspection sample images",
    )
    args = parser.parse_args(argv)
    try:
        return run(with_samples=not args.no_samples)
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
