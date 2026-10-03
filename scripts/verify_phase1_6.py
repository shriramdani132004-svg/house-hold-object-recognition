"""Phase 1-6 repair verification suite: 19 semantic checks in 10 stages.

Read-only by design: it never retrains, never edits reports, datasets,
weights, documentation or the roadmap, and never scans all 164,866 images
(that full scan is the explicit optional command below).

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\verify_phase1_6.py
    .\\.venv\\Scripts\\python.exe scripts\\verify_phase1_6.py --full-scan

``--full-scan`` additionally runs the optional full-dataset validation
(``scripts/validate_core50.py --no-samples``) with live progress; the
tracked summary artifact it refreshes is restored afterwards so the working
tree stays untouched.

Exit status 0 only when every check passes::

    PHASE 1–6 VERIFICATION: PASS
    FAILURES: 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.progress import PhaseProgress  # noqa: E402

CheckFn = Callable[[], tuple[bool, str]]

LOCKED_FINAL_SHA256 = (
    "b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351"
)
LOCKED_NAIVE_FINAL = 0.0234590411811794
LOCKED_NAIVE_FORGETTING = 0.620896
LOCKED_NAIVE_AVG_INCR = 0.045834
LOCKED_REPLAY_FINAL = 0.05378902428177533
LOCKED_REPLAY_FORGETTING = 0.536408
LOCKED_REPLAY_AVG_INCR = 0.063249
LOCKED_BASELINE_MAP50 = 0.548459
EXPECTED_IMAGES = 164_866

REQUIRED_COMMITS: tuple[str, ...] = (
    "Set up project foundation: structure, docs, venv, tests",
    "Implement Phase 2 CORe50 dataset",
    "Implement Phase 3 continual data pipeline",
    "Implement Phase 4 naive and replay continual learning",
    "Implement Phase 5 NIC continual experiment and evaluation",
    "Implement Phase 6 error analysis and final model selection",
    "Repair and certify Phase 1-6 project integrity",
)

WEIGHT_SUFFIXES = (".pt", ".pth", ".onnx", ".engine", ".ckpt")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {proc.stderr.strip()[:200]}"
        )
    return proc.stdout


def _load_json(relative: str):
    path = PROJECT_ROOT / relative
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _problem_list(problems: list[str]) -> str:
    shown = "; ".join(problems[:6])
    if len(problems) > 6:
        shown += f"; … +{len(problems) - 6} more"
    return shown


# ---------------------------------------------------------------------------
# stage 1 - environment
# ---------------------------------------------------------------------------


def check_environment() -> tuple[bool, str]:
    problems: list[str] = []
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 14):
        problems.append(f"Python {version} < 3.14")
    torch_version = "missing"
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        problems.append("torch is not importable")
    venv = PROJECT_ROOT / ".venv"
    if Path(sys.prefix).resolve() != venv.resolve():
        problems.append("interpreter is not the project .venv")
    if problems:
        return False, _problem_list(problems)
    return True, f"Python {version}, torch {torch_version} (CPU), .venv active"


# ---------------------------------------------------------------------------
# stage 2 - git repository + hygiene
# ---------------------------------------------------------------------------


def check_git_repository() -> tuple[bool, str]:
    problems: list[str] = []
    try:
        if _git("rev-parse", "--is-inside-work-tree").strip() != "true":
            problems.append("not inside a git work tree")
        head = _git("rev-parse", "--short", "HEAD").strip()
        branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    except RuntimeError as exc:
        return False, str(exc)
    if branch != "main":
        problems.append(f"branch {branch} != main")
    for subject in REQUIRED_COMMITS:
        found = _git(
            "log", "--oneline", "--fixed-strings", f"--grep={subject}", "-n", "1"
        )
        if not found.strip():
            problems.append(f"missing commit: {subject!r}")
    if problems:
        return False, _problem_list(problems)
    return True, f"HEAD {head} on main; {len(REQUIRED_COMMITS)} phase/repair commits in history"


def check_repository_hygiene() -> tuple[bool, str]:
    tracked = [
        line.replace("\\", "/")
        for line in _git("ls-files").splitlines()
        if line.strip()
    ]
    problems: list[str] = []
    allowed_data = {"data/splits/.gitkeep"}
    allowed_models = {"models/.gitkeep", "models/baseline/README.md"}
    for path in tracked:
        basename = path.rsplit("/", 1)[-1]
        stem = basename.split(".", 1)[0].lower()
        if path.startswith("data/") and path not in allowed_data:
            problems.append(f"dataset file tracked: {path}")
        if path.startswith("models/") and path not in allowed_models:
            problems.append(f"model file tracked: {path}")
        if path.endswith(WEIGHT_SUFFIXES):
            problems.append(f"weights tracked: {path}")
        if path.startswith("reports/") and path.lower().endswith(IMAGE_SUFFIXES):
            problems.append(f"generated image tracked: {path}")
        if (
            "__pycache__" in path
            or path.startswith(".venv/")
            or path.startswith("logs/")
            or path.startswith("runs/")
            or path.endswith(".log")
        ):
            problems.append(f"environment/cache file tracked: {path}")
        if path.startswith(".env") or basename.lower().endswith(
            (".key", ".pem")
        ) or stem in ("secrets", "credentials"):
            problems.append(f"secret-looking file tracked: {path}")
    if "phases map.txt" not in tracked:
        problems.append("canonical roadmap (phases map.txt) is not tracked")
    status = _git("status", "--porcelain")
    if status.strip():
        first = status.strip().splitlines()[:4]
        problems.append(f"working tree not clean: {first}")
    if problems:
        return False, _problem_list(problems)
    return True, (
        f"{len(tracked)} tracked files; no datasets/weights/binaries/secrets; "
        "working tree clean"
    )


# ---------------------------------------------------------------------------
# stage 3 - CORe50 presence + official metadata
# ---------------------------------------------------------------------------


def check_core50_dataset_presence() -> tuple[bool, str]:
    from src.data.core50 import DATASET_DIR, OBJECT_IDS, SESSION_IDS, session_name

    root = DATASET_DIR / "core50_128x128"
    if not root.is_dir():
        return False, f"dataset root missing: data/raw/core50/dataset/core50_128x128"
    problems: list[str] = []
    for session in SESSION_IDS:
        session_dir = root / session_name(session)
        if not session_dir.is_dir():
            problems.append(f"missing {session_name(session)}")
            continue
        objects = [entry.name for entry in session_dir.iterdir() if entry.is_dir()]
        if len(objects) != len(OBJECT_IDS):
            problems.append(
                f"{session_name(session)} has {len(objects)} object dirs "
                f"(expected {len(OBJECT_IDS)})"
            )
    if problems:
        return False, _problem_list(problems)
    return True, "11 sessions × 50 object directories present (stat-only, no image scan)"


def check_core50_official_metadata() -> tuple[bool, str]:
    from src.data.core50 import DOWNLOAD_DIR, METADATA_DIR

    problems: list[str] = []
    mapping = METADATA_DIR / "object_mapping.json"
    if not mapping.is_file():
        problems.append("object_mapping.json missing")
    else:
        try:
            json.loads(mapping.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            problems.append("object_mapping.json is not valid JSON")
    for subdir in ("dataset_dims", "bbox", "core50-official"):
        folder = METADATA_DIR / subdir
        if not folder.is_dir() or not any(folder.iterdir()):
            problems.append(f"metadata/{subdir} missing or empty")
    manifest = DOWNLOAD_DIR / "MANIFEST.json"
    entries = 0
    if not manifest.is_file():
        problems.append("downloads/MANIFEST.json missing")
    else:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            entries = len(payload) if isinstance(payload, (dict, list)) else 0
            if entries == 0:
                problems.append("downloads/MANIFEST.json has no entries")
        except json.JSONDecodeError:
            problems.append("downloads/MANIFEST.json is not valid JSON")
    if problems:
        return False, _problem_list(problems)
    return True, f"object mapping + dims/bbox/official-repo metadata; MANIFEST {entries} entries"


# ---------------------------------------------------------------------------
# stage 4 - object mapping semantics
# ---------------------------------------------------------------------------


def check_object_mapping_semantics() -> tuple[bool, str]:
    from src.data.core50 import OBJECT_MAPPING_PATH, validate_core50_object_mapping
    from src.evaluation.continual import load_class_names

    result = validate_core50_object_mapping()
    problems = list(result["errors"])
    if result["identity_count"] != 50:
        problems.append(f"identity_count {result['identity_count']} != 50")
    if result["category_count"] != 10:
        problems.append(f"category_count {result['category_count']} != 10")
    class_names = load_class_names(OBJECT_MAPPING_PATH)
    if len(class_names) != 50:
        problems.append(f"load_class_names resolved {len(class_names)} labels != 50")
    else:
        for item in result["identities"]:
            if class_names.get(str(item["label"])) != item["name"]:
                problems.append(
                    f"label {item['label']} disagrees with load_class_names"
                )
                break
    if problems:
        return False, _problem_list(problems)
    return True, (
        "50 unique identities × 10 categories; labels 0-49; directories/sessions "
        "verified; load_class_names agrees"
    )


# ---------------------------------------------------------------------------
# stage 5 - Phase-3 pipeline
# ---------------------------------------------------------------------------


def check_continual_pipeline_manifest() -> tuple[bool, str]:
    manifest = _load_json("reports/phase3_continual_pipeline_manifest.json")
    problems: list[str] = []
    if manifest.get("supported_scenarios") != ["NI", "NC", "NIC"]:
        problems.append(f"supported_scenarios {manifest.get('supported_scenarios')}")
    if manifest.get("image_copies_created") != 0:
        problems.append("image_copies_created != 0")
    if manifest.get("dataset_modified") is not False:
        problems.append("dataset_modified is not false")
    run_selection = manifest.get("run_selection") or {}
    if run_selection.get("all_checks_passed") is not True:
        problems.append("run_selection.all_checks_passed is not true")
    scenarios = manifest.get("scenarios") or []
    if not scenarios:
        problems.append("no scenarios recorded")
    checks_total = 0
    checks_passed = 0
    for scenario in scenarios:
        name = scenario.get("name", "?")
        for check in scenario.get("leakage_check_results") or []:
            checks_total += 1
            if check.get("passed") is True:
                checks_passed += 1
            else:
                problems.append(f"{name}: {check.get('name')} not passed")
        if scenario.get("unresolved_references"):
            problems.append(f"{name}: unresolved filelist references")
    if checks_total == 0:
        problems.append("no leakage checks recorded")
    if problems:
        return False, _problem_list(problems)
    return True, (
        f"{len(scenarios)} scenario variants; {checks_passed}/{checks_total} leakage "
        "checks passed; 0 image copies; dataset untouched"
    )


def check_ni_nc_nic_support() -> tuple[bool, str]:
    from src.data.continual import list_scenarios
    from src.data.core50 import FILELIST_DIR

    problems: list[str] = []
    expected_batches = {"NI_inc": 8, "NC_inc": 9, "NIC_inc": 79}
    for variant, batches in expected_batches.items():
        run_dir = FILELIST_DIR / variant / "run0"
        if not run_dir.is_dir():
            problems.append(f"filelists/{variant}/run0 missing")
            continue
        train = len(list(run_dir.glob("train_batch_*_filelist.txt")))
        if train != batches:
            problems.append(f"{variant}: {train} train batches != {batches}")
        if not (run_dir / "test_filelist.txt").is_file():
            problems.append(f"{variant}: test_filelist.txt missing")
    discovered = list_scenarios()
    names = sorted(
        {getattr(entry, "scenario_type", str(entry)) for entry in discovered}
    )
    variants = sorted(getattr(entry, "name", str(entry)) for entry in discovered)
    for scenario in ("NI", "NC", "NIC"):
        if scenario not in names:
            problems.append(f"loader cannot discover scenario {scenario}")
    if problems:
        return False, _problem_list(problems)
    return True, (
        "NI 8 / NC 9 / NIC 79 batches + test in official run0; loader discovers "
        f"{len(variants)} variants: {', '.join(variants)}"
    )


# ---------------------------------------------------------------------------
# stage 6 - continual learning APIs
# ---------------------------------------------------------------------------


def check_naive_continual_api() -> tuple[bool, str]:
    from src.training import (
        NaiveContinual,
        NaiveContinualTrainer,
        ReplayContinualTrainer,
        build_continual_trainer,
        supported_methods,
    )

    problems: list[str] = []
    if not issubclass(NaiveContinual, NaiveContinualTrainer):
        problems.append("NaiveContinual does not derive from NaiveContinualTrainer")
    if NaiveContinual.method_name != "naive":
        problems.append("NaiveContinual.method_name != 'naive'")
    if "naive" not in supported_methods():
        problems.append("'naive' missing from supported_methods()")
    trainer = build_continual_trainer("naive")
    if not isinstance(trainer, NaiveContinualTrainer) or isinstance(
        trainer, ReplayContinualTrainer
    ):
        problems.append("factory('naive') did not return the naive trainer")
    if problems:
        return False, _problem_list(problems)
    return True, "NaiveContinual -> NaiveContinualTrainer; factory returns naive trainer (no replay)"


def check_replay_memory_contract() -> tuple[bool, str]:
    from src.data.continual import SampleRecord
    from src.training import ReplayMemory, ReplayMemoryError

    def record(index: int, *, split: str = "train", experience_id: int = 0):
        return SampleRecord(
            relative_path=f"s1/o1/C_00_{index:03d}.png",
            label=0,
            split=split,
            experience_id=experience_id,
            source_filelist="synthetic/train_batch_00_filelist.txt",
            line_number=index + 1,
            object_id=1,
            session_id=1,
            category_id=0,
            category_name="synthetic",
            object_name="synthetic_object_1",
        )

    problems: list[str] = []
    memory = ReplayMemory(capacity=3, seed=11)
    memory.add([record(i) for i in range(5)], experience_index=0)
    if memory.size != 3:
        problems.append(f"capacity bound violated: size {memory.size} != 3")
    kept = [item.relative_path for item in memory.records()]
    if kept != ["s1/o1/C_00_002.png", "s1/o1/C_00_003.png", "s1/o1/C_00_004.png"]:
        problems.append("FIFO eviction order incorrect")
    for bad_record, fragment in (
        (record(99, split="test"), "cannot enter replay memory"),
        (record(50, experience_id=2), "Future-experience sample"),
    ):
        try:
            memory.add([bad_record], experience_index=1)
            problems.append(f"accepted forbidden record {bad_record.relative_path}")
        except ReplayMemoryError as exc:
            if fragment not in str(exc):
                problems.append(f"wrong rejection reason: {exc}")
    try:
        memory.add([record(60)], experience_index=5)
        problems.append("accepted out-of-order experience")
    except ReplayMemoryError:
        pass
    if memory.sample(2, seed=5) != memory.sample(2, seed=5):
        problems.append("seeded sampling is not deterministic")
    if problems:
        return False, _problem_list(problems)
    return True, "bounded FIFO (cap 3), train-only, official order, deterministic sampling"


# ---------------------------------------------------------------------------
# stage 7 - Phase 5 / Phase 6 artifacts
# ---------------------------------------------------------------------------


def check_phase5_results() -> tuple[bool, str]:
    naive = _load_json("reports/phase5_nic/naive_metrics.json")
    replay = _load_json("reports/phase5_nic/replay_metrics.json")
    summary = _load_json("reports/phase5_nic/experiment_summary.json")
    problems: list[str] = []
    for payload, label in ((naive, "naive"), (replay, "replay")):
        if payload.get("scenario") != "NIC_inc" or payload.get("run") != 0:
            problems.append(f"{label}: wrong scenario/run")
        records = payload.get("records") or []
        if len(records) != 79:
            problems.append(f"{label}: {len(records)} records != 79")
        elif [row.get("experience_id") for row in records] != list(range(79)):
            problems.append(f"{label}: experience ids not 0..78 in order")
    naive_final = (naive.get("records") or [{}])[-1]
    replay_final = (replay.get("records") or [{}])[-1]
    if naive_final.get("accuracy", {}).get("overall") != LOCKED_NAIVE_FINAL:
        problems.append("naive final accuracy changed")
    if naive_final.get("forgetting") != LOCKED_NAIVE_FORGETTING:
        problems.append("naive final forgetting changed")
    if replay_final.get("accuracy", {}).get("overall") != LOCKED_REPLAY_FINAL:
        problems.append("replay final accuracy changed")
    if replay_final.get("forgetting") != LOCKED_REPLAY_FORGETTING:
        problems.append("replay final forgetting changed")
    if summary.get("status") != "complete" or summary.get("num_experiences") != 79:
        problems.append("experiment_summary status/num_experiences wrong")
    if summary.get("seed") != 42:
        problems.append("experiment_summary seed != 42")
    integrity = summary.get("integrity") or {}
    bad = [
        name
        for name, entry in integrity.items()
        if isinstance(entry, dict) and entry.get("status") != "PASS"
    ]
    if bad:
        problems.append(f"integrity not PASS: {bad}")
    if problems:
        return False, _problem_list(problems)
    return True, (
        f"79 experiences × 2 methods; final acc {LOCKED_NAIVE_FINAL}/"
        f"{LOCKED_REPLAY_FINAL}; forgetting {LOCKED_NAIVE_FORGETTING}/"
        f"{LOCKED_REPLAY_FORGETTING}; integrity {len(integrity)}/{len(integrity)} PASS"
    )


def check_phase6_results() -> tuple[bool, str]:
    analysis = _load_json("reports/phase6_analysis/phase6_analysis.json")
    errors = _load_json("reports/phase6_analysis/representative_errors.json")
    selection = _load_json("reports/phase6_analysis/final_model_selection.json")
    environment = _load_json("reports/phase6_analysis/environment_analysis.json")
    problems: list[str] = []
    if analysis.get("status") != "complete":
        problems.append(f"analysis status {analysis.get('status')!r} != complete")
    if analysis.get("selected_method") != "replay":
        problems.append("selected_method != replay")
    integrity = analysis.get("integrity") or {}
    bad = [
        name
        for name, entry in integrity.items()
        if isinstance(entry, dict) and entry.get("status") not in ("PASS", "PRESERVED")
    ]
    if bad:
        problems.append(f"integrity not PASS/PRESERVED: {bad}")
    if analysis.get("representative_errors") != 12:
        problems.append("representative_errors != 12")
    load_test = analysis.get("load_test") or {}
    if not (
        load_test.get("checkpoint_loaded")
        and load_test.get("strict_state_dict")
        and load_test.get("num_classes") == 50
    ):
        problems.append("load_test does not record a strict 50-class load")
    if errors.get("n_examples") != 12 or len(errors.get("examples") or []) != 12:
        problems.append("representative errors file does not hold 12 examples")
    if errors.get("candidates_evaluated") != 100:
        problems.append("candidates_evaluated != 100")
    if errors.get("candidates_per_method") != {"naive": 87, "replay": 92}:
        problems.append(f"candidate errors changed: {errors.get('candidates_per_method')}")
    sessions: dict[int, int] = {}
    for example in errors.get("examples") or []:
        if example.get("split") != "test":
            problems.append(f"example {example.get('relative_path')} not a test record")
        session = example.get("session")
        sessions[session] = sessions.get(session, 0) + 1
    if sessions != {3: 4, 7: 4, 10: 4}:
        problems.append(f"session balance changed: {sessions}")
    if selection.get("selected_method") != "replay":
        problems.append("final_model_selection did not select replay")
    if selection.get("sha256") != LOCKED_FINAL_SHA256:
        problems.append("final_model_selection sha256 changed")
    naive_metrics = selection.get("naive_metrics") or {}
    replay_metrics = selection.get("replay_metrics") or {}
    if naive_metrics.get("average_incremental_accuracy") != LOCKED_NAIVE_AVG_INCR:
        problems.append("naive average incremental accuracy changed")
    if replay_metrics.get("average_incremental_accuracy") != LOCKED_REPLAY_AVG_INCR:
        problems.append("replay average incremental accuracy changed")
    source = (environment.get("session_facts") or {}).get("source")
    if not source or not (PROJECT_ROOT / source).is_file():
        problems.append(f"environment_analysis source missing: {source!r}")
    if problems:
        return False, _problem_list(problems)
    return True, (
        "status=complete, selected=replay, integrity PASS, 12 errors "
        "(s3:4, s7:4, s10:4), 100 candidates/method (87/92 errors)"
    )


# ---------------------------------------------------------------------------
# stage 8 - final model
# ---------------------------------------------------------------------------


def check_final_model_load() -> tuple[bool, str]:
    import torch

    from src.training import load_final_model

    model, _ = load_final_model()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64))
    problems: list[str] = []
    if tuple(output.shape) != (2, 50):
        problems.append(f"forward shape {tuple(output.shape)} != (2, 50)")
    if not bool(torch.isfinite(output).all()):
        problems.append("forward produced non-finite values")
    if problems:
        return False, _problem_list(problems)
    return True, "SmallConvNet strict-loaded; inference forward (2, 50) finite"


def check_final_model_hash() -> tuple[bool, str]:
    final_pt = PROJECT_ROOT / "models" / "continual" / "final_model.pt"
    replay_pt = (
        PROJECT_ROOT / "models" / "continual" / "phase5_nic" / "replay" / "checkpoint.pt"
    )
    meta = _load_json("models/continual/final_model.json")
    problems: list[str] = []
    if not final_pt.is_file() or not replay_pt.is_file():
        return False, "final or replay checkpoint missing"
    digest = _sha256(final_pt)
    if digest != LOCKED_FINAL_SHA256:
        problems.append(f"final_model.pt sha256 {digest[:16]}… != frozen digest")
    if meta.get("sha256") != digest:
        problems.append("final_model.json sha256 disagrees with the file")
    if final_pt.read_bytes() != replay_pt.read_bytes():
        problems.append("final model is not byte-identical to the replay checkpoint")
    if meta.get("method") != "replay" or meta.get("num_classes") != 50:
        problems.append("final_model.json method/num_classes wrong")
    if problems:
        return False, _problem_list(problems)
    return True, f"sha256 {digest[:16]}… == metadata == phase5 replay checkpoint (byte-identical)"


def check_final_model_schema() -> tuple[bool, str]:
    from src.training import read_checkpoint_schema, validate_checkpoint_schema

    problems = validate_checkpoint_schema(
        expected_num_classes=50, expected_method="replay"
    )
    schema = read_checkpoint_schema()
    if schema.get("format_version") != 1:
        problems.append(f"format_version {schema.get('format_version')!r} != 1")
    if schema.get("model_state_tensors") != 23:
        problems.append(f"model_state has {schema.get('model_state_tensors')} tensors != 23")
    if schema.get("current_experience") != 78:
        problems.append(f"current_experience {schema.get('current_experience')} != 78")
    if "model_state" not in schema.get("top_level_keys", []):
        problems.append("canonical 'model_state' key missing")
    if problems:
        return False, _problem_list(problems)
    return True, (
        "format_version 1, method replay, 50 classes, 23 tensors; documented "
        "'model_state' schema loads without any top-level state_dict key"
    )


# ---------------------------------------------------------------------------
# stage 9 - leakage contracts
# ---------------------------------------------------------------------------


def check_leakage_contracts() -> tuple[bool, str]:
    import yaml

    from src.data.continual import load_scenario
    from src.data.core50 import TEST_SESSIONS, TRAIN_SESSIONS

    problems: list[str] = []
    config_path = PROJECT_ROOT / "configs" / "phase5_nic.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    eval_sessions = (config.get("evaluation") or {}).get("sessions")
    if eval_sessions != list(TEST_SESSIONS):
        problems.append(f"config evaluation sessions {eval_sessions} != {list(TEST_SESSIONS)}")
    elif set(eval_sessions) & set(TRAIN_SESSIONS):
        problems.append("evaluation sessions overlap the training sessions")

    scenario = load_scenario("NIC")
    if len(scenario.experiences) != 79:
        problems.append(f"scenario has {len(scenario.experiences)} experiences != 79")
    reference_eval = None
    train_paths: set[str] = set()
    total_train = 0
    total_eval = 0
    for experience in scenario.experiences:
        eval_records = experience.evaluation_samples
        if reference_eval is None:
            reference_eval = eval_records
            total_eval = len(eval_records)
        elif eval_records != reference_eval:
            problems.append(
                f"experience {experience.experience_id}: evaluation set differs "
                "from the shared fixed test set"
            )
        eval_paths = {item.relative_path for item in eval_records}
        for record in eval_records:
            if record.split != "test":
                problems.append(f"evaluation record not split=test: {record.relative_path}")
                break
            if record.session_id not in TEST_SESSIONS:
                problems.append(
                    f"evaluation record from train session {record.session_id}"
                )
                break
        for record in experience.train_samples:
            total_train += 1
            if record.split != "train":
                problems.append(f"training record has split={record.split!r}")
                break
            if record.session_id in TEST_SESSIONS:
                problems.append(f"training record from test session {record.session_id}")
                break
            if record.relative_path in train_paths:
                problems.append(f"train path exposed twice: {record.relative_path}")
                break
            if record.relative_path in eval_paths:
                problems.append(f"evaluation path in training: {record.relative_path}")
                break
            train_paths.add(record.relative_path)
    if problems:
        return False, _problem_list(problems)
    return True, (
        f"NIC run0: 79 experiences, {total_eval} shared test records (s3/s7/s10), "
        f"{total_train} train paths, 0 overlaps"
    )


# ---------------------------------------------------------------------------
# stage 10 - tests, compilation, prototype preservation
# ---------------------------------------------------------------------------


def _stream_with_progress(
    command: list[str],
    *,
    title: str,
    line_filter: Callable[[str], int | None],
    total_hint: int,
) -> tuple[int, list[str], int]:
    """Run ``command``, feeding ``line_filter`` progress into a PhaseProgress."""
    progress = PhaseProgress(title, [title])
    progress.update(0, total_hint, "starting", force=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    tail: deque[str] = deque(maxlen=40)
    counted = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line.rstrip())
        counted_or_none = line_filter(line)
        if counted_or_none is not None:
            counted = counted_or_none
            progress.update(counted, total_hint)
    proc.wait()
    progress.finish()
    return proc.returncode or 0, list(tail), counted


def check_pytest_suite() -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    match = re.search(r"(\d+) tests? collected", collect.stdout)
    if collect.returncode != 0 or not match:
        tail = (collect.stdout + collect.stderr).strip().splitlines()[-6:]
        return False, f"test collection failed: {' | '.join(tail)}"
    total = int(match.group(1))

    counters = {"seen": set(), "current": 0, "summary": ""}
    result_line = re.compile(
        r"^(.*)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+\[\s*\d+%\]\s*$"
    )

    def line_filter(line: str) -> int | None:
        matched = result_line.match(line.rstrip())
        if matched:
            node = matched.group(1).strip()
            if node and node not in counters["seen"]:
                counters["seen"].add(node)
                counters["current"] += 1
                return counters["current"]
            return None
        if " in " in line and ("passed" in line or "failed" in line):
            found = re.search(r"([\w, ]+ (?:passed|failed)[\w, ]* in [\d.]+s)", line)
            if found:
                counters["summary"] = found.group(1).strip()
        return None

    status_before = _git("status", "--porcelain")
    returncode, tail, _ = _stream_with_progress(
        [sys.executable, "-m", "pytest", "tests", "-v", "--tb=short"],
        title="pytest tests -v",
        line_filter=line_filter,
        total_hint=total,
    )
    summary = counters["summary"]
    passed_line = summary or f"{counters['current']}/{total} result lines seen"
    problems: list[str] = []
    if returncode != 0:
        problems.append(
            f"pytest exit {returncode}: {summary or ' | '.join(tail[-6:])}"
        )
    if counters["current"] != total:
        problems.append(f"counted {counters['current']} of {total} collected tests")
    status_after = _git("status", "--porcelain")
    before_lines = set(status_before.splitlines())
    new_dirty = [
        line for line in status_after.splitlines() if line not in before_lines
    ]
    if new_dirty:
        problems.append(f"suite changed tracked files: {new_dirty[:4]}")
    if problems:
        return False, _problem_list(problems)
    return True, f"{passed_line}; suite left the working tree as it found it"


def check_bytecode_compilation() -> tuple[bool, str]:
    py_files = sorted(
        path
        for folder in ("src", "scripts", "tests")
        for path in (PROJECT_ROOT / folder).rglob("*.py")
    )
    total = len(py_files)
    counters = {"current": 0}

    def line_filter(line: str) -> int | None:
        if line.startswith("Compiling"):
            counters["current"] += 1
            return counters["current"]
        return None

    returncode, tail, current = _stream_with_progress(
        [sys.executable, "-m", "compileall", "-f", "src", "scripts", "tests"],
        title="compileall src scripts tests",
        line_filter=line_filter,
        total_hint=total,
    )
    if returncode != 0:
        return False, f"compileall exit {returncode}: {' | '.join(tail[-6:])}"
    if current != total:
        return False, f"compiled {current} files, expected {total}"
    return True, f"{current}/{total} files byte-compiled"


def check_coco_prototype_preservation() -> tuple[bool, str]:
    import yaml

    problems: list[str] = []
    train_data = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "train_data.yaml").read_text(encoding="utf-8")
    )
    if train_data.get("path") != "data/processed/household_objects":
        problems.append("train_data.yaml path changed")
    if "test" in train_data:
        problems.append("train_data.yaml must not contain a test split")
    if train_data.get("nc") != 19:
        problems.append("train_data.yaml class count != 19")
    yaml.safe_load((PROJECT_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))

    baseline = _load_json("reports/baseline_results.json")
    if baseline.get("metrics", {}).get("map50") != LOCKED_BASELINE_MAP50:
        problems.append("baseline mAP50 changed")
    if len(baseline.get("per_class_metrics") or []) != 19:
        problems.append("baseline per-class metrics != 19 classes")

    training = _load_json("reports/phase6_training_results.json")
    if training.get("status") != "completed" or training.get("experiment", {}).get("seed") != 42:
        problems.append("prototype YOLO training record changed")
    if training.get("validation_best_pt", {}).get("image_count") != 4544:
        problems.append("prototype validation image count changed")

    weights = (
        PROJECT_ROOT / "yolo26n.pt",
        PROJECT_ROOT / "models" / "baseline" / "yolo26n.pt",
        PROJECT_ROOT / "models" / "training" / "household_yolo26n" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "training" / "household_yolo26n" / "weights" / "last.pt",
    )
    for weight in weights:
        if not weight.is_file() or weight.stat().st_size < 1_000_000:
            problems.append(f"prototype weights missing: {weight.name}")
    for folder in ("data/raw/coco", "data/processed/household_objects"):
        if not (PROJECT_ROOT / folder).is_dir():
            problems.append(f"prototype dataset missing: {folder}")
    if problems:
        return False, _problem_list(problems)
    return True, (
        f"COCO prototype intact: baseline mAP50 {LOCKED_BASELINE_MAP50} (19 classes), "
        "YOLO weights + processed dataset present, no test split in train data"
    )


# ---------------------------------------------------------------------------
# optional full scan (explicit, never automatic)
# ---------------------------------------------------------------------------


def run_full_scan() -> tuple[bool, str]:
    summary_path = PROJECT_ROOT / "reports" / "phase2_core50_summary.json"
    backup = summary_path.read_bytes() if summary_path.is_file() else None
    counters = {"current": 0}
    pattern = re.compile(r"Current(?: sample)?:\s*([\d,]+)\s*/\s*([\d,]+)")

    def line_filter(line: str) -> int | None:
        found = pattern.search(line)
        if found:
            counters["current"] = int(found.group(1).replace(",", ""))
            return counters["current"]
        return None

    try:
        returncode, tail, _ = _stream_with_progress(
            [sys.executable, "scripts/validate_core50.py", "--no-samples"],
            title="full dataset scan (164,866 images)",
            line_filter=line_filter,
            total_hint=EXPECTED_IMAGES,
        )
    finally:
        if backup is not None:
            summary_path.write_bytes(backup)
    if returncode != 0:
        return False, f"validate_core50 exit {returncode}: {' | '.join(tail[-6:])}"
    return True, f"all {EXPECTED_IMAGES:,} images validated; tracked summary restored"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _stage_list() -> tuple[tuple[str, tuple[tuple[str, CheckFn], ...]], ...]:
    return (
        ("Environment and project root", (("environment", check_environment),)),
        (
            "Git repository and repository hygiene",
            (
                ("Git repository", check_git_repository),
                ("repository hygiene", check_repository_hygiene),
            ),
        ),
        (
            "CORe50 dataset and official metadata",
            (
                ("CORe50 dataset presence", check_core50_dataset_presence),
                ("CORe50 official metadata", check_core50_official_metadata),
            ),
        ),
        (
            "CORe50 object mapping semantics",
            (("50-identity mapping", check_object_mapping_semantics),),
        ),
        (
            "Continual data pipeline (Phase 3)",
            (
                ("Phase 3 manifest", check_continual_pipeline_manifest),
                ("NI/NC/NIC support", check_ni_nc_nic_support),
            ),
        ),
        (
            "Continual learning APIs (Phase 4)",
            (
                ("naive continual API", check_naive_continual_api),
                ("ReplayMemory contract", check_replay_memory_contract),
            ),
        ),
        (
            "Phase 5 and Phase 6 artifacts",
            (
                ("Phase 5 NIC results", check_phase5_results),
                ("Phase 6 analysis", check_phase6_results),
            ),
        ),
        (
            "Final model (Phase 6 selection)",
            (
                ("final model load", check_final_model_load),
                ("final model hash", check_final_model_hash),
                ("final model schema", check_final_model_schema),
            ),
        ),
        ("Leakage contracts", (("leakage contracts", check_leakage_contracts),)),
        (
            "Tests, compilation, COCO prototype preservation",
            (
                ("pytest test suite", check_pytest_suite),
                ("bytecode compilation", check_bytecode_compilation),
                ("COCO prototype preservation", check_coco_prototype_preservation),
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help=(
            "also run the explicit full 164,866-image dataset validation "
            "(delegates to scripts/validate_core50.py; never runs by default)"
        ),
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    stages = _stage_list()
    total_checks = sum(len(checks) for _, checks in stages)
    started = time.monotonic()
    results: list[dict] = []
    index = 0

    print("PHASE 1-6 REPAIR VERIFICATION")
    print(f"{total_checks} checks in {len(stages)} stages (read-only)")
    for stage_index, (title, checks) in enumerate(stages, start=1):
        print(f"\n[{stage_index}/{len(stages)}] {title}")
        for name, function in checks:
            index += 1
            check_started = time.monotonic()
            try:
                ok, detail = function()
            except Exception as exc:  # noqa: BLE001 - surfaced as a failure
                ok, detail = False, f"unexpected {type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - check_started
            results.append(
                {"index": index, "name": name, "ok": ok, "detail": detail}
            )
            tag = "OK  " if ok else "FAIL"
            suffix = f"  ({elapsed:.1f}s)" if elapsed >= 2.0 else ""
            print(f"  [{index:2d}/{total_checks}] {name:<28} {tag} - {detail}{suffix}")

    extra_failures: list[str] = []
    if args.full_scan:
        print("\n[optional] Full dataset scan (explicit --full-scan)")
        scan_ok, scan_detail = run_full_scan()
        tag = "OK  " if scan_ok else "FAIL"
        print(f"  [full] 164,866-image scan       {tag} - {scan_detail}")
        if not scan_ok:
            extra_failures.append(f"full scan: {scan_detail}")

    failures = [entry for entry in results if not entry["ok"]]
    elapsed_total = time.monotonic() - started
    print(
        f"\n{total_checks - len(failures)}/{total_checks} checks passed "
        f"in {elapsed_total:.1f}s"
    )
    if failures or extra_failures:
        print("PHASE 1–6 VERIFICATION: FAIL")
        print(f"FAILURES: {len(failures) + len(extra_failures)}")
        for entry in failures:
            print(f"  [{entry['index']}/{total_checks}] {entry['name']}: {entry['detail']}")
        for detail in extra_failures:
            print(f"  [full] {detail}")
        return 1
    print("PHASE 1–6 VERIFICATION: PASS")
    print("FAILURES: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
