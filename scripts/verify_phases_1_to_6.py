"""Permanent read-only integrity verifier for Phases 1-6.

Resolves the project root from this file's location (never from the current
working directory), validates the real JSON schemas of the Phase 5 and
Phase 6 reports, re-scans dataset counts and YOLO annotation structure,
loads the trained model, checks Git/secret safety, and runs the test
suite. It never retrains, never modifies datasets or weights, and never
rewrites reports.

Usage: .\\.venv\\Scripts\\python.exe scripts\\verify_phases_1_to_6.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

EXPECTED_CLASSES = [
    "chair",
    "book",
    "bottle",
    "cup",
    "dining table",
    "bowl",
    "potted plant",
    "wine glass",
    "cell phone",
    "clock",
    "tv",
    "couch",
    "remote",
    "sink",
    "laptop",
    "bed",
    "keyboard",
    "refrigerator",
    "mouse",
]

SPLIT_EXPECTATIONS = {
    "train": {"images": 40890, "instances": 184709},
    "val": {"images": 4544, "instances": 21100},
    "test": {"images": 1965, "instances": 9174},
}

RAW_TREE_EXPECTATIONS = {
    "data/raw": {"files": 123298, "bytes": 41368946252},
    "data/raw/coco": {"files": 123296, "bytes": 41368942519},
}

REQUIRED_COMMITS = ["948433c", "2951436", "8af6e04", "94f05a5", "2232b19"]

FORBIDDEN_TRACKED = [
    (re.compile(r"^data/raw/coco/(train2017|val2017|zips|annotations)/"), "raw COCO payload"),
    (re.compile(r"^data/processed/household_objects/(images|labels)/"), "prepared images/labels"),
    (re.compile(r"\.(pt|pth|onnx|safetensors)$"), "model weights"),
    (re.compile(r"\.cache$"), "label cache"),
    (re.compile(r"^models/training/.+\.log$"), "training logs"),
]

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{34,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
]


class Verifier:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.results: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, ok, detail))
        status = "PASS" if ok else "FAIL"
        suffix = f" - {detail}" if detail else ""
        print(f"[{status}] {name}{suffix}", flush=True)
        return ok

    def fatal(self, name: str, detail: str) -> None:
        self.record(name, False, detail)
        self.summary()
        sys.exit(1)

    def summary(self) -> int:
        failed = [r for r in self.results if not r[1]]
        print("", flush=True)
        print("=" * 60, flush=True)
        for name, ok, detail in self.results:
            if not ok:
                print(f"FAILED: {name} :: {detail}", flush=True)
        verdict = "PASS" if not failed else "FAIL"
        print(
            f"VERIFIER RESULT: {verdict} "
            f"({len(self.results) - len(failed)}/{len(self.results)} checks passed)",
            flush=True,
        )
        print("=" * 60, flush=True)
        return 0 if not failed else 1


def resolve_project_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    required = [".git", ".venv", "PROJECT_PLAN.md", "README.md", "DATASET.md", "configs", "data", "reports", "scripts", "tests"]
    missing = [item for item in required if not (root / item).exists()]
    if missing:
        raise SystemExit(
            f"FATAL: resolved root {root} is missing {missing}; "
            "this verifier must live in <project>/scripts/"
        )
    cwd = Path.cwd().resolve()
    if cwd != root:
        print(f"note: current directory {cwd} differs; using resolved root {root}", flush=True)
    return root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_totals(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def check_root(verifier: Verifier) -> None:
    root = verifier.root
    markers = [".git", ".venv/Scripts/python.exe", "PROJECT_PLAN.md", "README.md", "DATASET.md", "configs", "data", "reports", "scripts", "tests"]
    missing = [item for item in markers if not (root / item).exists()]
    verifier.record("project root resolves with all required entries", not missing, ", ".join(missing))


def check_git_state(verifier: Verifier) -> None:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=verifier.root, capture_output=True, text=True, check=True).stdout

    branch = git("branch", "--show-current").strip()
    verifier.record("git branch is main", branch == "main", branch)
    log = git("log", "--oneline", "-n", "20")
    missing = [commit for commit in REQUIRED_COMMITS if commit not in log]
    verifier.record("required phase commits present", not missing, "missing: " + ", ".join(missing))
    status = git("status", "--porcelain")
    modified = [line for line in status.splitlines() if not line.startswith("??") and line.strip()]
    untracked = [line for line in status.splitlines() if line.startswith("??")]
    verifier.record(
        "git tracked files unmodified",
        not modified,
        "; ".join(modified[:5]),
    )
    if untracked:
        print(f"    note: {len(untracked)} untracked path(s) present", flush=True)


def check_dataset(verifier: Verifier) -> None:
    root = verifier.root
    dataset = root / "data" / "processed" / "household_objects"
    problems: list[str] = []
    totals = {"images": 0, "instances": 0}
    started = time.time()
    for split, expected in SPLIT_EXPECTATIONS.items():
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            problems.append(f"{split}: missing images/ or labels/ directory")
            continue
        images = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        labels = {p.stem for p in labels_dir.iterdir() if p.suffix.lower() == ".txt"}
        missing_labels = images - labels
        orphan_labels = labels - images
        if missing_labels:
            problems.append(f"{split}: {len(missing_labels)} images without a label file")
        if orphan_labels:
            problems.append(f"{split}: {len(orphan_labels)} orphan label files")
        if len(images) != expected["images"]:
            problems.append(f"{split}: {len(images)} images != {expected['images']}")
        instances = 0
        for index, stem in enumerate(sorted(labels), 1):
            label_path = labels_dir / f"{stem}.txt"
            for lineno, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    problems.append(f"{label_path.name}:{lineno} has {len(fields)} fields")
                    continue
                try:
                    class_id = int(fields[0])
                    x, y, w, h = (float(value) for value in fields[1:])
                except ValueError:
                    problems.append(f"{label_path.name}:{lineno} non-numeric field")
                    continue
                if not 0 <= class_id <= 18:
                    problems.append(f"{label_path.name}:{lineno} class id {class_id} outside 0..18")
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    problems.append(f"{label_path.name}:{lineno} x/y not normalized")
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    problems.append(f"{label_path.name}:{lineno} w/h not positive-normalized")
                instances += 1
            if index % 10000 == 0:
                print(f"    progress: {split} {index}/{len(labels)} labels scanned", flush=True)
        if instances != expected["instances"]:
            problems.append(f"{split}: {instances} instances != {expected['instances']}")
        totals["images"] += len(images)
        totals["instances"] += instances
        print(f"    {split}: images={len(images)} labels={len(labels)} instances={instances}", flush=True)
    if totals["images"] != 47399:
        problems.append(f"total images {totals['images']} != 47399")
    if totals["instances"] != 214983:
        problems.append(f"total instances {totals['instances']} != 214983")
    verifier.record(
        "dataset counts + YOLO annotation structure (47,399 / 214,983)",
        not problems,
        "; ".join(problems[:5]) if problems else f"{time.time() - started:.0f}s",
    )


def check_raw_coco(verifier: Verifier) -> None:
    root = verifier.root
    problems: list[str] = []
    for relative, expected in RAW_TREE_EXPECTATIONS.items():
        path = root / relative
        files, size = tree_totals(path)
        if files != expected["files"]:
            problems.append(f"{relative}: {files} files != {expected['files']}")
        if size != expected["bytes"]:
            problems.append(f"{relative}: {size} bytes != {expected['bytes']}")
    aux = root / "data" / "raw" / "DATASET_INFO.txt"
    keep = root / "data" / "raw" / ".gitkeep"
    if not aux.is_file() or aux.stat().st_size != 3733:
        problems.append("data/raw/DATASET_INFO.txt missing or changed")
    if not keep.is_file():
        problems.append("data/raw/.gitkeep missing")
    if not problems:
        print(
            "    scope note: data/raw (123,298 files) = data/raw/coco (123,296) "
            "+ .gitkeep (0 B) + DATASET_INFO.txt (3,733 B); both scopes verified",
            flush=True,
        )
    verifier.record("raw COCO tree matches recorded totals (both scopes)", not problems, "; ".join(problems))


def check_phase5(verifier: Verifier) -> None:
    path = verifier.root / "reports" / "baseline_results.json"
    if not path.is_file():
        verifier.record("phase5 baseline_results.json exists", False, "missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    model = data.get("model")
    model_text = model.get("path", "") if isinstance(model, dict) else str(model)
    if not str(model_text).endswith("yolo26n.pt"):
        problems.append(f"model is {model_text!r}")
    if data.get("test_image_count") != 1965:
        problems.append(f"test_image_count={data.get('test_image_count')}")
    if data.get("test_instance_count") != 9174:
        problems.append(f"test_instance_count={data.get('test_instance_count')}")
    metrics = data.get("metrics", {})
    expected_metrics = {"precision": 0.6924, "recall": 0.3914, "f1": 0.5001, "map50": 0.5485, "map50_95": 0.3961}
    for key, value in expected_metrics.items():
        if round(float(metrics.get(key, -1)), 4) != value:
            problems.append(f"{key}={metrics.get(key)}")
    mapping = data.get("class_mapping", {})
    project_map = mapping.get("coco_to_project", {})
    if len(project_map) != 19:
        problems.append(f"class_mapping covers {len(project_map)} classes")
    if mapping.get("mapped_classes") != 19:
        problems.append(f"mapped_classes={mapping.get('mapped_classes')}")
    per_class = data.get("per_class_metrics", [])
    if len(per_class) != 19:
        problems.append(f"per_class rows={len(per_class)}")
    verifier.record("phase5 JSON semantic facts (schema-aware)", not problems, "; ".join(problems))


def check_phase6(verifier: Verifier) -> None:
    path = verifier.root / "reports" / "phase6_training_results.json"
    if not path.is_file():
        verifier.record("phase6 results JSON exists", False, "missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if data.get("phase") != 6:
        problems.append(f"phase={data.get('phase')}")
    if data.get("status") != "completed":
        problems.append(f"status={data.get('status')}")
    required_sections = [
        "experiment",
        "environment",
        "integrity",
        "training",
        "validation_best_pt",
        "zero_shot_baseline_val",
        "phase5_baseline_test_reference",
        "controlled_comparison_val_split",
        "speed_benchmark",
        "artifacts",
    ]
    absent = [section for section in required_sections if section not in data]
    if absent:
        problems.append("missing sections: " + ", ".join(absent))
    experiment = data.get("experiment", {})
    if not str(experiment.get("base_model", "")).endswith("yolo26n.pt"):
        problems.append(f"base_model={experiment.get('base_model')!r}")
    if experiment.get("epochs_completed") != 2:
        problems.append(f"epochs_completed={experiment.get('epochs_completed')}")
    if not 1 <= int(experiment.get("epochs_planned", 0)) <= 50:
        problems.append(f"epochs_planned={experiment.get('epochs_planned')}")
    if not experiment.get("device"):
        problems.append("experiment.device missing")
    environment = data.get("environment", {})
    for key in ("python", "torch", "ultralytics", "xpu_available", "device_used"):
        if key not in environment:
            problems.append(f"environment.{key} missing")
    training = data.get("training", {})
    if training.get("status") != "completed":
        problems.append(f"training.status={training.get('status')}")
    if training.get("train_images") != 40890:
        problems.append(f"training.train_images={training.get('train_images')}")
    if training.get("val_images_per_epoch") != 4544:
        problems.append(f"training.val_images_per_epoch={training.get('val_images_per_epoch')}")
    validation = data.get("validation_best_pt", {})
    if validation.get("image_count") != 4544:
        problems.append(f"validation.image_count={validation.get('image_count')}")
    if validation.get("instance_count") != 21100:
        problems.append(f"validation.instance_count={validation.get('instance_count')}")
    weights = validation.get("weights")
    if not weights or not (verifier.root / weights).is_file():
        problems.append(f"best weights path missing: {weights}")
    if data.get("integrity", {}).get("unchanged") is not True:
        problems.append("integrity.unchanged is not true")
    verifier.record("phase6 JSON nested schema + facts", not problems, "; ".join(problems))


def check_models(verifier: Verifier) -> None:
    try:
        from ultralytics import YOLO
    except Exception as error:
        verifier.record("ultralytics import for model check", False, str(error))
        return
    root = verifier.root
    problems: list[str] = []
    weights_dir = root / "models" / "training" / "household_yolo26n" / "weights"
    for filename in ("best.pt", "last.pt"):
        path = weights_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{filename} missing or empty")
            continue
        try:
            model = YOLO(str(path))
            names = [model.names[index] for index in range(len(model.names))]
            if names != EXPECTED_CLASSES:
                problems.append(f"{filename} class order mismatch: {names}")
        except Exception as error:
            problems.append(f"{filename} failed to load: {error}")
    verifier.record("trained weights load with 19 classes in exact order", not problems, "; ".join(problems))
    baseline = root / "models" / "baseline" / "yolo26n.pt"
    if not baseline.is_file():
        verifier.record("baseline yolo26n.pt present", False, "missing")
        return
    verifier.record("baseline yolo26n.pt present", True, f"{baseline.stat().st_size} bytes")
    root_copy = root / "yolo26n.pt"
    if root_copy.is_file():
        identical = sha256(baseline) == sha256(root_copy)
        verifier.record("root yolo26n.pt identical to baseline (SHA-256)", identical)


def check_git_tracking(verifier: Verifier) -> None:
    listing = subprocess.run(
        ["git", "ls-files"], cwd=verifier.root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    violations = []
    for path in listing:
        for pattern, label in FORBIDDEN_TRACKED:
            if pattern.search(path):
                violations.append(f"{path} ({label})")
    verifier.record(
        f"git tracks no forbidden payloads ({len(listing)} tracked files)",
        not violations,
        "; ".join(violations[:5]),
    )


def check_secrets(verifier: Verifier) -> None:
    listing = subprocess.run(
        ["git", "ls-files"], cwd=verifier.root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    hits: list[str] = []
    for relative in listing:
        path = verifier.root / relative
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(relative)
                break
    verifier.record("no obvious secrets in tracked text files", not hits, "; ".join(hits))


def check_docs(verifier: Verifier) -> None:
    root = verifier.root
    required = [
        "README.md",
        "PROJECT_PLAN.md",
        "DATASET.md",
        "reports/baseline_report.md",
        "reports/baseline_results.json",
        "reports/phase6_training_report.md",
        "reports/phase6_training_results.json",
        "reports/phase6_training_summary.md",
        "reports/phase6_training_commands.md",
    ]
    missing = [item for item in required if not (root / item).is_file()]
    verifier.record("documentation set complete", not missing, ", ".join(missing))
    summary = (root / "reports" / "phase6_training_summary.md").read_text(encoding="utf-8", errors="ignore")
    verifier.record("phase6 summary labeled as validation result", "VALIDATION RESULT" in summary)
    report = (root / "reports" / "phase6_training_report.md").read_text(encoding="utf-8", errors="ignore")
    sections = [int(value) for value in re.findall(r"^## (\d+)\. ", report, flags=re.MULTILINE)]
    verifier.record("phase6 report has all 19 sections", sections == list(range(1, 20)), str(sections))


def check_tests_and_compile(verifier: Verifier) -> None:
    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=verifier.root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = (pytest_run.stdout + pytest_run.stderr).strip().splitlines()[-1:]
    count_line = tail[0] if tail else "no output"
    verifier.record("pytest suite passes", pytest_run.returncode == 0, count_line)
    compile_run = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
        cwd=verifier.root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    verifier.record("compileall src scripts tests", compile_run.returncode == 0, compile_run.stderr.strip()[:200])


def check_test_portability(verifier: Verifier) -> None:
    problems: list[str] = []
    for path in sorted((verifier.root / "tests").glob("test_*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"[A-Za-z]:\\\\|/home/", line):
                if "assert" not in line:
                    problems.append(f"{path.name}:{lineno}")
    verifier.record("no category-A absolute runtime paths in tests", not problems, "; ".join(problems))


def main() -> int:
    root = resolve_project_root()
    print(f"project root: {root}", flush=True)
    print("read-only verification: no training, no dataset writes, no weight changes", flush=True)
    print("-" * 60, flush=True)
    verifier = Verifier(root)
    stages: list[tuple[str, Callable[[Verifier], None]]] = [
        ("root", check_root),
        ("git state", check_git_state),
        ("dataset", check_dataset),
        ("raw COCO", check_raw_coco),
        ("phase5 report", check_phase5),
        ("phase6 report", check_phase6),
        ("models", check_models),
        ("git tracking", check_git_tracking),
        ("secrets", check_secrets),
        ("documentation", check_docs),
        ("test portability", check_test_portability),
        ("tests + compile", check_tests_and_compile),
    ]
    for name, stage in stages:
        print(f"--- {name} ---", flush=True)
        started = time.time()
        stage(verifier)
        print(f"    ({time.time() - started:.1f}s)", flush=True)
    return verifier.summary()


if __name__ == "__main__":
    sys.exit(main())
