"""Smoke tests for the project foundation."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "app",
    "configs",
    "data/raw",
    "data/processed",
    "data/splits",
    "examples/input",
    "examples/output",
    "models",
    "notebooks",
    "reports/figures",
    "scripts",
    "src/data",
    "src/training",
    "src/evaluation",
    "src/inference",
    "src/utils",
    "tests",
    ".github/workflows",
]

REQUIRED_FILES = [
    "AGENTS.md",
    "PROJECT_PLAN.md",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "LICENSE",
]


def test_required_directories_exist() -> None:
    missing = [d for d in REQUIRED_DIRS if not (PROJECT_ROOT / d).is_dir()]
    assert not missing, f"Missing directories: {missing}"


def test_required_files_exist() -> None:
    missing = [f for f in REQUIRED_FILES if not (PROJECT_ROOT / f).is_file()]
    assert not missing, f"Missing files: {missing}"


def test_requirements_ml_deps_match_project_stack() -> None:
    lines = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    active = [ln for ln in lines.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    text = "\n".join(active)
    assert "torch" in text, "Phase 5 baseline needs torch"
    assert "ultralytics" in text, "Phase 5 baseline needs ultralytics"
    for forbidden in ("tensorflow", "keras"):
        assert forbidden not in text, f"Unexpected dependency: {forbidden}"


def test_project_plan_lists_all_phases() -> None:
    text = (PROJECT_ROOT / "PROJECT_PLAN.md").read_text(encoding="utf-8")
    for phase in ("Setup", "Dataset", "Training", "Evaluation", "Deployment", "Documentation"):
        assert phase in text, f"Phase missing from PROJECT_PLAN.md: {phase}"


def test_yaml_config_is_parseable(tmp_path: Path) -> None:
    config = {"seed": 42, "epochs": 100, "classes": ["cup", "bottle"]}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert loaded == config


def test_src_packages_importable() -> None:
    import src
    import src.data
    import src.evaluation
    import src.inference
    import src.training
    import src.utils

    assert src.__file__ is not None
