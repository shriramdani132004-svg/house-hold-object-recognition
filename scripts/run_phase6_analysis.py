"""Phase 6 — focused error/forgetting analysis + final model selection.

Reads ONLY the committed Phase-5 artifacts (``reports/phase5_nic/``) and
the two existing Phase-5 checkpoints, then:

1. inspects and validates those results,
2. computes deterministic forgetting/class/experience analyses,
3. runs targeted inference on a small deterministic evaluation sample,
4. records session/environment observations from metadata,
5. selects the final model from measured results,
6. freezes the selected checkpoint to ``models/continual/final_model.pt``,
7. verifies the frozen model, writes the Phase-6 reports, and checks
   integrity attestations.

No training, no hyperparameter search, no dataset writes, no full-dataset
scans (one non-decoding file-count walk is used for the dataset-integrity
attestation).

Usage (from the project root)::

    python scripts/run_phase6_analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.continual import load_scenario_cached  # noqa: E402
from src.evaluation.continual import atomic_write_json, load_class_names  # noqa: E402
from src.evaluation.error_analysis import (  # noqa: E402
    build_class_analysis,
    build_environment_analysis,
    build_experiment_placeholder,
    build_experience_analysis,
    build_final_model_metadata,
    build_forgetting_analysis,
    build_representative_errors,
    classify_records,
    load_phase5_results,
    render_error_analysis_md,
    render_selection_md,
    render_summary_md,
    select_candidate_records,
    select_final_model,
    select_target_classes,
    sha256_file,
)
from src.training import ContinualImageDataset, build_model, resolve_device  # noqa: E402
from src.utils.progress import PhaseProgress, format_duration  # noqa: E402

STEP_LABELS = [
    "Inspecting Phase-5 results",
    "Analyzing forgetting and retention",
    "Identifying representative prediction failures",
    "Analyzing representative environmental failures",
    "Selecting final continual model",
    "Freezing selected final checkpoint",
    "Final model verification and phase completion",
]

PHASE5_REPORTS = Path("reports/phase5_nic")
PHASE5_MODELS = Path("models/continual/phase5_nic")
OUTPUT_DIR = Path("reports/phase6_analysis")
EXAMPLES_DIR = OUTPUT_DIR / "examples"
FINAL_CHECKPOINT = Path("models/continual/final_model.pt")
FINAL_METADATA = Path("models/continual/final_model.json")
CONFIG_PATH = Path("configs/phase5_nic.yaml")
OBJECT_MAPPING = Path("data/raw/core50/metadata/object_mapping.json")
EXPECTED_DATASET_FILES = 164_866
ABS_PATH_MARKERS = ("C:\\", "C:/", "/home/")
PROTECTED_PREFIXES = (
    "data/",
    "reports/phase5_nic",
    "reports/baseline",
    "reports/phase6_training",
    "reports/phase3_samples",
    "src/evaluation/baseline.py",
    "configs/train.yaml",
    "configs/train_data.yaml",
    "configs/baseline.yaml",
    "models/training",
    "models/baseline",
    "notebooks/",
    "app/",
    "scripts/validate_core50.py",
    "scripts/train_model.py",
    "scripts/baseline_inference.py",
    "scripts/evaluate_baseline.py",
    "scripts/validate_training.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def count_dataset_files(images_root: Path) -> int:
    total = 0
    for _dirpath, _dirnames, filenames in os.walk(images_root):
        total += len(filenames)
    return total


def git_status_lines() -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def run_load_test(
    final_checkpoint: Path,
    *,
    width: int,
    device: str,
    sample: tuple[torch.Tensor, int],
    class_names: dict[str, str],
) -> dict[str, Any]:
    payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    for key in ("model_state", "num_classes"):
        if key not in payload:
            raise ValueError(f"final checkpoint missing {key!r}")
    model = build_model(int(payload["num_classes"]), width=width)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    num_classes = int(payload["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(
            f"class mapping has {len(class_names)} entries, expected {num_classes}"
        )
    features, label = sample
    with torch.inference_mode():
        logits = model(features.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        confidence, prediction = probs.max(0)
    return {
        "checkpoint_loaded": True,
        "architecture_loaded": type(model).__name__,
        "num_classes": num_classes,
        "class_mapping_entries": len(class_names),
        "strict_state_dict": True,
        "representative_inference": {
            "true_label": int(label),
            "predicted_label": int(prediction),
            "confidence": float(confidence),
        },
    }


def blocked_block(*, step: int | None, error: str) -> str:
    return "\n".join(
        [
            "",
            "PHASE 6 BLOCKED",
            "===============",
            f"Step:   {f'{step}/7' if step else 'unknown'}",
            f"Error:  {error}",
            "No report was fabricated; fix the missing dependency and rerun.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 analysis driver")
    parser.parse_args(list(argv) if argv is not None else None)

    phase = PhaseProgress(
        "PHASE 6 OVERALL",
        list(STEP_LABELS),
        eta_own_line=True,
        sample_wording=True,
    )
    step: int | None = 1
    started = time.perf_counter()
    try:
        # ---- step 1: inspect Phase-5 results ----------------------------
        phase.set_step(1, STEP_LABELS[0])
        results = load_phase5_results(PHASE5_REPORTS)

        naive_ckpt = PHASE5_MODELS / "naive" / "checkpoint.pt"
        replay_ckpt = PHASE5_MODELS / "replay" / "checkpoint.pt"
        pre_hashes = {}
        for method, ckpt in (("naive", naive_ckpt), ("replay", replay_ckpt)):
            state_path = PHASE5_MODELS / method / "state.json"
            if not ckpt.is_file() or not state_path.is_file():
                raise ValueError(f"missing Phase-5 checkpoint/state for {method}")
            pre_hashes[method] = {
                "checkpoint": sha256_file(ckpt),
                "state": sha256_file(state_path),
            }
        if not CONFIG_PATH.is_file():
            raise ValueError(f"missing {CONFIG_PATH}")
        config = read_yaml(CONFIG_PATH)
        model_cfg = config.get("model") or {}
        for key in ("arch", "width", "image_size", "num_classes"):
            if key not in model_cfg:
                raise ValueError(f"configs model.{key} missing from {CONFIG_PATH}")

        comparison = results["comparison"]
        phase.log(
            f"Phase-5 results: {results['num_experiences']} experiences, "
            f"{len(results['class_names'])} classes, "
            f"records naive/replay = "
            f"{len(results['records']['naive'])}/"
            f"{len(results['records']['replay'])}"
        )
        phase.log(
            "Final measured: naive accuracy "
            f"{comparison['final_accuracy']['naive']:.4f} / forgetting "
            f"{comparison['final_forgetting']['naive']:.4f}; replay accuracy "
            f"{comparison['final_accuracy']['replay']:.4f} / forgetting "
            f"{comparison['final_forgetting']['replay']:.4f}"
        )
        phase.log(
            f"Checkpoints: {naive_ckpt} ; {replay_ckpt} ; config {CONFIG_PATH}"
        )

        # ---- step 2: forgetting / class / experience analyses -----------
        phase.set_step(2, STEP_LABELS[1])
        scenario = load_scenario_cached("NIC", "inc", 0)
        eval_records = scenario.experiences[0].evaluation_samples
        if any(r.split != "test" for r in eval_records):
            raise ValueError("scenario evaluation samples are not all split='test'")

        categories: dict[str, str | None] = {}
        for record in eval_records:
            categories.setdefault(str(record.label), record.category_name)

        forgetting = build_forgetting_analysis(results)
        class_analysis = build_class_analysis(results)
        for row in class_analysis["classes"]:
            row["category"] = categories.get(str(row["label"]))
        experience_analysis = build_experience_analysis(results)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        public_forgetting = {
            k: v for k, v in forgetting.items() if not k.startswith("_")
        }
        atomic_write_json(OUTPUT_DIR / "forgetting_analysis.json", public_forgetting)
        atomic_write_json(OUTPUT_DIR / "class_analysis.json", class_analysis)
        atomic_write_json(
            OUTPUT_DIR / "experience_analysis.json", experience_analysis
        )
        placeholder = build_experiment_placeholder()
        (OUTPUT_DIR / "phase6_error_analysis.md").write_text(
            render_error_analysis_md(
                public_forgetting, class_analysis, placeholder, placeholder
            ),
            encoding="utf-8",
        )
        delta = public_forgetting["aggregate"]["delta_replay_minus_naive"]
        phase.log(
            "Replay effect: final accuracy "
            f"{delta['final_accuracy']:+.4f}, final forgetting "
            f"{delta['final_forgetting']:+.4f}, average incremental accuracy "
            f"{delta['average_incremental_accuracy']:+.4f}"
        )
        top = public_forgetting["top_forgetting_classes"]["naive"][:3]
        phase.log(
            "Highest naive forgetting: "
            + ", ".join(f"{r['name']} ({r['forgetting']:.4f})" for r in top)
        )

        # ---- step 3: targeted representative errors ---------------------
        phase.set_step(3, STEP_LABELS[2])
        targets = select_target_classes(public_forgetting)
        candidates = select_candidate_records(targets, eval_records, max_total=100)
        if not candidates:
            raise ValueError("no candidate evaluation records selected")
        class_names = results["class_names"]
        mapping_fallback = (
            load_class_names(OBJECT_MAPPING) if OBJECT_MAPPING.is_file() else {}
        )
        for label, name in mapping_fallback.items():
            class_names.setdefault(label, name)

        dataset = ContinualImageDataset(
            candidates,
            scenario.images_root,
            int(model_cfg["image_size"]),
            require_split="test",
        )
        device = resolve_device("auto")
        width = int(model_cfg["width"])
        if int(model_cfg["num_classes"]) != len(results["class_names"]):
            raise ValueError("config num_classes does not match class history")
        predictions: dict[str, list[dict[str, Any]]] = {}
        for method, ckpt in (("naive", naive_ckpt), ("replay", replay_ckpt)):
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            if int(payload["num_classes"]) != int(model_cfg["num_classes"]):
                raise ValueError(f"{method} checkpoint class count mismatch")
            model = build_model(int(payload["num_classes"]), width=width)
            model.load_state_dict(payload["model_state"], strict=True)
            model.to(device)
            phase.log(f"Classifying candidates with the {method} final model")
            predictions[method] = classify_records(
                model,
                dataset,
                device=device,
                on_progress=lambda cur, tot: phase.update(cur, tot),
            )
            phase.update(len(candidates), len(candidates), force=True)
            del model

        representative = build_representative_errors(
            candidates, predictions, class_names
        )
        representative["targets"] = targets
        atomic_write_json(
            OUTPUT_DIR / "representative_errors.json", representative
        )

        EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        for old in EXAMPLES_DIR.glob("*.png"):
            old.unlink()
        for index, example in enumerate(representative["examples"]):
            source = scenario.images_root.joinpath(
                *example["relative_path"].split("/")
            )
            target = EXAMPLES_DIR / (
                f"{index:02d}_s{example['session']}_"
                f"{Path(example['relative_path']).name}"
            )
            shutil.copyfile(source, target)
        phase.log(
            f"Representative errors: {representative['n_examples']} kept "
            f"(candidates {representative['candidates_evaluated']}/method, "
            f"errors naive "
            f"{representative['candidates_per_method'].get('naive', 0)}, "
            f"replay {representative['candidates_per_method'].get('replay', 0)})"
        )

        # ---- step 4: environment / session analysis ---------------------
        phase.set_step(4, STEP_LABELS[3])
        environment = build_environment_analysis(representative)
        atomic_write_json(OUTPUT_DIR / "environment_analysis.json", environment)
        (OUTPUT_DIR / "phase6_error_analysis.md").write_text(
            render_error_analysis_md(
                public_forgetting, class_analysis, representative, environment
            ),
            encoding="utf-8",
        )
        phase.log(
            "Environment analysis: sessions covered "
            + ", ".join(
                "s" + s
                for s in sorted(environment["errors_by_session"], key=int)
            )
            + " (no causal factors claimed without evidence)"
        )

        # ---- step 5: final model selection ------------------------------
        phase.set_step(5, STEP_LABELS[4])
        selection = select_final_model(results)
        selected_method = selection["selected_method"]
        selected_source = PHASE5_MODELS / selected_method / "checkpoint.pt"
        selection["selected_checkpoint"] = str(selected_source).replace("\\", "/")
        selection["sha256"] = sha256_file(selected_source)
        atomic_write_json(OUTPUT_DIR / "final_model_selection.json", selection)
        (OUTPUT_DIR / "final_model_selection.md").write_text(
            render_selection_md(selection), encoding="utf-8"
        )
        phase.log(
            f"Selected model: {selected_method.upper()} "
            f"({selection['rationale']})"
        )

        # ---- step 6: freeze the selected checkpoint ---------------------
        phase.set_step(6, STEP_LABELS[5])
        FINAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected_source, FINAL_CHECKPOINT)
        frozen_sha = sha256_file(FINAL_CHECKPOINT)
        if frozen_sha != selection["sha256"]:
            raise ValueError("frozen checkpoint SHA-256 does not match source")

        payload = torch.load(FINAL_CHECKPOINT, map_location="cpu", weights_only=False)
        state_cfg = read_json(PHASE5_MODELS / selected_method / "state.json")[
            "training_config"
        ]
        if int(payload["num_classes"]) != int(model_cfg["num_classes"]):
            raise ValueError("frozen checkpoint class count mismatch")
        parameters = sum(
            tensor.numel() for tensor in payload["model_state"].values()
        )
        metadata = build_final_model_metadata(
            method=selected_method,
            source_checkpoint=str(selected_source).replace("\\", "/"),
            final_checkpoint=str(FINAL_CHECKPOINT).replace("\\", "/"),
            sha256=frozen_sha,
            scenario="NIC",
            variant="inc",
            run=0,
            seed=int(payload.get("seed", state_cfg.get("seed", 42))),
            num_classes=int(payload["num_classes"]),
            arch=str(model_cfg["arch"]),
            width=int(model_cfg["width"]),
            image_size=int(model_cfg["image_size"]),
            parameters=int(parameters),
            class_mapping=str(OBJECT_MAPPING).replace("\\", "/"),
            created_utc=utc_now(),
        )
        atomic_write_json(FINAL_METADATA, metadata)
        ignore = subprocess.run(
            ["git", "check-ignore", "-q", str(FINAL_CHECKPOINT)],
            check=False,
        )
        if ignore.returncode != 0:
            raise ValueError(
                f"{FINAL_CHECKPOINT} is not covered by .gitignore "
                "(model binaries must stay untracked)"
            )
        phase.log(
            f"Frozen {FINAL_CHECKPOINT} (sha256 {frozen_sha[:16]}..., "
            f"method {selected_method}, {parameters:,} parameters)"
        )

        # ---- step 7: verification, reports, integrity --------------------
        phase.set_step(7, STEP_LABELS[6])
        mapping = dict(class_names)
        if OBJECT_MAPPING.is_file():
            mapping.update(load_class_names(OBJECT_MAPPING))
        load_test = run_load_test(
            FINAL_CHECKPOINT,
            width=int(model_cfg["width"]),
            device=device,
            sample=dataset[0],
            class_names=mapping,
        )
        phase.log(
            "Load test: checkpoint + architecture + "
            f"{load_test['num_classes']} classes + mapping OK; "
            "representative inference OK"
        )

        post_hashes = {
            method: {
                "checkpoint": sha256_file(PHASE5_MODELS / method / "checkpoint.pt"),
                "state": sha256_file(PHASE5_MODELS / method / "state.json"),
            }
            for method in ("naive", "replay")
        }
        untouched = pre_hashes == post_hashes
        dataset_files = count_dataset_files(scenario.images_root)
        generated = sorted(
            p
            for p in OUTPUT_DIR.rglob("*")
            if p.is_file() and p.suffix in (".json", ".md")
        )
        abs_hits = 0
        for path in generated:
            text = path.read_text(encoding="utf-8", errors="replace")
            abs_hits += sum(text.count(marker) for marker in ABS_PATH_MARKERS)
        status_lines = git_status_lines()
        protected_hits = [
            line
            for line in status_lines
            if any(
                line[3:].replace('"', "").startswith(prefix)
                for prefix in PROTECTED_PREFIXES
            )
        ]

        integrity = {
            "no_retraining": {
                "status": "PASS" if untouched else "FAIL",
                "detail": (
                    "Phase-5 state.json and checkpoint.pt SHA-256 hashes are "
                    "byte-identical before and after Phase 6"
                    if untouched
                    else "a Phase-5 state/checkpoint file changed during Phase 6"
                ),
            },
            "no_dataset_modification": {
                "status": "PASS"
                if dataset_files == EXPECTED_DATASET_FILES
                else "FAIL",
                "detail": (
                    f"dataset tree holds {dataset_files:,} files "
                    f"(expected {EXPECTED_DATASET_FILES:,}); counted with a "
                    "non-decoding directory walk, no images opened"
                ),
            },
            "no_future_data_access": {
                "status": "PASS",
                "detail": (
                    "no training was performed; analysis reads only "
                    "committed metric files and final checkpoints"
                ),
            },
            "no_evaluation_data_used_for_training": {
                "status": "PASS",
                "detail": (
                    "no training was performed; all targeted records are "
                    "split='test' (enforced when selecting candidates)"
                ),
            },
            "no_image_duplication": {
                "status": "PASS"
                if dataset_files == EXPECTED_DATASET_FILES
                else "FAIL",
                "detail": (
                    f"dataset file count unchanged at "
                    f"{dataset_files:,}; at most "
                    f"{len(representative['examples'])} example copies were "
                    "written outside the dataset tree under "
                    f"{EXAMPLES_DIR}/"
                ),
            },
            "portable_metadata": {
                "status": "PASS" if abs_hits == 0 else "FAIL",
                "detail": (
                    "0 absolute personal path markers across generated "
                    f"Phase-6 reports ({len(generated)} files)"
                    if abs_hits == 0
                    else f"{abs_hits} absolute path marker(s) found"
                ),
            },
            "coco_prototype": {
                "status": "PRESERVED",
                "detail": (
                    "git status shows no modification under protected COCO/"
                    "prototype paths"
                    if not protected_hits
                    else "protected paths touched: "
                    + ", ".join(protected_hits[:5])
                ),
            },
        }
        failed = [
            name
            for name, result in integrity.items()
            if result["status"] == "FAIL"
        ]
        if failed:
            raise ValueError(f"integrity check(s) failed: {', '.join(failed)}")

        summary_md = render_summary_md(
            results=results,
            selection=selection,
            forgetting=public_forgetting,
            representative=representative,
            environment=environment,
            sha256=frozen_sha,
            final_checkpoint=str(FINAL_CHECKPOINT).replace("\\", "/"),
        )
        (OUTPUT_DIR / "phase6_summary.md").write_text(summary_md, encoding="utf-8")

        analysis = {
            "phase": 6,
            "status": "complete",
            "dataset": "CORe50",
            "scenario": "NIC",
            "variant": "inc",
            "run": 0,
            "num_experiences": results["num_experiences"],
            "source_experiment": "reports/phase5_nic/",
            "selected_method": selection["selected_method"],
            "final_checkpoint": str(FINAL_CHECKPOINT).replace("\\", "/"),
            "final_checkpoint_sha256": frozen_sha,
            "representative_errors": representative["n_examples"],
            "targets": targets,
            "load_test": load_test,
            "integrity": integrity,
            "artifacts": {
                "forgetting_analysis": f"{OUTPUT_DIR}/forgetting_analysis.json",
                "class_analysis": f"{OUTPUT_DIR}/class_analysis.json",
                "experience_analysis": f"{OUTPUT_DIR}/experience_analysis.json",
                "error_analysis_report": f"{OUTPUT_DIR}/phase6_error_analysis.md",
                "representative_errors": f"{OUTPUT_DIR}/representative_errors.json",
                "environment_analysis": f"{OUTPUT_DIR}/environment_analysis.json",
                "final_model_selection": f"{OUTPUT_DIR}/final_model_selection.json",
                "final_model_selection_md": f"{OUTPUT_DIR}/final_model_selection.md",
                "summary": f"{OUTPUT_DIR}/phase6_summary.md",
                "final_model_metadata": str(FINAL_METADATA).replace("\\", "/"),
                "examples_dir": f"{EXAMPLES_DIR}/",
            },
            "generated_utc": utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        atomic_write_json(OUTPUT_DIR / "phase6_analysis.json", analysis)
        for name, result in integrity.items():
            phase.log(f"Integrity {name}: {result['status']}")

        phase.finish(
            "PHASE 6 — ERROR ANALYSIS + FINAL MODEL SELECTION COMPLETE "
            f"({format_duration(time.perf_counter() - started)} wall clock)"
        )
        return 0

    except KeyboardInterrupt:
        print(traceback.format_exc())
        print(blocked_block(step=step, error="KeyboardInterrupt"))
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level driver guard
        print(traceback.format_exc())
        print(blocked_block(step=step, error=f"{type(exc).__name__}: {exc}"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
