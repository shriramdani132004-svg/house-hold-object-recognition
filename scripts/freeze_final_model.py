"""Phase 40 — freeze the development-selected final model (ONE final artifact).

Selection (comparable, development-only): the per-experience full
development evaluation records (``dev_metrics.json``) compare every
experience on the SAME development set with the SAME cumulative-class
definition, and their argmax is taken as the development-selected
checkpoint. In this run the argmax is the final experience, so the
trainer's final ``checkpoint.pt`` (experience-best weights, restored at
the end of each experience) is the selected model — asserted, not
assumed: the script refuses to freeze unless the argmax equals the last
record.

The trainer's ``best_model.pt`` is NOT used: its global subset signal
compares development subsets with different numbers of seen classes
across experiences, which is biased toward early experiences (documented
in the frozen payload's ``selection`` block).

The historical Phase-5 baseline is verified against its pinned SHA-256
before AND after the freeze and lives on at
``models/continual/baseline_replay_phase5.pt``.

Usage:
    python scripts/freeze_final_model.py
    python scripts/freeze_final_model.py --candidate-dir models/continual/candidates/final_full_nic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpoint_schema import (  # noqa: E402
    FINAL_MODEL_PATH,
    load_final_model,
    validate_checkpoint_schema,
)
from src.training.config import config_fingerprint  # noqa: E402
from scripts.train_candidates import build_config  # noqa: E402

BASELINE_PATH = PROJECT_ROOT / "models/continual/baseline_replay_phase5.pt"
METADATA_PATH = PROJECT_ROOT / "models/continual/final_model.json"
FINAL_CONFIG = PROJECT_ROOT / "configs/final_training.yaml"
EXPECTED_BASELINE_SHA256 = (
    "b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351"
)
NUM_CLASSES = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(candidate_dir: Path, target: Path) -> dict:
    if not BASELINE_PATH.is_file():
        raise SystemExit(f"baseline preservation copy missing: {BASELINE_PATH}")
    baseline_sha = sha256_file(BASELINE_PATH)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise SystemExit(
            "baseline preservation copy does not match the pinned SHA-256\n"
            f"  expected {EXPECTED_BASELINE_SHA256}\n"
            f"  actual   {baseline_sha}\n"
            "refusing to freeze a pipeline whose historical baseline changed"
        )

    metrics_path = candidate_dir / "dev_metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(
            f"development records missing: {metrics_path}\n"
            "the final training run must produce per-experience records"
        )
    state_path = candidate_dir / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"training state missing: {state_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    records = json.loads(metrics_path.read_text(encoding="utf-8"))["records"]
    if not records:
        raise SystemExit("development records are empty; cannot freeze")
    if len(records) != 79:
        raise SystemExit(
            f"expected records for all 79 experiences, found {len(records)}"
        )
    overalls = [
        (int(r["experience_id"]), float(r["accuracy"]["overall"]))
        for r in records
        if r["accuracy"]["overall"] is not None
    ]
    if not overalls:
        raise SystemExit("development records contain no overall accuracies")
    argmax_experience, argmax_value = max(overalls, key=lambda item: item[1])
    last_experience = overalls[-1][0]
    if argmax_experience != last_experience:
        raise SystemExit(
            "the development-selected best experience "
            f"({argmax_experience}) is not the final experience "
            f"({last_experience}); only the final checkpoint persists, "
            "refusing to freeze a non-selected model"
        )
    if int(state["current_experience"]) != last_experience:
        raise SystemExit(
            f"state current_experience={state['current_experience']} != "
            f"final record experience {last_experience}"
        )

    source_path = candidate_dir / "checkpoint.pt"
    if not source_path.is_file():
        raise SystemExit(
            f"final training checkpoint missing: {source_path}\n"
            "the final training run must complete before freezing"
        )
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"training checkpoint is not a dict payload: {source_path}")
    for key in (
        "format_version",
        "method",
        "num_classes",
        "current_experience",
        "seed",
        "model_state",
        "arch",
        "image_size",
        "training_config",
        "config_fingerprint",
        "mapping_version",
        "mapping_checksum",
    ):
        if key not in payload:
            raise SystemExit(
                f"training checkpoint missing required key {key!r}: {source_path}"
            )
    if int(payload["num_classes"]) != NUM_CLASSES:
        raise SystemExit(
            f"checkpoint num_classes={payload['num_classes']} != {NUM_CLASSES}"
        )

    config_payload = yaml.safe_load(FINAL_CONFIG.read_text(encoding="utf-8"))
    config = build_config(config_payload, epochs_override=None)
    if payload["config_fingerprint"] != config_fingerprint(config):
        raise SystemExit(
            "checkpoint config fingerprint "
            f"{payload['config_fingerprint']} does not match "
            f"configs/final_training.yaml ({config_fingerprint(config)})"
        )

    excluded_best: dict[str, object] = {}
    best_path = candidate_dir / "best_model.pt"
    if best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if isinstance(best, dict) and "dev_value" in best:
            excluded_best = {
                "path": best_path.relative_to(PROJECT_ROOT).as_posix(),
                "subset_dev_value": float(best["dev_value"]),
                "reason": (
                    "global subset-best compares development subsets with "
                    "different numbers of seen classes across experiences "
                    "(biased toward early experiences); the comparable "
                    "full-development records are the selection metric"
                ),
            }

    last = records[-1]
    new_classes = []
    for record in reversed(records):
        if record["classes_introduced"]:
            new_classes = [int(c) for c in record["classes_introduced"]]
            break

    selection = {
        "signal": (
            "argmax of per-experience full development accuracy "
            "(train-only dev split, cumulative classes, identical set "
            "across experiences)"
        ),
        "dev_value": argmax_value,
        "selected_experience": argmax_experience,
        "records": len(records),
        "argmax_equals_final_record": True,
        "final_record_dev_value": overalls[-1][1],
        "last_introducing_experience": last["experience_id"]
        if last["classes_introduced"]
        else next(
            int(r["experience_id"])
            for r in reversed(records)
            if r["classes_introduced"]
        ),
        "last_introduced_classes": new_classes,
        "held_out_sessions_used": False,
        "excluded_signal": excluded_best,
    }

    payload = dict(payload)
    payload["selection"] = selection
    payload["provenance"] = {
        "source_checkpoint": source_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_sha256": sha256_file(source_path),
        "candidate_dir": candidate_dir.relative_to(PROJECT_ROOT).as_posix(),
        "baseline_preserved_at": BASELINE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # The in-memory payload is written first, then the ON-DISK artifact is
    # validated (the validator reads files, never memory).
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)

    problems = validate_checkpoint_schema(
        target,
        expected_num_classes=NUM_CLASSES,
        expected_method="replay",
    )
    if problems:
        raise SystemExit(
            "frozen checkpoint failed schema validation:\n  " + "\n  ".join(problems)
        )
    model, schema = load_final_model(target, num_classes=NUM_CLASSES)
    param_count = sum(p.numel() for p in model.parameters())

    metadata = {
        "phase": 7,
        "method": "replay",
        "source_checkpoint": str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "final_checkpoint": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "scenario": "NIC",
        "variant": "inc",
        "run": 0,
        "seed": int(payload["seed"]),
        "sha256": sha256_file(target),
        "dataset": "CORe50",
        "num_classes": NUM_CLASSES,
        "arch": str(payload["arch"]),
        "width": int(config.model_width),
        "image_size": int(payload["image_size"]),
        "parameters": int(param_count),
        "class_mapping": "data/raw/core50/metadata/object_mapping.json",
        "selection": selection,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    baseline_after = sha256_file(BASELINE_PATH)
    if baseline_after != EXPECTED_BASELINE_SHA256:
        raise SystemExit("baseline preservation copy changed during freeze")

    return {
        "frozen": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sha256": metadata["sha256"],
        "schema": schema,
        "parameters": param_count,
        "selection": selection,
        "baseline_verified": baseline_after,
        "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        default="models/continual/candidates/final_full_nic",
        help="final run directory holding best_model.pt / state.json / dev_metrics.json",
    )
    parser.add_argument(
        "--target",
        default=str(FINAL_MODEL_PATH),
        help="destination checkpoint (default: models/continual/final_model.pt)",
    )
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    if not candidate_dir.is_absolute():
        candidate_dir = PROJECT_ROOT / candidate_dir
    target = Path(args.target)
    if not target.is_absolute():
        target = PROJECT_ROOT / target

    report = freeze(candidate_dir, target)
    print("FREEZE OK")
    print(f"checkpoint: {report['frozen']}")
    print(f"sha256:     {report['sha256']}")
    print(f"arch:       {report['schema']['arch']} @ {report['schema']['image_size']}px")
    print(f"parameters: {report['parameters']}")
    print(
        "selection:  dev="
        f"{report['selection']['dev_value']:.4f} "
        f"exp={report['selection']['selected_experience']} "
        f"(argmax over {report['selection']['records']} records)"
    )
    print(f"baseline:   preserved ({report['baseline_verified'][:16]}...)")
    print(f"metadata:   {report['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
