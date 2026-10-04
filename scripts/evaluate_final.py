"""Phases 41-42 — the ONE held-out evaluation of the frozen final model.

Loads the official evaluation cache for the held-out sessions (s3/s7/s10,
44,972 references) exactly once and evaluates, in the same session, the
frozen final model and the preserved Phase-5 baseline for comparison.
The held-out set is read only here — never for training, selection,
debugging, or tuning — and no training happens after this script runs.

Metrics: top-1 and top-5 accuracy, per-class accuracy / precision /
recall / F1, macro P/R/F1, confusion matrix, old/new split (new = classes
introduced by the final training experience), per-session accuracy, and
the development-record-based continual measures (forgetting and average
incremental accuracy are computed from the legal development records,
because per-experience held-out evaluations are forbidden).

Usage:
    python scripts/evaluate_final.py
    python scripts/evaluate_final.py --candidate-dir models/continual/candidates/final_full_nic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_mapping import load_class_mapping  # noqa: E402
from src.evaluation.continual import EvalCache, atomic_write_json, compute_forgetting  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    average_incremental_accuracy,
    confusion_matrix,
    precision_recall_f1,
)
from src.training.checkpoint_schema import load_final_model  # noqa: E402

EVAL_CACHE_PATH = PROJECT_ROOT / "models/continual/phase5_nic/shared/eval_cache.pt"
FINAL_MODEL_PATH = PROJECT_ROOT / "models/continual/final_model.pt"
BASELINE_PATH = PROJECT_ROOT / "models/continual/baseline_replay_phase5.pt"
REPORT_PATH = PROJECT_ROOT / "reports/final_results/final_evaluation.json"
EXPECTED_BASELINE_SHA256 = (
    "b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351"
)
EXPECTED_TEST_SAMPLES = 44972
NUM_CLASSES = 50
_SESSION_RE = re.compile(r"(?:^|/)s(\d+)/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_checkpoint(
    model_path: Path, cache: EvalCache, device: str
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Full metric block for one checkpoint over the held-out cache."""
    model, schema = load_final_model(model_path, num_classes=NUM_CLASSES)
    recorded_size = schema.get("image_size")
    if recorded_size is None:
        # Legacy baseline payload predates the image_size key; it was
        # trained at 64px per the pinned Phase-5 configuration, which is
        # asserted against the cache here.
        if cache.image_size != 64:
            raise SystemExit(
                f"{model_path.name} is a legacy checkpoint without recorded "
                f"image_size but the cache is {cache.image_size}px (expected 64)"
            )
    elif int(recorded_size) != cache.image_size:
        raise SystemExit(
            f"{model_path.name} image_size={recorded_size} but the "
            f"evaluation cache is {cache.image_size}px — refusing to evaluate"
        )
    recorded_arch = schema.get("arch") or "small_cnn"

    all_logits: list[torch.Tensor] = []
    model.eval()
    batch_size = 256
    n = len(cache)
    t0 = time.monotonic()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            images = cache.float_batch(start, end).to(device)
            all_logits.append(model(images).cpu())
            if start == 0 or end % (batch_size * 20) == 0 or end == n:
                print(f"  forward {end}/{n}", flush=True)
    logits = torch.cat(all_logits)
    seconds = time.monotonic() - t0

    targets = cache.labels.to(torch.int64)
    top5_idx = logits.topk(5, dim=1).indices
    predictions = top5_idx[:, 0]
    top5_hits = top5_idx.eq(targets.unsqueeze(1)).any(dim=1)

    matrix = confusion_matrix(
        targets.tolist(), predictions.tolist(), NUM_CLASSES
    )
    prf = precision_recall_f1(matrix)
    per_class_acc = matrix.diag() / matrix.sum(dim=1).clamp(min=1)
    top1 = float(predictions.eq(targets).float().mean())
    top5 = float(top5_hits.float().mean())

    names = load_class_mapping().names
    if len(names) != NUM_CLASSES:
        raise SystemExit(f"expected {NUM_CLASSES} class names, got {len(names)}")

    return {
        "checkpoint": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "arch": recorded_arch,
        "image_size": int(recorded_size) if recorded_size is not None else 64,
        "n": n,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "per_class": {
            str(class_id): {
                "name": names[class_id],
                "accuracy": float(per_class_acc[class_id]),
                "precision": float(prf["per_class"][class_id]["precision"]),
                "recall": float(prf["per_class"][class_id]["recall"]),
                "f1": float(prf["per_class"][class_id]["f1"]),
                "support": int(matrix[class_id].sum()),
            }
            for class_id in range(NUM_CLASSES)
        },
        "macro": {
            "precision": float(prf["macro"]["precision"]),
            "recall": float(prf["macro"]["recall"]),
            "f1": float(prf["macro"]["f1"]),
        },
        "confusion_matrix": matrix.tolist(),
        "forward_seconds": round(seconds, 2),
    }, targets, predictions, top5_hits, cache.paths


def old_new_split(
    targets: torch.Tensor, predictions: torch.Tensor, new_classes: set[int]
) -> dict:
    if not new_classes:
        raise SystemExit("empty new-class set — records do not match NIC ordering")
    mask_new = torch.tensor(
        [int(t) in new_classes for t in targets.tolist()], dtype=torch.bool
    )
    mask_old = ~mask_new
    if int(mask_new.sum()) == 0 or int(mask_old.sum()) == 0:
        raise SystemExit("old/new split produced an empty side")
    return {
        "new_classes": sorted(int(c) for c in new_classes),
        "old_accuracy": float(predictions[mask_old].eq(targets[mask_old]).float().mean()),
        "new_accuracy": float(predictions[mask_new].eq(targets[mask_new]).float().mean()),
        "old_n": int(mask_old.sum()),
        "new_n": int(mask_new.sum()),
    }


def session_accuracy(
    targets: torch.Tensor, predictions: torch.Tensor, paths: tuple[str, ...]
) -> dict:
    sessions: dict[int, list[bool]] = {}
    for target, prediction, path in zip(targets, predictions, paths, strict=True):
        match = _SESSION_RE.search(path)
        if match is None:
            raise SystemExit(f"cannot parse session from cached path {path!r}")
        session = int(match.group(1))
        sessions.setdefault(session, []).append(bool(target == prediction))
    if set(sessions) != {3, 7, 10}:
        raise SystemExit(
            f"expected held-out sessions {{3, 7, 10}}, found {sorted(sessions)}"
        )
    return {
        str(session): {
            "accuracy": sum(hits) / len(hits),
            "n": len(hits),
        }
        for session, hits in sorted(sessions.items())
    }


def development_measures(candidate_dir: Path) -> dict:
    records_path = candidate_dir / "dev_metrics.json"
    if not records_path.is_file():
        raise SystemExit(f"development records missing: {records_path}")
    records = json.loads(records_path.read_text(encoding="utf-8"))["records"]
    if not records:
        raise SystemExit(f"development records empty: {records_path}")
    overalls = [
        r["accuracy"]["overall"]
        for r in records
        if r["accuracy"]["overall"] is not None
    ]
    forgetting = compute_forgetting(records)
    forged = [f for f in forgetting if f is not None]
    last = records[-1]
    introducing = [
        r for r in records if r["classes_introduced"]
    ]
    if not introducing:
        raise SystemExit("no development record introduces classes")
    newest = introducing[-1]
    return {
        "source": "development split (train-only dev records; held-out never used)",
        "records": len(records),
        "final_development_accuracy": overalls[-1],
        "best_development_accuracy": max(overalls),
        "development_forgetting": forged[-1],
        "development_average_incremental_accuracy": average_incremental_accuracy(overalls),
        "final_experience_id": int(last["experience_id"]),
        "old_new_definition": (
            "new = classes introduced by the most recent experience that "
            "introduced classes (experience "
            f"{int(newest['experience_id'])}); later experiences introduce "
            "none. old = the remaining classes."
        ),
        "new_classes": [int(c) for c in newest["classes_introduced"]],
        "new_classes_experience": int(newest["experience_id"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        default="models/continual/candidates/final_full_nic",
        help="final run directory with dev_metrics.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    if not candidate_dir.is_absolute():
        candidate_dir = PROJECT_ROOT / candidate_dir

    for required in (FINAL_MODEL_PATH, BASELINE_PATH, EVAL_CACHE_PATH):
        if not required.is_file():
            raise SystemExit(f"required artifact missing: {required}")
    baseline_sha = sha256_file(BASELINE_PATH)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise SystemExit(
            "preserved baseline does not match the pinned SHA-256 "
            f"({baseline_sha}); refusing to evaluate against a changed baseline"
        )

    print("loading held-out evaluation cache (sessions s3/s7/s10) once...", flush=True)
    cache = EvalCache.load(EVAL_CACHE_PATH)
    if len(cache) != EXPECTED_TEST_SAMPLES:
        raise SystemExit(
            f"evaluation cache holds {len(cache)} samples, "
            f"expected {EXPECTED_TEST_SAMPLES}"
        )
    print(f"cache: {len(cache)} samples @ {cache.image_size}px", flush=True)

    dev = development_measures(candidate_dir)
    new_classes = set(dev["new_classes"])

    print("evaluating frozen final model...", flush=True)
    final_metrics, f_targets, f_pred, f_top5, f_paths = evaluate_checkpoint(
        FINAL_MODEL_PATH, cache, args.device
    )
    print("evaluating preserved Phase-5 baseline...", flush=True)
    baseline_metrics, b_targets, b_pred, b_top5, b_paths = evaluate_checkpoint(
        BASELINE_PATH, cache, args.device
    )
    if not torch.equal(f_targets, b_targets) or f_paths != b_paths:
        raise SystemExit("final and baseline were evaluated on different data")

    final_metrics["old_new"] = old_new_split(f_targets, f_pred, new_classes)
    baseline_metrics["old_new"] = old_new_split(b_targets, b_pred, new_classes)
    final_metrics["per_session"] = session_accuracy(f_targets, f_pred, f_paths)
    baseline_metrics["per_session"] = session_accuracy(b_targets, b_pred, b_paths)

    report = {
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": (
            "single held-out evaluation session over the official test "
            "split (s3/s7/s10); never used for training or selection"
        ),
        "session_note": (
            "the first invocation of this script crashed after the final "
            "model's forward pass (legacy baseline payload lacked the "
            "image_size key) and was completed by a rerun; no training, "
            "tuning, or model selection happened between the two reads"
        ),
        "held_out_sessions": [3, 7, 10],
        "n": len(cache),
        "class_mapping_version": "core50-object-mapping-v1",
        "final": final_metrics,
        "baseline": baseline_metrics,
        "delta_final_minus_baseline": {
            "top1_accuracy": final_metrics["top1_accuracy"]
            - baseline_metrics["top1_accuracy"],
            "top5_accuracy": final_metrics["top5_accuracy"]
            - baseline_metrics["top5_accuracy"],
            "macro_f1": final_metrics["macro"]["f1"] - baseline_metrics["macro"]["f1"],
            "old_accuracy": final_metrics["old_new"]["old_accuracy"]
            - baseline_metrics["old_new"]["old_accuracy"],
            "new_accuracy": final_metrics["old_new"]["new_accuracy"]
            - baseline_metrics["old_new"]["new_accuracy"],
        },
        "development_measures": dev,
        "baseline_sha256": baseline_sha,
    }

    atomic_write_json(REPORT_PATH, report)
    print()
    print("=== ONE-SHOT HELD-OUT EVALUATION ===")
    print(
        f"final    top1={final_metrics['top1_accuracy']:.4f} "
        f"top5={final_metrics['top5_accuracy']:.4f} "
        f"macroF1={final_metrics['macro']['f1']:.4f}"
    )
    print(
        f"baseline top1={baseline_metrics['top1_accuracy']:.4f} "
        f"top5={baseline_metrics['top5_accuracy']:.4f} "
        f"macroF1={baseline_metrics['macro']['f1']:.4f}"
    )
    for key, value in report["delta_final_minus_baseline"].items():
        print(f"delta {key}: {value:+.4f}")
    print(
        f"dev measures: final={dev['final_development_accuracy']:.4f} "
        f"forgetting={dev['development_forgetting']:.4f} "
        f"avg_incr={dev['development_average_incremental_accuracy']:.4f}"
    )
    print(f"report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
