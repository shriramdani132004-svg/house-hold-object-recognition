"""Phase-6 error analysis and final model selection over Phase-5 artifacts.

Every function here is a pure, deterministic transformation of the
committed Phase-5 metric files (``reports/phase5_nic/``) plus, for the
targeted-error helpers, official test records loaded through the Phase-3
pipeline. Nothing in this module trains a model, writes into the dataset
tree, or fabricates missing values: unavailable data stays ``None`` and is
rendered as ``N-A``.

Metric conventions are read from the Phase-5 artifacts themselves
(``forgetting_definition`` / ``accuracy_definition``), so Phase 6 can never
diverge from what Phase 5 actually measured.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.data.continual import SampleRecord

NA = "N-A"

# Session facts, transcribed from the repository's own dataset documentation
# (data/raw/core50/DATASET_INFO.md). Only documented facts belong here.
SESSION_FACTS: dict[str, Any] = {
    "source": "data/raw/core50/DATASET_INFO.md",
    "sessions_total": 11,
    "test_sessions": [3, 7, 10],
    "train_sessions": [1, 2, 4, 5, 6, 8, 9, 11],
    "recording": "15-second videos at 20 fps (Kinect 2.0)",
    "indoor_outdoor": "8 indoor + 3 outdoor sessions per the official page; "
    "the repository does not record which sessions are outdoor",
    "varying_across_sessions": [
        "background",
        "lighting",
        "pose",
        "viewpoint",
        "hand occlusion",
    ],
}

ENV_FALLBACK_SENTENCE = (
    "The failure occurred on an example from a different "
    "session/environment; the specific causal factor cannot be "
    "established from this sample alone."
)


# ---------------------------------------------------------------------------
# Loading + validation of Phase-5 artifacts
# ---------------------------------------------------------------------------


def load_phase5_results(reports_dir: str | Path) -> dict[str, Any]:
    """Load and validate the committed Phase-5 metric artifacts.

    Raises ``ValueError`` when an artifact is missing or structurally
    invalid so the driver can stop with ``PHASE 6 BLOCKED`` instead of
    inventing data.
    """
    root = Path(reports_dir)
    required = {
        "naive": root / "naive_metrics.json",
        "replay": root / "replay_metrics.json",
        "summary": root / "experiment_summary.json",
        "per_class": root / "per_class_metrics.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(
            f"missing Phase-5 artifact(s) in {root}: {', '.join(missing)}"
        )

    naive = json.loads(required["naive"].read_text(encoding="utf-8"))
    replay = json.loads(required["replay"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    per_class = json.loads(required["per_class"].read_text(encoding="utf-8"))

    if naive.get("method") != "naive" or replay.get("method") != "replay":
        raise ValueError("metrics files do not match the naive/replay methods")
    if summary.get("status") != "complete":
        raise ValueError(f"experiment_summary status is {summary.get('status')!r}")

    for name, envelope in (("naive", naive), ("replay", replay)):
        records = envelope.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{name} metrics contain no records")
        ids = [r.get("experience_id") for r in records]
        if ids != list(range(len(records))):
            raise ValueError(f"{name} records are not contiguous 0..n")
    n = len(naive["records"])
    if len(replay["records"]) != n:
        raise ValueError("naive and replay record counts differ")
    if summary.get("num_experiences") != n:
        raise ValueError("summary experience count does not match records")

    histories = {
        "naive": per_class.get("naive"),
        "replay": per_class.get("replay"),
    }
    for name, hist in histories.items():
        if not isinstance(hist, dict) or len(hist) != len(
            per_class.get("class_names", hist)
        ):
            raise ValueError(f"per-class histories invalid for {name}")

    comparison = summary.get("comparison", {})
    for key in ("final_accuracy", "final_forgetting", "average_incremental_accuracy"):
        if key not in comparison:
            raise ValueError(f"summary comparison missing {key!r}")

    return {
        "reports_dir": root,
        "naive": naive,
        "replay": replay,
        "summary": summary,
        "per_class": per_class,
        "records": {"naive": naive["records"], "replay": replay["records"]},
        "histories": histories,
        "class_names": {
            str(k): str(v) for k, v in per_class.get("class_names", {}).items()
        },
        "num_experiences": n,
        "forgetting_definition": per_class.get("forgetting_definition")
        or summary.get("forgetting_definition", ""),
        "accuracy_definition": summary.get("accuracy_definition", ""),
        "average_incremental_definition": summary.get(
            "average_incremental_definition", ""
        ),
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# Per-class statistics (deterministic, fabrication-free)
# ---------------------------------------------------------------------------


def final_class_stats(history: Sequence[float | None]) -> dict[str, Any]:
    """Final accuracy / peak / forgetting for one class history.

    ``history[t]`` is the accuracy after experience ``t`` (``None`` before
    the class is introduced). Forgetting follows the Phase-5 definition:
    ``max prior accuracy - final accuracy``; ``None`` when the class has no
    prior measurement. Negative values (net improvement) are preserved.
    """
    measured = [(i, v) for i, v in enumerate(history) if v is not None]
    if not measured:
        return {
            "introduced_experience": None,
            "final_experience": None,
            "final_accuracy": None,
            "peak_accuracy": None,
            "forgetting": None,
            "n_measurements": 0,
        }
    intro = measured[0][0]
    final_index, final_value = measured[-1]
    prior = [v for i, v in measured[:-1]]
    peak = max(v for _, v in measured)
    forgetting = (max(prior) - final_value) if prior else None
    return {
        "introduced_experience": intro,
        "final_experience": final_index,
        "final_accuracy": final_value,
        "peak_accuracy": peak,
        "forgetting": forgetting,
        "n_measurements": len(measured),
    }


def _rank_rows(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    *,
    descending: bool,
    limit: int,
    tiebreak: str = "label",
) -> list[dict[str, Any]]:
    usable = [dict(r) for r in rows if r.get(key) is not None]
    usable.sort(key=lambda r: (-r[key] if descending else r[key], r[tiebreak]))
    return usable[:limit]


def build_forgetting_analysis(results: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate + per-class + per-experience forgetting findings."""
    histories = results["histories"]
    names = results["class_names"]
    comparison = results["comparison"]

    per_class: dict[str, dict[str, dict[str, Any]]] = {}
    for method in ("naive", "replay"):
        per_class[method] = {
            label: final_class_stats(hist) for label, hist in histories[method].items()
        }

    def rows(method: str, stat_key: str, limit: int, descending: bool):
        out = []
        for label in sorted(per_class[method], key=int):
            stats = per_class[method][label]
            out.append(
                {
                    "label": int(label),
                    "name": names.get(label, f"class_{label}"),
                    "final_accuracy": stats["final_accuracy"],
                    "peak_accuracy": stats["peak_accuracy"],
                    "forgetting": stats["forgetting"],
                    stat_key: stats[stat_key],
                }
            )
        return out

    records = results["records"]

    def experience_increases(method: str) -> list[dict[str, Any]]:
        out = []
        prev: float | None = None
        for record in records[method]:
            value = record.get("forgetting")
            increase = None
            if value is not None and prev is not None:
                increase = value - prev
            out.append(
                {
                    "experience_id": record["experience_id"],
                    "forgetting": value,
                    "increase": increase,
                }
            )
            if value is not None:
                prev = value
        return out

    naive_inc = experience_increases("naive")
    replay_inc = experience_increases("replay")
    exp_rows = []
    for n_row, r_row in zip(naive_inc, replay_inc):
        exp_rows.append(
            {
                "experience_id": n_row["experience_id"],
                "naive_forgetting": n_row["forgetting"],
                "naive_increase": n_row["increase"],
                "replay_forgetting": r_row["forgetting"],
                "replay_increase": r_row["increase"],
            }
        )
    top_experiences = sorted(
        [e for e in exp_rows if e["naive_increase"] is not None],
        key=lambda e: (-e["naive_increase"], e["experience_id"]),
    )[:8]

    naive_rows = rows("naive", "forgetting", len(histories["naive"]), False)
    replay_rows = rows("replay", "forgetting", len(histories["replay"]), False)

    diff_rows = []
    for label in sorted(per_class["naive"], key=int):
        n_final = per_class["naive"][label]["final_accuracy"]
        r_final = per_class["replay"][label]["final_accuracy"]
        if n_final is None or r_final is None:
            continue
        diff_rows.append(
            {
                "label": int(label),
                "name": names.get(label, f"class_{label}"),
                "naive_final_accuracy": n_final,
                "replay_final_accuracy": r_final,
                "difference": r_final - n_final,
            }
        )

    return {
        "phase": 6,
        "source": "reports/phase5_nic",
        "forgetting_definition": results["forgetting_definition"],
        "accuracy_definition": results["accuracy_definition"],
        "aggregate": {
            "naive": {
                "final_mean_forgetting": comparison["final_forgetting"]["naive"],
                "final_overall_accuracy": comparison["final_accuracy"]["naive"],
                "average_incremental_accuracy": comparison[
                    "average_incremental_accuracy"
                ]["naive"],
            },
            "replay": {
                "final_mean_forgetting": comparison["final_forgetting"]["replay"],
                "final_overall_accuracy": comparison["final_accuracy"]["replay"],
                "average_incremental_accuracy": comparison[
                    "average_incremental_accuracy"
                ]["replay"],
            },
            "delta_replay_minus_naive": {
                "final_accuracy": (
                    comparison["final_accuracy"]["replay"]
                    - comparison["final_accuracy"]["naive"]
                ),
                "final_forgetting": (
                    comparison["final_forgetting"]["replay"]
                    - comparison["final_forgetting"]["naive"]
                ),
                "average_incremental_accuracy": (
                    comparison["average_incremental_accuracy"]["replay"]
                    - comparison["average_incremental_accuracy"]["naive"]
                ),
            },
        },
        "top_forgetting_classes": {
            "naive": _rank_rows(naive_rows, "forgetting", descending=True, limit=10),
            "replay": _rank_rows(replay_rows, "forgetting", descending=True, limit=10),
        },
        "most_retained_classes": {
            "naive": _rank_rows(
                naive_rows, "final_accuracy", descending=True, limit=10
            ),
            "replay": _rank_rows(
                replay_rows, "final_accuracy", descending=True, limit=10
            ),
        },
        "lowest_final_accuracy_classes": {
            "naive": _rank_rows(
                naive_rows, "final_accuracy", descending=False, limit=10
            ),
            "replay": _rank_rows(
                replay_rows, "final_accuracy", descending=False, limit=10
            ),
        },
        "largest_naive_replay_differences": sorted(
            diff_rows, key=lambda r: (-abs(r["difference"]), r["label"])
        )[:10],
        "experiences_with_largest_forgetting_increase": top_experiences,
        "_per_class": per_class,
    }


def build_class_analysis(results: Mapping[str, Any]) -> dict[str, Any]:
    """One row per object identity with both methods' final statistics."""
    names = results["class_names"]
    histories = results["histories"]
    intro: dict[str, int | None] = {}
    categories: dict[str, str | None] = {}
    for record in results["records"]["naive"]:
        for label in record.get("classes_introduced", []):
            intro[str(label)] = record["experience_id"]

    rows = []
    for label in sorted(histories["naive"], key=int):
        n_stats = final_class_stats(histories["naive"][label])
        r_stats = final_class_stats(histories["replay"].get(label, []))
        n_final, r_final = n_stats["final_accuracy"], r_stats["final_accuracy"]
        rows.append(
            {
                "label": int(label),
                "name": names.get(label, f"class_{label}"),
                "category": categories.get(label),
                "introduced_experience": intro.get(label),
                "naive": n_stats,
                "replay": r_stats,
                "difference_replay_minus_naive": (
                    r_final - n_final
                    if n_final is not None and r_final is not None
                    else None
                ),
            }
        )
    return {
        "phase": 6,
        "source": "reports/phase5_nic",
        "definition": results["per_class"].get("definition", ""),
        "classes": rows,
    }


def build_experience_analysis(results: Mapping[str, Any]) -> dict[str, Any]:
    """Per-experience overall/old/new accuracy and forgetting, both methods."""
    records = results["records"]
    rows = []
    for n_rec, r_rec in zip(records["naive"], records["replay"]):
        rows.append(
            {
                "experience_id": n_rec["experience_id"],
                "naive_overall": n_rec["accuracy"]["overall"],
                "replay_overall": r_rec["accuracy"]["overall"],
                "naive_old": n_rec["accuracy"]["old"],
                "replay_old": r_rec["accuracy"]["old"],
                "naive_new": n_rec["accuracy"]["new"],
                "replay_new": r_rec["accuracy"]["new"],
                "naive_forgetting": n_rec.get("forgetting"),
                "replay_forgetting": r_rec.get("forgetting"),
            }
        )
    return {
        "phase": 6,
        "source": "reports/phase5_nic",
        "experiences": rows,
    }


# ---------------------------------------------------------------------------
# Final model selection (deterministic, evidence-driven)
# ---------------------------------------------------------------------------

PRIMARY_CRITERIA = [
    "final_accuracy",
    "final_forgetting",
    "old_knowledge_retention",
    "average_incremental_accuracy",
]


def select_final_model(results: Mapping[str, Any]) -> dict[str, Any]:
    """Select the final checkpoint from measured Phase-5 results only.

    Deterministic rule: compare replay vs naive on the four primary
    measured criteria (final accuracy — higher better; final mean
    forgetting — lower better; old-knowledge retention (final old-class
    accuracy) — higher better; average incremental accuracy — higher
    better). The method that is at least as good on every primary criterion
    and strictly better on at least one is selected. If neither method
    dominates, the tiebreak order above decides and the tradeoff is
    recorded explicitly.
    """
    comparison = results["comparison"]
    naive_records = results["records"]["naive"]
    replay_records = results["records"]["replay"]
    if not naive_records or not replay_records:
        raise ValueError("both methods need at least one record")

    values = {
        "final_accuracy": (
            comparison["final_accuracy"]["naive"],
            comparison["final_accuracy"]["replay"],
        ),
        "final_forgetting": (
            comparison["final_forgetting"]["naive"],
            comparison["final_forgetting"]["replay"],
        ),
        "old_knowledge_retention": (
            naive_records[-1]["accuracy"]["old"],
            replay_records[-1]["accuracy"]["old"],
        ),
        "average_incremental_accuracy": (
            comparison["average_incremental_accuracy"]["naive"],
            comparison["average_incremental_accuracy"]["replay"],
        ),
    }
    lower_is_better = {"final_forgetting"}

    criteria_rows = []
    replay_weak = False
    for criterion in PRIMARY_CRITERIA:
        naive_value, replay_value = values[criterion]
        if naive_value is None or replay_value is None:
            verdict = "incomparable (missing measurement)"
            replay_better = False
        elif criterion in lower_is_better:
            replay_better = replay_value < naive_value
            verdict = "replay better" if replay_better else (
                "naive better" if replay_value > naive_value else "equal"
            )
        else:
            replay_better = replay_value > naive_value
            verdict = "replay better" if replay_better else (
                "naive better" if replay_value < naive_value else "equal"
            )
        if verdict == "naive better":
            replay_weak = True
        criteria_rows.append(
            {
                "criterion": criterion,
                "naive": naive_value,
                "replay": replay_value,
                "verdict": verdict,
            }
        )

    replay_dominates = any(r["verdict"] == "replay better" for r in criteria_rows)
    if replay_dominates and not replay_weak:
        selected = "replay"
        rationale = (
            "Replay is at least as good as naive on every primary measured "
            "criterion and strictly better on at least one; select replay."
        )
    else:
        selected = "naive" if replay_weak else "replay"
        rationale = (
            "Primary criteria conflict; resolved by the documented tiebreak "
            "order (final accuracy, final forgetting, old-knowledge "
            "retention, average incremental accuracy)."
        )

    rationale_detail = "; ".join(
        f"{row['criterion']}: naive {row['naive']:.4f} vs replay "
        f"{row['replay']:.4f} ({row['verdict']})"
        if isinstance(row["naive"], (int, float))
        and isinstance(row["replay"], (int, float))
        else f"{row['criterion']}: {row['verdict']}"
        for row in criteria_rows
    )

    summary = results["summary"]
    return {
        "phase": 6,
        "selected_method": selected,
        "rationale": rationale,
        "rationale_detail": rationale_detail,
        "primary_criteria": list(PRIMARY_CRITERIA),
        "criteria": criteria_rows,
        "tradeoffs": [
            "Both methods share the identical architecture "
            f"({summary.get('model', 'SmallConvNet')}), so final model size "
            "and inference cost are the same; the replay checkpoint file is "
            "larger only because its payload carries the bounded replay "
            "memory.",
            "Replay training took longer per experience (replay batch "
            "loading); training time is not a quality criterion for the "
            "frozen inference model.",
        ],
        "naive_metrics": {
            "final_accuracy": comparison["final_accuracy"]["naive"],
            "final_forgetting": comparison["final_forgetting"]["naive"],
            "average_incremental_accuracy": comparison[
                "average_incremental_accuracy"
            ]["naive"],
            "final_old_accuracy": values["old_knowledge_retention"][0],
            "training_time_seconds": summary["naive"]["training_time_seconds"],
        },
        "replay_metrics": {
            "final_accuracy": comparison["final_accuracy"]["replay"],
            "final_forgetting": comparison["final_forgetting"]["replay"],
            "average_incremental_accuracy": comparison[
                "average_incremental_accuracy"
            ]["replay"],
            "final_old_accuracy": values["old_knowledge_retention"][1],
            "training_time_seconds": summary["replay"]["training_time_seconds"],
            "replay_capacity": summary["replay"].get("replay_capacity"),
        },
        "source_experiment": {
            "dataset": summary.get("dataset"),
            "scenario": summary.get("scenario"),
            "variant": summary.get("variant"),
            "run": summary.get("run"),
            "seed": summary.get("seed"),
            "num_experiences": summary.get("num_experiences"),
            "artifacts": "reports/phase5_nic/",
        },
    }


# ---------------------------------------------------------------------------
# Targeted representative-error selection (deterministic)
# ---------------------------------------------------------------------------


def select_target_classes(
    forgetting_analysis: Mapping[str, Any],
    *,
    per_bucket: int = 3,
    max_classes: int = 8,
) -> list[int]:
    """Deterministic class targets for targeted inference.

    Priority buckets (first-seen order wins up to ``max_classes``):
    1. highest naive forgetting, 2. lowest naive final accuracy,
    3. largest naive-vs-replay final difference, 4. lowest replay final
    accuracy. The result is sorted by label for stable quotas.
    """
    chosen: list[int] = []

    def add(rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            label = int(row["label"])
            if label not in chosen:
                chosen.append(label)

    add(forgetting_analysis["top_forgetting_classes"]["naive"][:per_bucket])
    add(forgetting_analysis["lowest_final_accuracy_classes"]["naive"][:per_bucket])
    add(forgetting_analysis["largest_naive_replay_differences"][:per_bucket])
    add(forgetting_analysis["lowest_final_accuracy_classes"]["replay"][:per_bucket])
    return sorted(chosen[:max_classes])


def select_candidate_records(
    class_labels: Sequence[int],
    evaluation_records: Sequence[SampleRecord],
    *,
    max_total: int = 100,
) -> list[SampleRecord]:
    """Up to ``max_total`` evaluation records over ``class_labels``.

    Deterministic: records are grouped per class and, within a class,
    round-robined across recording sessions (ascending) so the sample
    covers every recorded test session when the quota allows; each
    session's records are taken in relative-path order. Quota =
    ceil(max_total / classes); the final list is sorted by
    (label, relative_path).
    """
    if not class_labels:
        return []
    by_label: dict[int, list[SampleRecord]] = {}
    wanted = {int(v) for v in class_labels}
    for record in evaluation_records:
        if record.split != "test":
            raise ValueError(
                f"candidate source must be evaluation data, got split="
                f"{record.split!r} ({record.relative_path})"
            )
        if int(record.label) in wanted:
            by_label.setdefault(int(record.label), []).append(record)

    quota = max(1, -(-int(max_total) // len(class_labels)))
    picked: list[SampleRecord] = []
    for label in sorted(wanted):
        by_session: dict[int, list[SampleRecord]] = {}
        for record in by_label.get(label, []):
            by_session.setdefault(int(record.session_id), []).append(record)
        for session_records in by_session.values():
            session_records.sort(key=lambda r: r.relative_path)
        queues = [by_session[s] for s in sorted(by_session)]
        label_pick: list[SampleRecord] = []
        while len(label_pick) < quota and any(queues):
            for queue in queues:
                if len(label_pick) >= quota:
                    break
                if queue:
                    label_pick.append(queue.pop(0))
        picked.extend(label_pick)
    picked.sort(key=lambda r: (int(r.label), r.relative_path))
    if len(picked) > max_total:
        picked = picked[:max_total]
    return picked


def classify_records(
    model: Any,
    dataset: Any,
    *,
    device: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the frozen classifier over ``dataset`` (no_grad, softmax conf)."""
    import torch

    model.eval()
    outputs: list[dict[str, Any]] = []
    total = len(dataset)
    with torch.inference_mode():
        for index in range(total):
            features, label = dataset[index]
            logits = model(features.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0]
            confidence, prediction = probs.max(0)
            outputs.append(
                {
                    "index": index,
                    "true": int(label),
                    "predicted": int(prediction),
                    "confidence": float(confidence),
                }
            )
            if on_progress is not None:
                on_progress(index + 1, total)
    return outputs


def build_representative_errors(
    candidates: Sequence[SampleRecord],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    class_names: Mapping[str, str],
    *,
    max_examples: int = 12,
    max_per_class: int = 2,
) -> dict[str, Any]:
    """Pick a small, deterministic, session-balanced error set.

    Ranking: method-disagreement errors (one method right, the other
    wrong) first, then errors where both methods fail; ties broken by
    (label, session, relative_path). First pass picks up to
    ``ceil(max_examples / sessions)`` rows per session in ranked order
    (so every recorded test session is represented when errors exist),
    then remaining slots are filled in ranked order; the per-class cap
    keeps the set diverse. If fewer than ``max_examples`` errors exist,
    all of them are returned — nothing is fabricated.
    """
    methods = sorted(predictions)
    rows = []
    n_errors = {"disagreement": 0, "both_wrong": 0}
    per_method_errors = {m: 0 for m in methods}
    for index, record in enumerate(candidates):
        states = {}
        for method in methods:
            pred = predictions[method][index]
            ok = pred["predicted"] == pred["true"]
            states[method] = {
                "predicted": pred["predicted"],
                "confidence": pred["confidence"],
                "correct": ok,
            }
            if not ok:
                per_method_errors[method] += 1
        correct_flags = [states[m]["correct"] for m in methods]
        if all(correct_flags):
            continue
        disagreement = not all(correct_flags) and any(correct_flags)
        rank = 0 if disagreement else 1
        n_errors["disagreement" if disagreement else "both_wrong"] += 1
        rows.append(
            {
                "rank": rank,
                "label": int(record.label),
                "record": record,
                "states": states,
                "disagreement": disagreement,
            }
        )

    rows.sort(
        key=lambda r: (
            r["rank"],
            r["label"],
            int(r["record"].session_id),
            r["record"].relative_path,
        )
    )

    def make_example(row: Mapping[str, Any]) -> dict[str, Any]:
        record = row["record"]
        label = int(record.label)
        name = class_names.get(str(label), f"class_{label}")
        failing = [m for m in methods if not row["states"][m]["correct"]]
        return {
            "relative_path": record.relative_path,
            "true_label": label,
            "true_name": name,
            "true_object_name": record.object_name,
            "category": record.category_name,
            "session": int(record.session_id),
            "object_id": int(record.object_id),
            "split": record.split,
            "checkpoint_experience": "after final experience (id 78 of 79)",
            "disagreement": row["disagreement"],
            "showcase_method": "+".join(failing) if failing else "none",
            "predictions": {
                m: {
                    "predicted": row["states"][m]["predicted"],
                    "predicted_name": class_names.get(
                        str(row["states"][m]["predicted"]),
                        f"class_{row['states'][m]['predicted']}",
                    ),
                    "confidence": row["states"][m]["confidence"],
                    "correct": row["states"][m]["correct"],
                }
                for m in methods
            },
        }

    chosen: list[dict[str, Any]] = []
    chosen_paths: set[str] = set()
    per_class_count: dict[int, int] = {}

    def try_take(row: Mapping[str, Any]) -> bool:
        label = row["label"]
        if len(chosen) >= max_examples:
            return False
        if per_class_count.get(label, 0) >= max_per_class:
            return False
        path = row["record"].relative_path
        if path in chosen_paths:
            return False
        per_class_count[label] = per_class_count.get(label, 0) + 1
        chosen_paths.add(path)
        chosen.append(make_example(row))
        return True

    sessions = sorted({int(r["record"].session_id) for r in rows})
    n_sessions = max(len(sessions), 1)
    if sessions:
        per_session_quota = -(-max_examples // len(sessions))
        for session in sessions:
            taken = 0
            for row in rows:
                if taken >= per_session_quota:
                    break
                if int(row["record"].session_id) != session:
                    continue
                if try_take(row):
                    taken += 1
    for row in rows:
        if len(chosen) >= max_examples:
            break
        try_take(row)

    return {
        "phase": 6,
        "candidates_evaluated": len(candidates),
        "candidates_per_method": dict(per_method_errors),
        "total_errors": n_errors,
        "selection_rule": (
            "disagreement errors first, then both-method errors; ties by "
            "(label, session, relative_path); per-session balanced picks "
            f"(ceil({max_examples}/{n_sessions}) per session) then "
            f"ranked fill; per-class cap {max_per_class}; cap "
            f"{max_examples}; all available errors returned when fewer "
            "exist"
        ),
        "examples": chosen,
        "n_examples": len(chosen),
    }


def build_environment_analysis(
    representative: Mapping[str, Any],
    *,
    session_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Session/environment observations for the selected examples.

    OBSERVED entries only restate measured facts (session, labels,
    predictions). POSSIBLE factors are limited to attributes the dataset
    documentation says vary across sessions, always with the fallback
    uncertainty sentence; no causal claim is ever asserted.
    """
    facts = dict(session_facts or SESSION_FACTS)
    examples = representative.get("examples", [])
    errors_by_session: dict[str, int] = {}
    observations = []
    for example in examples:
        session = example["session"]
        key = str(session)
        errors_by_session[key] = errors_by_session.get(key, 0) + 1
        predictions = example["predictions"]
        parts = [
            f"{m} predicted {p['predicted_name']} "
            f"(conf {p['confidence']:.3f}, {'correct' if p['correct'] else 'wrong'})"
            for m, p in sorted(predictions.items())
        ]
        observed = (
            f"session s{session} -> class {example['true_name']} -> "
            f"true label {example['true_label']}; " + "; ".join(parts)
        )
        if session not in facts["test_sessions"]:
            possible = (
                f"session s{session} is not one of the documented fixed test "
                f"sessions {facts['test_sessions']}; the specific causal "
                "factor cannot be established from this sample alone."
            )
        else:
            attributes = ", ".join(facts["varying_across_sessions"])
            possible = (
                f"The example is from test session s{session}; per the "
                "dataset documentation sessions vary in "
                f"{attributes}, so session shift is a plausible background "
                "factor. However, the specific causal factor cannot be "
                "established from this sample alone."
            )
        observations.append(
            {
                "relative_path": example["relative_path"],
                "session": session,
                "true_name": example["true_name"],
                "observed": observed,
                "possible_factor": possible,
            }
        )

    overall_observed = [
        "All evaluated examples come from the fixed official test sessions "
        f"{facts['test_sessions']}; those sessions are never used for "
        f"training (training sessions: {facts['train_sessions']}).",
        f"Selected example sessions: {sorted(errors_by_session, key=int) or []}.",
    ]
    overall_possible = [ENV_FALLBACK_SENTENCE]

    return {
        "phase": 6,
        "session_facts": facts,
        "errors_by_session": errors_by_session,
        "observations": observations,
        "observed_evidence": overall_observed,
        "possible_factors": overall_possible,
        "causal_claim_policy": (
            "No causal environmental claim is made without evidence; the "
            "fallback sentence is used whenever a specific factor cannot be "
            "isolated."
        ),
    }


# ---------------------------------------------------------------------------
# Checkpoint freeze helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_final_model_metadata(
    *,
    method: str,
    source_checkpoint: str,
    final_checkpoint: str,
    sha256: str,
    scenario: str,
    variant: str,
    run: int,
    seed: int,
    dataset: str = "CORe50",
    num_classes: int | None = None,
    arch: str | None = None,
    width: int | None = None,
    image_size: int | None = None,
    parameters: int | None = None,
    class_mapping: str | None = None,
    created_utc: str | None = None,
    phase: int = 6,
) -> dict[str, Any]:
    """Portable frozen-checkpoint metadata (project-relative paths only)."""
    for label, value in (
        ("source_checkpoint", source_checkpoint),
        ("final_checkpoint", final_checkpoint),
    ):
        if Path(value).is_absolute() or ":" in value.replace("://", ""):
            raise ValueError(f"{label} must be project-relative, got {value!r}")
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError(f"sha256 must be 64 lowercase hex chars, got {sha256!r}")
    payload: dict[str, Any] = {
        "phase": phase,
        "method": method,
        "source_checkpoint": source_checkpoint,
        "final_checkpoint": final_checkpoint,
        "scenario": scenario,
        "variant": variant,
        "run": run,
        "seed": seed,
        "sha256": sha256,
        "dataset": dataset,
    }
    optional = {
        "num_classes": num_classes,
        "arch": arch,
        "width": width,
        "image_size": image_size,
        "parameters": parameters,
        "class_mapping": class_mapping,
        "created_utc": created_utc,
    }
    payload.update({k: v for k, v in optional.items() if v is not None})
    return payload


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else NA


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def build_experiment_placeholder() -> dict[str, Any]:
    """Empty but structurally complete inputs for interim report renders.

    Used only for the step-2 draft of ``phase6_error_analysis.md`` before
    targeted inference exists; step 4 overwrites it with real results.
    """
    return {
        "candidates_evaluated": 0,
        "candidates_per_method": {"naive": 0, "replay": 0},
        "total_errors": {"disagreement": 0, "both_wrong": 0},
        "examples": [],
        "n_examples": 0,
        "errors_by_session": {},
        "observed_evidence": [],
        "observations": [],
        "possible_factors": [],
    }


def render_error_analysis_md(
    forgetting: Mapping[str, Any],
    class_analysis: Mapping[str, Any],
    representative: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> str:
    agg = forgetting["aggregate"]
    top_f = forgetting["top_forgetting_classes"]["naive"]
    low = forgetting["lowest_final_accuracy_classes"]["naive"]
    diffs = forgetting["largest_naive_replay_differences"]
    exps = forgetting["experiences_with_largest_forgetting_increase"]

    lines = [
        "# Phase 6 — Focused Error + Forgetting Analysis",
        "",
        "Source: measured Phase-5 artifacts (`reports/phase5_nic/`), "
        "Scenario NIC / variant inc / run 0, 79 experiences. No retraining "
        "was performed; every number below is read from the committed "
        "metrics.",
        "",
        "## OBSERVED — aggregate effect of replay",
        "",
        *_table(
            ["Metric", "Naive", "Replay", "Delta (replay - naive)"],
            [
                [
                    "Final overall accuracy",
                    _pct(agg["naive"]["final_overall_accuracy"]),
                    _pct(agg["replay"]["final_overall_accuracy"]),
                    _pct(agg["delta_replay_minus_naive"]["final_accuracy"]),
                ],
                [
                    "Final mean forgetting (lower better)",
                    _pct(agg["naive"]["final_mean_forgetting"]),
                    _pct(agg["replay"]["final_mean_forgetting"]),
                    _pct(agg["delta_replay_minus_naive"]["final_forgetting"]),
                ],
                [
                    "Average incremental accuracy",
                    _pct(agg["naive"]["average_incremental_accuracy"]),
                    _pct(agg["replay"]["average_incremental_accuracy"]),
                    _pct(
                        agg["delta_replay_minus_naive"][
                            "average_incremental_accuracy"
                        ]
                    ),
                ],
            ],
        ),
        "",
        f"Definition in effect: {forgetting['forgetting_definition']}",
        "",
        "## OBSERVED — highest-forgetting classes (naive)",
        "",
        *_table(
            ["Label", "Name", "Naive forgetting", "Naive final acc",
             "Replay final acc"],
            [
                [
                    str(row["label"]),
                    row["name"],
                    _pct(row["forgetting"]),
                    _pct(row["final_accuracy"]),
                    _pct(
                        next(
                            (
                                r["replay"]["final_accuracy"]
                                for r in class_analysis["classes"]
                                if r["label"] == row["label"]
                            ),
                            None,
                        )
                    ),
                ]
                for row in top_f
            ],
        ),
        "",
        "## OBSERVED — lowest final accuracy (naive)",
        "",
        *_table(
            ["Label", "Name", "Naive final acc", "Naive forgetting"],
            [
                [
                    str(row["label"]),
                    row["name"],
                    _pct(row["final_accuracy"]),
                    _pct(row["forgetting"]),
                ]
                for row in low
            ],
        ),
        "",
        "## OBSERVED — largest naive/replay differences (final accuracy)",
        "",
        *_table(
            ["Label", "Name", "Naive final", "Replay final", "Difference"],
            [
                [
                    str(row["label"]),
                    row["name"],
                    _pct(row["naive_final_accuracy"]),
                    _pct(row["replay_final_accuracy"]),
                    _pct(row["difference"]),
                ]
                for row in diffs
            ],
        ),
        "",
        "## OBSERVED — experiences with the largest forgetting increase "
        "(naive)",
        "",
        *_table(
            ["Experience", "Naive forgetting", "Naive increase",
             "Replay forgetting", "Replay increase"],
            [
                [
                    str(row["experience_id"]),
                    _pct(row["naive_forgetting"]),
                    _pct(row["naive_increase"]),
                    _pct(row["replay_forgetting"]),
                    _pct(row["replay_increase"]),
                ]
                for row in exps
            ],
        ),
        "",
        "## OBSERVED — representative errors",
        "",
        f"Candidates evaluated: {representative['candidates_evaluated']} "
        f"evaluation images per method; representative examples kept: "
        f"{representative['n_examples']} "
        f"(errors seen: {representative['total_errors']}).",
        "",
        *_table(
            ["Image (dataset-relative)", "Session", "True", "Naive prediction",
             "Replay prediction"],
            [
                [
                    ex["relative_path"],
                    f"s{ex['session']}",
                    ex["true_name"],
                    f"{ex['predictions']['naive']['predicted_name']} "
                    f"({'ok' if ex['predictions']['naive']['correct'] else 'wrong'}, "
                    f"{ex['predictions']['naive']['confidence']:.3f})",
                    f"{ex['predictions']['replay']['predicted_name']} "
                    f"({'ok' if ex['predictions']['replay']['correct'] else 'wrong'}, "
                    f"{ex['predictions']['replay']['confidence']:.3f})",
                ]
                for ex in representative["examples"]
            ],
        ),
        "",
        "## OBSERVED EVIDENCE — sessions/environment",
        "",
        *[f"- {line}" for line in environment["observed_evidence"]],
        "",
        "Per-example observations:",
        "",
        *[
            f"- Session s{obs['session']} -> {obs['true_name']} -> "
            f"{obs['observed']}"
            for obs in environment["observations"]
        ],
        "",
        "## POSSIBLE FACTORS",
        "",
        *[f"- {line}" for line in environment["possible_factors"]],
        "",
    ]
    return "\n".join(lines)


def render_selection_md(selection: Mapping[str, Any]) -> str:
    naive = selection["naive_metrics"]
    replay = selection["replay_metrics"]
    lines = [
        "# Phase 6 — Final Model Selection",
        "",
        "Selected method: "
        f"**{selection['selected_method'].upper()}**",
        "",
        f"Rationale: {selection['rationale']}",
        "",
        f"Measured detail: {selection['rationale_detail']}.",
        "",
        "## Primary measured criteria",
        "",
        *_table(
            ["Criterion", "Naive", "Replay", "Verdict"],
            [
                [
                    row["criterion"],
                    _pct(row["naive"]),
                    _pct(row["replay"]),
                    row["verdict"],
                ]
                for row in selection["criteria"]
            ],
        ),
        "",
        "## Full comparison",
        "",
        *_table(
            ["Metric", "Naive", "Replay"],
            [
                ["Final accuracy", _pct(naive["final_accuracy"]),
                 _pct(replay["final_accuracy"])],
                ["Final mean forgetting", _pct(naive["final_forgetting"]),
                 _pct(replay["final_forgetting"])],
                ["Old-class retention (final)",
                 _pct(naive["final_old_accuracy"]),
                 _pct(replay["final_old_accuracy"])],
                ["Average incremental accuracy",
                 _pct(naive["average_incremental_accuracy"]),
                 _pct(replay["average_incremental_accuracy"])],
                ["Training time (s)",
                 _pct(naive["training_time_seconds"]),
                 _pct(replay["training_time_seconds"])],
            ],
        ),
        "",
        "## Tradeoffs",
        "",
        *[f"- {line}" for line in selection["tradeoffs"]],
        "",
        "## Source experiment",
        "",
        f"- {selection['source_experiment']['dataset']} / "
        f"{selection['source_experiment']['scenario']} / "
        f"{selection['source_experiment']['variant']} / run "
        f"{selection['source_experiment']['run']}, "
        f"{selection['source_experiment']['num_experiences']} experiences, "
        f"seed {selection['source_experiment']['seed']}",
        f"- artifacts: {selection['source_experiment']['artifacts']}",
        "",
    ]
    return "\n".join(lines)


def render_summary_md(
    *,
    results: Mapping[str, Any],
    selection: Mapping[str, Any],
    forgetting: Mapping[str, Any],
    representative: Mapping[str, Any],
    environment: Mapping[str, Any],
    sha256: str,
    final_checkpoint: str,
) -> str:
    comparison = results["comparison"]
    top_f = forgetting["top_forgetting_classes"]["naive"][:5]
    low = forgetting["lowest_final_accuracy_classes"]["naive"][:5]
    exps = forgetting["experiences_with_largest_forgetting_increase"][:5]
    examples = representative["examples"]
    sessions = sorted(environment["errors_by_session"], key=lambda s: int(s))
    return f"""PHASE 6 — ERROR ANALYSIS + FINAL MODEL SELECTION

Dataset:
CORe50

Scenario:
{results['summary']['scenario']}

Variant:
{results['summary']['variant']}

Run:
{results['summary']['run']}

Experiences:
{results['num_experiences']}

------------------------------------------------------------
NAIVE
------------------------------------------------------------

Final accuracy:
{comparison['final_accuracy']['naive']}

Final forgetting:
{comparison['final_forgetting']['naive']}

Average incremental accuracy:
{comparison['average_incremental_accuracy']['naive']}

------------------------------------------------------------
REPLAY
------------------------------------------------------------

Final accuracy:
{comparison['final_accuracy']['replay']}

Final forgetting:
{comparison['final_forgetting']['replay']}

Average incremental accuracy:
{comparison['average_incremental_accuracy']['replay']}

------------------------------------------------------------
OBSERVED FORGETTING
------------------------------------------------------------

- Highest forgetting classes (naive): {', '.join(f"{r['name']} ({_pct(r['forgetting'])})" for r in top_f)}
- Lowest-accuracy classes (naive): {', '.join(f"{r['name']} ({_pct(r['final_accuracy'])})" for r in low)}
- Experiences with largest forgetting increase (naive): {', '.join(str(r['experience_id']) for r in exps)}
- Naive vs Replay differences: replay final accuracy {_pct(comparison['final_accuracy']['replay'])} vs naive {_pct(comparison['final_accuracy']['naive'])}; replay final forgetting {_pct(comparison['final_forgetting']['replay'])} vs naive {_pct(comparison['final_forgetting']['naive'])}; replay average incremental accuracy {_pct(comparison['average_incremental_accuracy']['replay'])} vs naive {_pct(comparison['average_incremental_accuracy']['naive'])}

------------------------------------------------------------
REPRESENTATIVE ERRORS
------------------------------------------------------------

- {len(examples)} examples (candidates evaluated: {representative['candidates_evaluated']} per method; errors: naive {representative['candidates_per_method'].get('naive', 0)}, replay {representative['candidates_per_method'].get('replay', 0)})
- Sessions covered: {', '.join('s' + s for s in sessions) or 'none'}
""" + "\n".join(
        f"- {ex['relative_path']} (s{ex['session']}): true={ex['true_name']}; "
        f"naive={ex['predictions']['naive']['predicted_name']} "
        f"({ex['predictions']['naive']['confidence']:.3f}, "
        f"{'ok' if ex['predictions']['naive']['correct'] else 'wrong'}); "
        f"replay={ex['predictions']['replay']['predicted_name']} "
        f"({ex['predictions']['replay']['confidence']:.3f}, "
        f"{'ok' if ex['predictions']['replay']['correct'] else 'wrong'})"
        for ex in examples
    ) + f"""

- Evidence-based observations: all examples are from official test
  sessions {SESSION_FACTS['test_sessions']} that are never used for
  training; session-level environmental factors vary per the dataset
  documentation, but no specific causal factor is claimed from these
  samples alone.

------------------------------------------------------------
FINAL MODEL
------------------------------------------------------------

Selected method:
{selection['selected_method']}

Checkpoint:
{final_checkpoint}

Reason:
Evidence-based comparison of Phase-5 measured accuracy, forgetting,
old-knowledge retention, and incremental performance. Selection rule:
{selection['rationale']} Measured detail: {selection['rationale_detail']}.

SHA-256:
{sha256}
"""
