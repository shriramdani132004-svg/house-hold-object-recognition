"""Load official CORe50 scenario filelists into the continual data model.

The loader streams official batch filelists in official order, keeps a
per-load flyweight of sample records (so cumulative variants reference the
same record objects instead of duplicating them), and exposes everything as
immutable tuples. No image is decoded and no file is copied.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from src.data import core50
from src.data.class_mapping import ClassMappingError, load_object_lookup
from src.data.continual.models import (
    ContinualExperience,
    ContinualScenario,
    MalformedFilelistError,
    SampleRecord,
)
from src.data.continual.scenarios import (
    ScenarioVariant,
    iter_filelist,
    list_batch_files,
    project_relative,
    resolve_run,
    resolve_variant,
    test_filelist_path,
    validate_reference,
)

DEFAULT_IMAGES_ROOT = core50.DATASET_DIR / "core50_128x128"
DEFAULT_OBJECT_MAPPING = core50.METADATA_DIR / "object_mapping.json"

ProgressHook = Callable[[str, int, int], None]


def _load_object_mapping(path: Path) -> dict[int, dict[str, Any]]:
    """Per-object lookup via the shared class-mapping parser.

    Kept as a thin wrapper so scenario loading and label/name resolution
    use ONE parsing implementation (:mod:`src.data.class_mapping`).
    Structural errors surface as the loader's documented
    :class:`MalformedFilelistError`.
    """
    try:
        return load_object_lookup(path)
    except ClassMappingError as exc:
        raise MalformedFilelistError(str(exc)) from exc


def _parse_evaluation_set(
    path: Path,
    mapping: dict[int, dict[str, Any]],
    rel_sources: str,
    *,
    require_official_label_rule: bool = False,
) -> tuple[SampleRecord, ...]:
    records: list[SampleRecord] = []
    seen: dict[str, SampleRecord] = {}
    for line_number, rel_path, label, session_id, object_id in iter_filelist(path):
        validate_reference(
            rel_path,
            session_id,
            object_id,
            label,
            mapping,
            source=path,
            line_number=line_number,
            require_official_label_rule=require_official_label_rule,
        )
        existing = seen.get(rel_path)
        if existing is not None:
            if existing.label != label:
                raise MalformedFilelistError(
                    f"{path}:{line_number}: inconsistent label {label} for "
                    f"{rel_path!r} (already {existing.label})"
                )
            raise MalformedFilelistError(
                f"{path}:{line_number}: duplicate evaluation reference "
                f"{rel_path!r}"
            )
        info = mapping[object_id]
        record = SampleRecord(
            relative_path=rel_path,
            label=label,
            split="test",
            experience_id=0,
            source_filelist=rel_sources,
            line_number=line_number,
            object_id=object_id,
            session_id=session_id,
            category_id=info["category_id"],
            category_name=info["category_name"],
            object_name=info["name"],
        )
        seen[rel_path] = record
        records.append(record)
    if not records:
        raise MalformedFilelistError(f"Official test filelist is empty: {path}")
    return tuple(records)


def _load_batch_train_samples(
    batch_path: Path,
    mapping: dict[int, dict[str, Any]],
    flyweight: dict[str, SampleRecord],
    rel_source: str,
    experience_id: int,
    *,
    require_official_label_rule: bool = False,
) -> tuple[SampleRecord, ...]:
    samples: list[SampleRecord] = []
    for line_number, rel_path, label, session_id, object_id in iter_filelist(batch_path):
        validate_reference(
            rel_path,
            session_id,
            object_id,
            label,
            mapping,
            source=batch_path,
            line_number=line_number,
            require_official_label_rule=require_official_label_rule,
        )
        record = flyweight.get(rel_path)
        if record is None:
            info = mapping[object_id]
            record = SampleRecord(
                relative_path=rel_path,
                label=label,
                split="train",
                experience_id=experience_id,
                source_filelist=rel_source,
                line_number=line_number,
                object_id=object_id,
                session_id=session_id,
                category_id=info["category_id"],
                category_name=info["category_name"],
                object_name=info["name"],
            )
            flyweight[rel_path] = record
        elif record.label != label:
            raise MalformedFilelistError(
                f"{batch_path}:{line_number}: inconsistent label {label} for "
                f"{rel_path!r} (already {record.label})"
            )
        samples.append(record)
    if not samples:
        raise MalformedFilelistError(f"Official training batch is empty: {batch_path}")
    return tuple(samples)


def load_scenario(
    scenario: str = "NIC",
    *,
    variant: str | None = None,
    run: int = 0,
    images_root: Path | None = None,
    filelist_root: Path | None = None,
    object_mapping: Path | None = None,
    on_progress: ProgressHook | None = None,
) -> ContinualScenario:
    """Load one official scenario variant for one run.

    Parameters mirror what later training phases need to select explicitly:
    ``scenario`` (NI/NC/NIC), ``variant`` (``inc``/``cum``/``v2_*`` …) and
    ``run`` (official run id). Defaults follow the official loader
    (incremental flavour, run 0) and are always recorded in ``metadata``.
    """
    images_root = Path(images_root) if images_root else DEFAULT_IMAGES_ROOT
    filelist_root_arg = Path(filelist_root) if filelist_root else core50.FILELIST_DIR
    mapping_path = Path(object_mapping) if object_mapping else DEFAULT_OBJECT_MAPPING

    resolved: ScenarioVariant = resolve_variant(scenario, variant, filelist_root_arg)
    run_dir = resolve_run(resolved, run, filelist_root_arg)
    batch_files = list_batch_files(run_dir)
    eval_source = project_relative(test_filelist_path(run_dir))
    mapping = _load_object_mapping(mapping_path)
    # NI/NIC filelists use the official identity convention
    # label == object_id - 1; enforce it so a broken reference fails the
    # load instead of silently training under the wrong identity.
    official_label_rule = resolved.scenario_type in ("NI", "NIC")

    if on_progress:
        on_progress("evaluation", 0, len(batch_files) + 1)
    evaluation_samples = _parse_evaluation_set(
        test_filelist_path(run_dir),
        mapping,
        eval_source,
        require_official_label_rule=official_label_rule,
    )

    flyweight: dict[str, SampleRecord] = {}
    experiences: list[ContinualExperience] = []
    seen_classes: set[int] = set()
    seen_objects: set[int] = set()
    seen_categories: set[int] = set()
    train_sessions: set[int] = set()

    for experience_id, batch_path in enumerate(batch_files):
        rel_source = project_relative(batch_path)
        train_samples = _load_batch_train_samples(
            batch_path,
            mapping,
            flyweight,
            rel_source,
            experience_id,
            require_official_label_rule=official_label_rule,
        )
        classes_here = {record.label for record in train_samples}
        objects_here = {record.object_id for record in train_samples}
        categories_here = {record.category_id for record in train_samples}
        sessions_here = {record.session_id for record in train_samples}

        experiences.append(
            ContinualExperience(
                experience_id=experience_id,
                scenario_name=resolved.name,
                run_id=run,
                train_samples=train_samples,
                evaluation_samples=evaluation_samples,
                train_source=rel_source,
                evaluation_source=eval_source,
                classes_introduced=tuple(sorted(classes_here - seen_classes)),
                classes_seen=tuple(sorted(seen_classes | classes_here)),
                objects_introduced=tuple(sorted(objects_here - seen_objects)),
                objects_seen=tuple(sorted(seen_objects | objects_here)),
                categories_introduced=tuple(
                    sorted(categories_here - seen_categories)
                ),
                categories_seen=tuple(sorted(seen_categories | categories_here)),
                sessions_present=tuple(sorted(sessions_here)),
                sessions_seen=tuple(sorted(train_sessions | sessions_here)),
            )
        )
        seen_classes |= classes_here
        seen_objects |= objects_here
        seen_categories |= categories_here
        train_sessions |= sessions_here

        if on_progress:
            on_progress(
                "batch",
                experience_id + 1,
                len(batch_files),
            )

    all_labels = [record.label for record in flyweight.values()]
    all_labels.extend(record.label for record in evaluation_samples)
    metadata: dict[str, Any] = {
        "official_source": core50.OFFICIAL_PAGE,
        "official_repository": core50.OFFICIAL_REPO,
        "filelist_root": project_relative(filelist_root_arg),
        "images_root": project_relative(images_root),
        "object_mapping": project_relative(mapping_path),
        "batch_files": [project_relative(p) for p in batch_files],
        "evaluation_source": eval_source,
        "experience_count": len(experiences),
        "train_samples_total": sum(e.train_count for e in experiences),
        "evaluation_samples": len(evaluation_samples),
        "evaluation_unique_paths": len(
            {record.relative_path for record in evaluation_samples}
        ),
        "train_unique_paths": len(flyweight),
        "train_sessions": sorted(train_sessions),
        "evaluation_sessions": sorted(
            {record.session_id for record in evaluation_samples}
        ),
        "label_min": min(all_labels),
        "label_max": max(all_labels),
        "label_equals_object_minus_one": all(
            record.label == record.object_id - 1 for record in flyweight.values()
        ),
        "run": run,
        "variant": resolved.variant,
        "runs_available": list(resolved.runs),
    }

    return ContinualScenario(
        name=resolved.name,
        scenario_type=resolved.scenario_type,
        variant=resolved.variant,
        run_id=run,
        experiences=tuple(experiences),
        metadata=metadata,
        images_root=images_root,
    )


@lru_cache(maxsize=8)
def load_scenario_cached(
    scenario: str = "NIC",
    variant: str | None = None,
    run: int = 0,
    images_root: Path | None = None,
    filelist_root: Path | None = None,
    object_mapping: Path | None = None,
) -> ContinualScenario:
    """Cached repeat loader (deterministic content, avoids re-parsing)."""
    return load_scenario(
        scenario,
        variant=variant,
        run=run,
        images_root=images_root,
        filelist_root=filelist_root,
        object_mapping=object_mapping,
    )
