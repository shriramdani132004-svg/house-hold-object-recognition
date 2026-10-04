"""Label-reference forensics: exhaustive manifest audit + image label proof.

Two pure, deterministic checks over the official filelists (metadata
level, no model involved):

- :func:`audit_references` — for EVERY reference used by training and
  evaluation verifies: parse validity, session resolves, object resolves,
  label in 0..49, identity in the official 50, ``label == object_id - 1``
  (the official NI/NIC convention), reverse mapping round-trip,
  path/object agreement, category agreement, duplicate/missing identity
  detection, and train/eval session separation. Any single failure fails
  the whole audit — nothing is skipped.
- :func:`representative_samples` — deterministic per-identity sample
  selection across sessions, printing path/session/identity/label/
  reverse-identity for hand-checkable proof that
  official identity == reverse-mapped identity.

Both functions return structured results; the thin CLIs in ``scripts/``
print them and set the exit code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.data import core50
from src.data.class_mapping import (
    MAPPING_VERSION,
    OFFICIAL_LABELS_PATH,
    ClassMapping,
    load_class_mapping,
    mapping_checksum,
)
from src.data.continual.scenarios import (
    iter_filelist,
    list_batch_files,
    resolve_run,
    resolve_variant,
    test_filelist_path,
)

OFFICIAL_LABEL_RULE = "label == object_id - 1"


@dataclass(frozen=True)
class AuditFailure:
    where: str
    line_number: int
    problem: str


@dataclass
class AuditResult:
    train_references: int = 0
    eval_references: int = 0
    files_checked: int = 0
    identity_counts: dict[int, int] = field(default_factory=dict)
    sessions_seen: set[int] = field(default_factory=set)
    failures: list[AuditFailure] = field(default_factory=list)
    mapping_checksum: str = ""
    mapping_version: str = ""

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def total_references(self) -> int:
        return self.train_references + self.eval_references


def _identity_name_for_object(mapping: ClassMapping, object_id: int) -> str:
    return mapping.name_for(object_id - 1)


def audit_references(
    *,
    scenario: str = "NIC",
    variant: str | None = None,
    run: int = 0,
    filelist_root: Path | None = None,
    images_root: Path | None = None,
    check_files_exist: bool = False,
    mapping: ClassMapping | None = None,
) -> AuditResult:
    """Exhaustively audit every filelist reference. Failures never skipped."""
    from src.data.continual.loader import DEFAULT_IMAGES_ROOT
    from src.data.continual.scenarios import project_relative

    mapping = mapping if mapping is not None else load_class_mapping()
    root = Path(filelist_root) if filelist_root else core50.FILELIST_DIR
    images = Path(images_root) if images_root else DEFAULT_IMAGES_ROOT
    resolved = resolve_variant(scenario, variant, root)
    run_dir = resolve_run(resolved, run, root)
    batch_files = list_batch_files(run_dir)

    result = AuditResult(
        mapping_checksum=mapping_checksum(),
        mapping_version=MAPPING_VERSION,
    )
    official_rule = resolved.scenario_type in ("NI", "NIC")
    train_sessions = set(core50.TRAIN_SESSIONS)
    test_sessions = set(core50.TEST_SESSIONS)
    seen_paths: dict[str, str] = {}

    def record_failure(where: str, line_number: int, problem: str) -> None:
        result.failures.append(AuditFailure(where, line_number, problem))

    def audit_file(
        path: Path,
        *,
        split: str,
        counter_name: str,
    ) -> None:
        where = project_relative(path)
        identity_seen_in_file: dict[int, str] = {}
        for line_number, rel_path, label, session_id, object_id in iter_filelist(path):
            setattr(result, counter_name, getattr(result, counter_name) + 1)
            result.sessions_seen.add(session_id)

            if session_id not in core50.SESSION_IDS:
                record_failure(where, line_number, f"unknown session {session_id}")
                continue
            expected_split_sessions = test_sessions if split == "test" else train_sessions
            if session_id not in expected_split_sessions:
                record_failure(
                    where,
                    line_number,
                    f"session {session_id} not in official {split} sessions "
                    f"{sorted(expected_split_sessions)}",
                )
            if label < 0 or label > 49:
                record_failure(where, line_number, f"label {label} outside 0..49")
                continue
            if object_id not in mapping.object_ids:
                record_failure(where, line_number, f"object id {object_id} not in mapping")
                continue
            if official_rule and label != object_id - 1:
                record_failure(
                    where,
                    line_number,
                    f"label {label} violates {OFFICIAL_LABEL_RULE} (object {object_id})",
                )
                continue

            path_object_token = rel_path.split("/")[1]  # s{S}/o{O}/...
            if path_object_token != f"o{object_id}":
                record_failure(
                    where, line_number, f"path token {path_object_token} != o{object_id}"
                )
                continue

            reverse = mapping.name_for(label)
            official_name = _identity_name_for_object(mapping, object_id)
            if reverse != official_name:
                record_failure(
                    where,
                    line_number,
                    f"reverse mapping gave {reverse!r} != official {official_name!r}",
                )
                continue
            if mapping.label_for(official_name) != label:
                record_failure(
                    where, line_number, f"round trip failed for {official_name!r}"
                )
                continue
            expected_category = core50.OBJECT_CATEGORY[object_id]
            if mapping.category_for(label) != expected_category:
                record_failure(
                    where,
                    line_number,
                    f"category mismatch: {mapping.category_for(label)!r} != "
                    f"{expected_category!r}",
                )
                continue

            previous = identity_seen_in_file.get(object_id)
            if previous is not None and previous != reverse:
                record_failure(
                    where, line_number, f"inconsistent identity for object {object_id}"
                )
            identity_seen_in_file[object_id] = reverse
            result.identity_counts[label] = result.identity_counts.get(label, 0) + 1

            other_split = seen_paths.get(rel_path)
            if other_split is not None and other_split != split:
                record_failure(
                    where, line_number, f"path {rel_path} appears in both splits"
                )
            seen_paths.setdefault(rel_path, split)

            if check_files_exist:
                result.files_checked += 1
                if not images.joinpath(*rel_path.split("/")).is_file():
                    record_failure(where, line_number, f"image missing: {rel_path}")

    for batch_path in batch_files:
        audit_file(batch_path, split="train", counter_name="train_references")
    audit_file(test_filelist_path(run_dir), split="test", counter_name="eval_references")

    present = set(result.identity_counts)
    expected = set(range(50))
    for label in sorted(expected - present):
        result.failures.append(
            AuditFailure("coverage", 0, f"identity {label} has no references")
        )
    if len(present - expected):
        result.failures.append(
            AuditFailure("coverage", 0, f"unexpected labels present: {sorted(present - expected)}")
        )
    if not (result.sessions_seen & test_sessions) or not (
        result.sessions_seen & train_sessions
    ):
        result.failures.append(
            AuditFailure("sessions", 0, "train and eval sessions not both observed")
        )
    return result


def representative_samples(
    *,
    per_identity: int = 3,
    scenario: str = "NIC",
    variant: str | None = None,
    run: int = 0,
    filelist_root: Path | None = None,
    check_files_exist: bool = True,
    mapping: ClassMapping | None = None,
) -> list[dict[str, object]]:
    """Deterministic per-identity samples spread across sessions.

    Each entry carries path/session/official identity/label/reverse
    identity plus ``matches`` (official identity == reverse identity).
    """
    from src.data.continual.scenarios import project_relative

    mapping = mapping if mapping is not None else load_class_mapping()
    root = Path(filelist_root) if filelist_root else core50.FILELIST_DIR
    resolved = resolve_variant(scenario, variant, root)
    run_dir = resolve_run(resolved, run, root)
    # Independent official source: the repository's extras/core50_labels.txt
    # (label for object N lives at line N-1), used to cross-check the
    # mapping-derived reverse identity.
    official_lines = [
        line.strip()
        for line in OFFICIAL_LABELS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    by_identity: dict[int, list[tuple[str, int]]] = {}
    sources = [*list_batch_files(run_dir), test_filelist_path(run_dir)]
    for path in sources:
        for _line, rel_path, label, session_id, _object_id in iter_filelist(path):
            by_identity.setdefault(label, []).append((rel_path, session_id))

    samples: list[dict[str, object]] = []
    for identity in sorted(by_identity):
        entries = sorted(by_identity[identity], key=lambda e: (e[1], e[0]))
        # deterministic spread: first, middle, last (deduplicated)
        picks = {entries[0], entries[len(entries) // 2], entries[-1]}
        for rel_path, session_id in sorted(picks)[: max(1, per_identity)]:
            path_object_id = int(rel_path.split("/")[1][1:])  # s{S}/o{O}/...
            official = official_lines[path_object_id - 1]
            reverse = mapping.name_for(identity)
            samples.append(
                {
                    "relative_path": rel_path,
                    "source": project_relative(run_dir),
                    "session": session_id,
                    "path_object_id": path_object_id,
                    "official_identity": official,
                    "label": identity,
                    "reverse_identity": reverse,
                    "matches": official == reverse,
                    "file_exists": (
                        None
                        if not check_files_exist
                        else (
                            core50.DATASET_DIR
                            / "core50_128x128"
                            / rel_path
                        ).is_file()
                    ),
                }
            )
    return samples
