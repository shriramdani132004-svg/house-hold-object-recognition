"""Discovery and streaming parsing of the official CORe50 scenario filelists.

Scenario definitions are never invented: everything here reads the shipped
official filelist tree under ``data/raw/core50/filelists`` (extracted from
the official ``batches_filelists.zip`` / ``batches_filelists_NICv2.zip``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.data import core50
from src.data.continual.models import (
    MalformedFilelistError,
    RunNotFoundError,
    SampleResolutionError,
    ScenarioNotFoundError,
    VariantNotFoundError,
)

SCENARIO_DIR_RE = re.compile(r"^(NI|NC|NIC)_(.+)$")
BATCH_FILE_RE = re.compile(r"^train_batch_(\d+)_filelist\.txt$")
REL_PATH_RE = re.compile(r"^s(\d+)/o(\d+)/([^/]+\.png)$")
SCENARIO_ORDER = {"NI": 0, "NC": 1, "NIC": 2}


@dataclass(frozen=True)
class ScenarioVariant:
    """One discovered official scenario variant (e.g. ``NIC_inc``)."""

    name: str
    scenario_type: str
    variant: str
    runs: tuple[int, ...]

    def describe(self) -> str:
        return f"{self.name} (runs: {', '.join(str(r) for r in self.runs)})"


def _sort_key(variant: ScenarioVariant) -> tuple[int, int, str]:
    # Incremental variants first (they are the official default flavour).
    return (
        SCENARIO_ORDER.get(variant.scenario_type, 99),
        0 if variant.variant.lower() == "inc" else 1,
        variant.variant,
    )


def discover_variants(filelist_root: Path | None = None) -> tuple[ScenarioVariant, ...]:
    """Scan the official filelist tree and return sorted scenario variants."""
    root = Path(filelist_root) if filelist_root else core50.FILELIST_DIR
    if not root.is_dir():
        raise ScenarioNotFoundError(
            f"Official filelist directory not found: {root}. "
            "Expected data/raw/core50/filelists from the Phase 2 acquisition."
        )
    variants: list[ScenarioVariant] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = SCENARIO_DIR_RE.match(entry.name)
        if match is None:
            continue
        runs = sorted(
            int(p.name[3:])
            for p in entry.iterdir()
            if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit()
        )
        if not runs:
            continue
        variants.append(
            ScenarioVariant(
                name=entry.name,
                scenario_type=match.group(1),
                variant=match.group(2),
                runs=tuple(runs),
            )
        )
    variants.sort(key=_sort_key)
    return tuple(variants)


def list_scenarios(filelist_root: Path | None = None) -> tuple[ScenarioVariant, ...]:
    """Public alias: list official scenario variants with their runs."""
    return discover_variants(filelist_root)


def resolve_variant(
    scenario: str,
    variant: str | None = None,
    filelist_root: Path | None = None,
) -> ScenarioVariant:
    """Resolve ``scenario`` (+ optional ``variant``) to an official variant.

    When no variant is requested the incremental (``inc``) flavour is used —
    the same default as the official ``data_loader.py`` (``cumul=False``).
    Any other ambiguity or miss is rejected with the available options.
    """
    variants = discover_variants(filelist_root)
    wanted = scenario.strip().upper()
    matches = [v for v in variants if v.scenario_type == wanted]
    if not matches:
        available = ", ".join(v.name for v in variants) or "<none>"
        raise ScenarioNotFoundError(
            f"Unknown scenario {scenario!r}. Available: {available}"
        )
    if variant is None:
        variant = "inc"
    wanted_variant = variant.strip().lower()
    for v in matches:
        if v.variant.lower() == wanted_variant:
            return v
    options = ", ".join(v.variant for v in matches)
    raise VariantNotFoundError(
        f"Unknown variant {variant!r} for scenario {wanted}. Available: {options}"
    )


def resolve_run(
    variant: ScenarioVariant,
    run: int,
    filelist_root: Path | None = None,
) -> Path:
    """Return the ``run{N}`` directory for a variant, validating the run id."""
    if run not in variant.runs:
        options = ", ".join(str(r) for r in variant.runs)
        raise RunNotFoundError(
            f"Run {run} does not exist for {variant.name}. Available runs: {options}"
        )
    root = Path(filelist_root) if filelist_root else core50.FILELIST_DIR
    return root / variant.name / f"run{run}"


def list_batch_files(run_dir: Path) -> list[Path]:
    """Official training batch filelists in strict numeric batch order.

    Rejects gaps or non-contiguous batch numbering instead of silently
    reordering or dropping official files.
    """
    found: dict[int, Path] = {}
    for entry in run_dir.iterdir():
        match = BATCH_FILE_RE.match(entry.name)
        if match:
            found[int(match.group(1))] = entry
    if not found:
        raise MalformedFilelistError(f"No train_batch_*_filelist.txt files in {run_dir}")
    indices = sorted(found)
    expected = list(range(len(indices)))
    if indices != expected:
        raise MalformedFilelistError(
            f"Non-contiguous batch numbering in {run_dir}: found {indices}"
        )
    return [found[i] for i in indices]


def test_filelist_path(run_dir: Path) -> Path:
    path = run_dir / "test_filelist.txt"
    if not path.is_file():
        raise MalformedFilelistError(f"Missing official test filelist: {path}")
    return path


def project_relative(path: Path) -> str:
    """POSIX project-relative path for persisted metadata (no absolute paths)."""
    resolved = Path(path).resolve()
    root = core50.PROJECT_ROOT.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_filelist_line(
    line: str,
    *,
    source: Path,
    line_number: int,
) -> tuple[str, int, int, int]:
    """Parse one official line into ``(relative_path, label, session_id, object_id)``.

    Official format: ``s{S}/o{O}/{frame}.png {label}`` (single space).
    """
    stripped = line.rstrip("\r\n")
    if not stripped:
        raise MalformedFilelistError(f"{source}:{line_number}: empty line")
    try:
        rel_path, label_text = stripped.rsplit(" ", 1)
    except ValueError as exc:
        raise MalformedFilelistError(
            f"{source}:{line_number}: missing '<path> <label>' separator: {stripped!r}"
        ) from exc
    match = REL_PATH_RE.match(rel_path)
    if match is None:
        raise MalformedFilelistError(
            f"{source}:{line_number}: path does not match s{{S}}/o{{O}}/*.png: {rel_path!r}"
        )
    try:
        label = int(label_text)
    except ValueError as exc:
        raise MalformedFilelistError(
            f"{source}:{line_number}: label is not an integer: {label_text!r}"
        ) from exc
    session_id = int(match.group(1))
    object_id = int(match.group(2))
    return rel_path, label, session_id, object_id


def iter_filelist(
    path: Path,
) -> Iterator[tuple[int, str, int, int, int]]:
    """Stream ``(line_number, rel_path, label, session_id, object_id)`` tuples."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            rel_path, label, session_id, object_id = parse_filelist_line(
                line, source=path, line_number=line_number
            )
            yield line_number, rel_path, label, session_id, object_id


def validate_reference(
    rel_path: str,
    session_id: int,
    object_id: int,
    label: int,
    known_objects: dict[int, dict],
    *,
    source: Path,
    line_number: int,
) -> None:
    """Reject unknown sessions, objects, or out-of-range official labels."""
    if session_id not in core50.SESSION_IDS:
        raise SampleResolutionError(
            f"{source}:{line_number}: unknown session id {session_id} in {rel_path!r}"
        )
    info = known_objects.get(object_id)
    if info is None:
        raise SampleResolutionError(
            f"{source}:{line_number}: unknown object id {object_id} in {rel_path!r}"
        )
    max_label = max(core50.OBJECT_IDS) - 1
    if label < 0 or label > max_label:
        raise SampleResolutionError(
            f"{source}:{line_number}: label {label} outside official range "
            f"0..{max_label} in {rel_path!r}"
        )
