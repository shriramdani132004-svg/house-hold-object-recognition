"""THE single authoritative CORe50 50-class identity mapping.

Every component that needs label <-> name resolution (training, replay,
evaluation, inference, the web app, reports) must obtain it from here —
never by re-parsing ``object_mapping.json`` independently.

Contract guaranteed by :func:`load_class_mapping` and re-validated on
every load:

- exactly **50** identities with ``object_id`` 1..50 appearing once each;
- labels are **0-based and contiguous** (``label == object_id - 1``);
- names are unique, non-empty strings (official ``core50_labels.txt``
  order, e.g. ``plug_adapter1`` … ``remote_control5``);
- categories come from the official 10-category ``category_order`` and
  each category owns exactly 5 identities;
- the JSON categories agree with the independent official constants in
  :mod:`src.data.core50` (``OBJECT_CATEGORY``), so a corrupted or edited
  mapping fails loudly instead of silently reordering labels.

The mapping serialises back to the same payload shape (``objects`` +
``category_order``) for round-trip fidelity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.data import core50

DEFAULT_MAPPING_PATH: Path = core50.OBJECT_MAPPING_PATH
OFFICIAL_LABELS_PATH: Path = (
    core50.METADATA_DIR / "core50-official" / "extras" / "core50_labels.txt"
)
NUM_CLASSES: int = 50
#: Version tag recorded in checkpoints/reports alongside the checksum.
MAPPING_VERSION: str = "core50-object-mapping-v1"


class ClassMappingError(ValueError):
    """The object mapping file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class ClassMapping:
    """Immutable 50-class mapping; index positions ARE the labels."""

    names: tuple[str, ...]
    object_ids: tuple[int, ...]
    categories: tuple[str, ...]
    category_order: tuple[str, ...]
    source_path: str

    # -- basic access ---------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(range(self.num_classes))

    @property
    def name_to_label(self) -> dict[str, int]:
        return {name: label for label, name in enumerate(self.names)}

    @property
    def category_ids(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.category_order)}

    def name_for(self, label: int) -> str:
        if not isinstance(label, int) or not 0 <= label < self.num_classes:
            raise ClassMappingError(
                f"label {label!r} outside valid range 0..{self.num_classes - 1}"
            )
        return self.names[label]

    def label_for(self, name: str) -> int:
        try:
            return self.name_to_label[name]
        except KeyError:
            raise ClassMappingError(f"unknown class name {name!r}") from None

    def category_for(self, label: int) -> str:
        return self.categories[label]

    def category_id_for(self, label: int) -> int:
        return self.category_ids[self.categories[label]]

    # -- interop --------------------------------------------------------
    def to_class_names(self) -> dict[str, str]:
        """``{"0": "plug_adapter1", ...}`` — the legacy reader shape."""
        return {str(label): name for label, name in enumerate(self.names)}

    def to_payload(self) -> dict[str, Any]:
        """Serialise back to the official object-mapping payload shape."""
        return {
            "objects": [
                {
                    "object_id": object_id,
                    "directory": f"o{object_id}",
                    "name": name,
                    "category": category,
                }
                for object_id, name, category in zip(
                    self.object_ids, self.names, self.categories
                )
            ],
            "category_order": list(self.category_order),
        }

    def __len__(self) -> int:
        return self.num_classes


def _fail(source: str, message: str) -> ClassMappingError:
    return ClassMappingError(f"{message} (source: {source})")


def _validate(payload: Any, source: str) -> ClassMapping:
    if not isinstance(payload, dict):
        raise _fail(source, f"object mapping must be a JSON object, got {type(payload).__name__}")
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise _fail(source, "'objects' must be a non-empty list")
    raw_order = payload.get("category_order")
    if not isinstance(raw_order, list) or not raw_order:
        raise _fail(source, "'category_order' must be a non-empty list")
    category_order = tuple(str(c) for c in raw_order)
    if len(set(category_order)) != len(category_order):
        raise _fail(source, f"duplicate categories in category_order: {category_order}")

    entries: dict[int, dict[str, Any]] = {}
    for position, entry in enumerate(raw_objects):
        if not isinstance(entry, dict) or "object_id" not in entry:
            raise _fail(source, f"objects[{position}] must be an object with 'object_id'")
        try:
            object_id = int(entry["object_id"])
        except (TypeError, ValueError):
            raise _fail(source, f"objects[{position}].object_id is not an integer") from None
        if object_id in entries:
            raise _fail(source, f"duplicate object_id {object_id}")
        entries[object_id] = entry

    expected_ids = set(core50.OBJECT_IDS)
    got_ids = set(entries)
    if got_ids != expected_ids:
        missing = sorted(expected_ids - got_ids)
        extra = sorted(got_ids - expected_ids)
        raise _fail(
            source,
            f"object_ids must be exactly 1..50 (missing={missing}, unexpected={extra})",
        )

    names: list[str] = []
    categories: list[str] = []
    object_ids: list[int] = []
    seen_names: set[str] = set()
    for object_id in sorted(entries):
        entry = entries[object_id]
        name = entry.get("name")
        category = entry.get("category")
        if not isinstance(name, str) or not name:
            raise _fail(source, f"object {object_id} has an empty/non-string name")
        if name in seen_names:
            raise _fail(source, f"duplicate class name {name!r}")
        seen_names.add(name)
        if not isinstance(category, str) or category not in category_order:
            raise _fail(
                source,
                f"object {object_id} category {category!r} not in category_order",
            )
        expected_category = core50.OBJECT_CATEGORY[object_id]
        if category != expected_category:
            raise _fail(
                source,
                f"object {object_id} category {category!r} disagrees with the "
                f"official constant {expected_category!r}",
            )
        names.append(name)
        categories.append(category)
        object_ids.append(object_id)

    if len(names) != NUM_CLASSES:
        raise _fail(source, f"expected {NUM_CLASSES} classes, found {len(names)}")

    counts: dict[str, int] = {}
    for category in categories:
        counts[category] = counts.get(category, 0) + 1
    wrong = {c: n for c, n in counts.items() if n != 5}
    if wrong:
        raise _fail(source, f"each category must own exactly 5 identities, got {wrong}")
    if category_order != tuple(core50.OFFICIAL_CATEGORY_ORDER):
        raise _fail(
            source,
            "category_order disagrees with the official category order "
            "(exact list, not merely the same set)",
        )

    return ClassMapping(
        names=tuple(names),
        object_ids=tuple(object_ids),
        categories=tuple(categories),
        category_order=category_order,
        source_path=source,
    )


def mapping_from_payload(payload: Any, *, source: str = "<payload>") -> ClassMapping:
    """Validate an already-parsed object-mapping payload."""
    return _validate(payload, source)


def load_class_mapping(path: str | Path | None = None) -> ClassMapping:
    """Load and validate the authoritative mapping (defaults to official JSON).

    For the default official mapping the 50 names are additionally
    compared byte-for-byte against the official repository's
    ``extras/core50_labels.txt`` snapshot when that file is present, so a
    hand-edited name list cannot silently become the class contract.
    """
    mapping_path = Path(path) if path is not None else DEFAULT_MAPPING_PATH
    if not mapping_path.is_file():
        raise ClassMappingError(f"object mapping not found: {mapping_path}")
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClassMappingError(f"object mapping unreadable ({mapping_path}): {exc}") from exc
    mapping = _validate(payload, str(mapping_path))
    if mapping_path.resolve() == DEFAULT_MAPPING_PATH.resolve():
        assert_matches_official_names(mapping)
    return mapping


def assert_matches_official_names(mapping: ClassMapping) -> None:
    """Fail unless ``mapping.names`` equals official ``core50_labels.txt``.

    Silently returns when the official snapshot is not installed (the
    strict 50-class + category checks in :func:`load_class_mapping` still
    apply); raises :class:`ClassMappingError` on any difference.
    """
    if not OFFICIAL_LABELS_PATH.is_file():
        return
    official = [
        line.strip()
        for line in OFFICIAL_LABELS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if official != list(mapping.names):
        diffs = [
            (i, expected, got)
            for i, (expected, got) in enumerate(zip(official, mapping.names, strict=False))
            if expected != got
        ]
        raise ClassMappingError(
            f"mapping names disagree with official {OFFICIAL_LABELS_PATH} at "
            f"{len(diffs)} position(s), first: {diffs[:3]!r}"
            + (
                f" (length {len(official)} vs {len(mapping.names)})"
                if len(official) != len(mapping.names)
                else ""
            )
        )


def mapping_checksum(path: str | Path | None = None) -> str:
    """SHA-256 of the mapping file bytes — recorded in checkpoints/reports."""
    mapping_path = Path(path) if path is not None else DEFAULT_MAPPING_PATH
    if not mapping_path.is_file():
        raise ClassMappingError(f"object mapping not found: {mapping_path}")
    return hashlib.sha256(mapping_path.read_bytes()).hexdigest()


def load_class_names(path: str | Path | None = None) -> dict[str, str]:
    """``{"0": name, ...}`` label -> name view of the authoritative mapping."""
    return load_class_mapping(path).to_class_names()


def load_object_lookup(path: str | Path) -> dict[int, dict[str, Any]]:
    """Shared per-object lookup used by the scenario loader.

    Parses ``object_mapping.json`` in exactly one place. Structural checks
    only (unique ids, non-empty names, category declared in
    ``category_order``) — synthetic fixture mappings with fewer than 50
    objects are allowed here because scenario-level validation checks the
    references themselves. The strict 50-class contract lives in
    :func:`load_class_mapping`.
    """
    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise ClassMappingError(f"object mapping not found: {mapping_path}")
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClassMappingError(
            f"object mapping unreadable ({mapping_path}): {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ClassMappingError(
            f"object mapping must be a JSON object ({mapping_path})"
        )
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ClassMappingError(f"'objects' must be a non-empty list ({mapping_path})")
    raw_order = payload.get("category_order", [])
    category_order = tuple(str(c) for c in raw_order) if isinstance(raw_order, list) else ()
    lookup: dict[int, dict[str, Any]] = {}
    for position, entry in enumerate(raw_objects):
        if not isinstance(entry, dict) or "object_id" not in entry:
            raise ClassMappingError(
                f"objects[{position}] must be an object with 'object_id' ({mapping_path})"
            )
        try:
            object_id = int(entry["object_id"])
        except (TypeError, ValueError):
            raise ClassMappingError(
                f"objects[{position}].object_id is not an integer ({mapping_path})"
            ) from None
        if object_id in lookup:
            raise ClassMappingError(f"duplicate object_id {object_id} ({mapping_path})")
        name = entry.get("name")
        category = entry.get("category")
        if not isinstance(name, str) or not name:
            raise ClassMappingError(
                f"object {object_id} has an empty/non-string name ({mapping_path})"
            )
        if not isinstance(category, str):
            raise ClassMappingError(
                f"object {object_id} has a non-string category ({mapping_path})"
            )
        if category_order and category not in category_order:
            raise ClassMappingError(
                f"object {object_id} category {category!r} not in category_order "
                f"({mapping_path})"
            )
        lookup[object_id] = {
            "name": name,
            "category_name": category,
            "category_id": category_order.index(category) if category in category_order else -1,
        }
    return lookup
