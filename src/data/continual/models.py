"""Typed data model for the continual (experience-based) CORe50 pipeline.

The model is intentionally passive: it stores official filelist references and
derived metadata only. Image bytes are never loaded here, images are never
copied, and no persisted field contains machine-specific absolute paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


class ContinualDataError(Exception):
    """Base error for the continual data pipeline."""


class ScenarioNotFoundError(ContinualDataError):
    """The requested scenario/variant is not present in the official filelists."""


class VariantNotFoundError(ContinualDataError):
    """The requested variant of a scenario is not present."""


class RunNotFoundError(ContinualDataError):
    """The requested official run does not exist for this scenario variant."""


class MalformedFilelistError(ContinualDataError):
    """A filelist line or batch layout violates the official format."""


class SampleResolutionError(ContinualDataError):
    """A filelist reference could not be resolved to a known object/session."""


class ExperienceNotFoundError(ContinualDataError, IndexError):
    """Requested experience index does not exist for this scenario."""


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One official filelist reference.

    ``relative_path`` is a POSIX-style path relative to the official images
    root (e.g. ``s1/o1/C_01_01_000.png``). ``source_filelist`` and
    ``line_number`` describe where the sample is *introduced* (its first
    appearance in official batch order), which is stable for cumulative
    variants where later experiences repeat earlier samples.

    ``experience_id`` is the introducing experience for training samples and
    ``0`` for evaluation samples (the official test set is fixed and shared
    by every experience).
    """

    relative_path: str
    label: int
    split: str
    experience_id: int
    source_filelist: str
    line_number: int
    object_id: int
    session_id: int
    category_id: int
    category_name: str
    object_name: str

    @property
    def image_name(self) -> str:
        return self.relative_path.rsplit("/", 1)[-1]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view; contains only project-relative references."""
        return {
            "relative_path": self.relative_path,
            "label": self.label,
            "split": self.split,
            "experience_id": self.experience_id,
            "source_filelist": self.source_filelist,
            "line_number": self.line_number,
            "object_id": self.object_id,
            "session_id": self.session_id,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "object_name": self.object_name,
        }


@dataclass(frozen=True)
class ContinualExperience:
    """One sequential experience (official training batch) of a scenario run."""

    experience_id: int
    scenario_name: str
    run_id: int
    train_samples: tuple[SampleRecord, ...]
    evaluation_samples: tuple[SampleRecord, ...]
    train_source: str
    evaluation_source: str
    classes_introduced: tuple[int, ...]
    classes_seen: tuple[int, ...]
    objects_introduced: tuple[int, ...]
    objects_seen: tuple[int, ...]
    categories_introduced: tuple[int, ...]
    categories_seen: tuple[int, ...]
    sessions_present: tuple[int, ...]
    sessions_seen: tuple[int, ...]

    @property
    def train_count(self) -> int:
        return len(self.train_samples)

    @property
    def evaluation_count(self) -> int:
        return len(self.evaluation_samples)

    @property
    def source_lines(self) -> int:
        """Number of official filelist lines backing this experience's train data."""
        return len(self.train_samples)


@dataclass(frozen=True)
class ContinualScenario:
    """A loaded official scenario variant for one run.

    ``experiences`` is a tuple in official batch order (experience 0 first).
    The image root lives only in memory; ``metadata`` is JSON-serializable
    and uses project-relative paths exclusively.
    """

    name: str
    scenario_type: str
    variant: str
    run_id: int
    experiences: tuple[ContinualExperience, ...]
    metadata: dict[str, Any]
    images_root: Any  # pathlib.Path, typed loosely to keep the model import-light

    def __len__(self) -> int:
        return len(self.experiences)

    def get_experience(self, index: int) -> ContinualExperience:
        """Return an experience by index (supports negative indexing)."""
        try:
            return self.experiences[index]
        except IndexError as exc:
            raise ExperienceNotFoundError(
                f"Experience index {index} out of range for scenario "
                f"'{self.name}' with {len(self.experiences)} experiences"
            ) from exc

    def iter_experiences(self) -> Iterator[ContinualExperience]:
        yield from self.experiences

    def iter_train_samples(
        self, experience: ContinualExperience | int | None = None
    ) -> Iterator[SampleRecord]:
        """Iterate training samples.

        With no argument, walks every experience in official order (cumulative
        variants therefore re-yield earlier samples — their official content).
        """
        if experience is None:
            for exp in self.experiences:
                yield from exp.train_samples
            return
        if isinstance(experience, int):
            experience = self.get_experience(experience)
        yield from experience.train_samples

    def iter_evaluation_samples(self) -> Iterator[SampleRecord]:
        """Iterate the official (fixed) evaluation set once."""
        if self.experiences:
            yield from self.experiences[0].evaluation_samples

    def get_seen_classes(self, index: int) -> tuple[int, ...]:
        return self.get_experience(index).classes_seen

    def get_seen_objects(self, index: int) -> tuple[int, ...]:
        return self.get_experience(index).objects_seen

    def get_seen_sessions(self, index: int) -> tuple[int, ...]:
        return self.get_experience(index).sessions_seen

    def image_path(self, record: SampleRecord) -> Any:
        """Resolve a record to a local image path (in memory only)."""
        from pathlib import Path

        return Path(self.images_root).joinpath(*record.relative_path.split("/"))

    def scenario_summary(self) -> dict[str, Any]:
        """Aggregate, JSON-serializable summary (no absolute paths)."""
        last = self.experiences[-1] if self.experiences else None
        first = self.experiences[0] if self.experiences else None
        return {
            "name": self.name,
            "scenario_type": self.scenario_type,
            "variant": self.variant,
            "run_id": self.run_id,
            "experience_count": len(self.experiences),
            "train_samples_total": sum(e.train_count for e in self.experiences),
            "evaluation_samples": first.evaluation_count if first else 0,
            "classes_seen": list(last.classes_seen) if last else [],
            "objects_seen": list(last.objects_seen) if last else [],
            "sessions_seen": list(last.sessions_seen) if last else [],
            "first_experience_source": first.train_source if first else None,
            "last_experience_source": last.train_source if last else None,
            "metadata": dict(self.metadata),
        }
