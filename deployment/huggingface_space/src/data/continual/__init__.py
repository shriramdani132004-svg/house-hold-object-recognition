"""Session/experience-based continual data layer for official CORe50 scenarios.

Public API::

    from src.data.continual import load_scenario

    scenario = load_scenario("NIC")            # official NIC, variant "inc", run 0
    for experience in scenario.experiences:
        train_data = experience.train_samples
        evaluation_data = experience.evaluation_samples

Official ordering is preserved, experiences are immutable, images are
referenced (never copied), and every persisted path is project-relative.
"""

from __future__ import annotations

from src.data.continual.loader import load_scenario, load_scenario_cached
from src.data.continual.manifests import (
    build_manifest,
    build_scenario_manifest,
    write_manifest,
)
from src.data.continual.models import (
    ContinualDataError,
    ContinualExperience,
    ContinualScenario,
    ExperienceNotFoundError,
    MalformedFilelistError,
    RunNotFoundError,
    SampleRecord,
    SampleResolutionError,
    ScenarioNotFoundError,
    VariantNotFoundError,
)
from src.data.continual.scenarios import (
    ScenarioVariant,
    discover_variants,
    list_scenarios,
)
from src.data.continual.validation import (
    CheckResult,
    ValidationReport,
    validate_scenario,
)

__all__ = [
    "CheckResult",
    "ContinualDataError",
    "ContinualExperience",
    "ContinualScenario",
    "ExperienceNotFoundError",
    "MalformedFilelistError",
    "RunNotFoundError",
    "SampleRecord",
    "SampleResolutionError",
    "ScenarioNotFoundError",
    "ScenarioVariant",
    "ValidationReport",
    "VariantNotFoundError",
    "build_manifest",
    "build_scenario_manifest",
    "discover_variants",
    "list_scenarios",
    "load_scenario",
    "load_scenario_cached",
    "validate_scenario",
    "write_manifest",
]
