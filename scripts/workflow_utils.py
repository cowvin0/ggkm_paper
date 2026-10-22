"""Shared job definitions for local execution and PBS workers.

Commands are module names followed by arguments; the experiment modules retain
their existing settings, array ordering, checkpointing, and output locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("simulation", "melanoma", "breast-cancer")
STAGES = ("all", "pilot", "experiments", "figures", "tables", "postprocess")


@dataclass(frozen=True)
class Job:
    key: str
    dataset: str
    stage: str
    pbs_file: str
    commands: tuple[tuple[str, ...], ...]

    @property
    def is_array(self) -> bool:
        return self.stage in {"pilot", "experiments"}

    @property
    def prerequisite(self) -> str | None:
        if self.key == "simulation-pilot-summary":
            return "simulation-pilot"
        if self.key == "simulation-experiments":
            return "simulation-pilot-summary"
        if self.stage == "pilot":
            return None
        if self.stage == "experiments":
            return None
        return f"{self.dataset}-experiments"


_COMMANDS = {
    "simulation": {
        "pilot": (("scripts.run_simulation_pilot",),),
        "pilot-summary": (("scripts.summarize_simulation_pilot",),),
        "experiments": (("scripts.run_pcm_experiment",),),
        "figures": (
            ("scripts.create_paired_plot_1_2_3_methods",),
            ("scripts.create_paired_plot_interaction_quadratic_surfaces",),
            ("scripts.recover_cure_surfaces_figure",),
        ),
        "tables": (("scripts.create_simulation_tables",),),
    },
    "melanoma": {
        "experiments": (("scripts.run_melanoma_experiment",),),
        "figures": (
            ("scripts.melanoma_auc_plot",),
            ("scripts.melanoma_individual_cure_variation",),
        ),
        "tables": (
            ("scripts.create_clinical_performance_tables", "--dataset", "melanoma"),
        ),
    },
    "breast-cancer": {
        "experiments": (("scripts.run_breast_cancer_experiment",),),
        "figures": (
            ("scripts.breast_cancer_auc_plot",),
            ("scripts.breast_cancer_individual_cure_variation",),
        ),
        "tables": (
            ("scripts.create_clinical_performance_tables", "--dataset", "breast-cancer"),
        ),
    },
}

_PBS_FILES = {
    "simulation-pilot": "run_simulation_pilot.pbs",
    "simulation-pilot-summary": "summarize_simulation_pilot.pbs",
    "simulation-experiments": "run_experiment.pbs",
    "melanoma-experiments": "run_melanoma_experiment.pbs",
    "breast-cancer-experiments": "run_breast_cancer_experiment.pbs",
    "simulation-figures": "create_simulation_figures.pbs",
    "melanoma-figures": "create_melanoma_figures.pbs",
    "breast-cancer-figures": "create_breast_cancer_figures.pbs",
    "simulation-tables": "create_simulation_tables.pbs",
    "melanoma-tables": "create_melanoma_table.pbs",
    "breast-cancer-tables": "create_breast_cancer_table.pbs",
}

JOBS = {
    f"{dataset}-{stage}": Job(
        key=f"{dataset}-{stage}",
        dataset=dataset,
        stage=stage,
        pbs_file=_PBS_FILES[f"{dataset}-{stage}"],
        commands=_COMMANDS[dataset][stage],
    )
    for stage in ("pilot", "pilot-summary", "experiments", "figures", "tables")
    for dataset in DATASETS
    if stage in _COMMANDS[dataset]
}


def select_jobs(datasets: Iterable[str], stage: str = "all") -> list[Job]:
    """Select jobs in dependency order, with stable dataset and module ordering."""
    datasets = set(datasets)
    unknown = datasets.difference(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(sorted(unknown))}")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if stage == "pilot" and "simulation" not in datasets:
        raise ValueError("--stage pilot requires simulation in --datasets")
    stages = {
        "all": {"pilot", "pilot-summary", "experiments", "figures", "tables"},
        "pilot": {"pilot", "pilot-summary"},
        "experiments": {"pilot", "pilot-summary", "experiments"},
        "postprocess": {"figures", "tables"},
    }.get(stage, {stage})
    return [job for job in JOBS.values() if job.dataset in datasets and job.stage in stages]
