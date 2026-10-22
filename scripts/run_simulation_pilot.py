#!/usr/bin/env python3
"""Run independent five-replicate cells for the primary simulation precision pilot.

Use run_workflow.py --mode local/cluster --datasets simulation --stage pilot
for the complete pilot and its precision report. A PBS array worker runs one
cell; --task-id selects a cell manually and --all runs every cell sequentially.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts import run_pcm_experiment as pcm
from scripts.experiment_utils.artifacts import atomic_write_json
from scripts.table_utils.data import ALL_METHODS, PRIMARY_SCENARIOS, Method, Scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "pilot_results" / "primary"
PILOT_REPLICATIONS = 5
PILOT_MASTER_SEED = 20260818
SCHEMA_VERSION = 1


def build_experiments():
    """Retain all 18 primary scenarios and 18 reported model configurations."""
    scenarios = set(PRIMARY_SCENARIOS)
    methods = set(ALL_METHODS)
    cells = []
    for source_task_id, experiment in enumerate(pcm.build_experiments()):
        scenario = Scenario(experiment["method"], experiment["generating_name"], experiment["n"])
        method = Method(experiment["model_name"], experiment["kernel"] or "parametric")
        if scenario in scenarios and method in methods:
            cells.append({**experiment, "task_id": len(cells), "source_task_id": source_task_id})
    expected = {(scenario, method) for scenario in scenarios for method in methods}
    actual = {
        (Scenario(cell["method"], cell["generating_name"], cell["n"]),
         Method(cell["model_name"], cell["kernel"] or "parametric"))
        for cell in cells
    }
    if actual != expected or len(cells) != len(expected):
        raise ValueError("The PCM grid must contain every primary scenario and reported model configuration exactly once.")
    return cells


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _factory_settings(factory):
    keywords = {}
    if isinstance(factory, functools.partial):
        keywords = dict(factory.keywords or {})
        factory = factory.func
    source = inspect.getsourcefile(factory)
    return {
        "class": f"{factory.__module__}.{factory.__qualname__}",
        "fixed_parameters": keywords,
        "source_sha256": _digest(source) if source else None,
    }


def build_pilot_design(*, replications=PILOT_REPLICATIONS, seed=PILOT_MASTER_SEED):
    """Record the fitting design, software, and source checksums with each pilot."""
    if type(replications) is not int or replications < 2:
        raise ValueError("Pilot replications must be an integer of at least two.")
    if type(seed) is not int or seed < 0 or seed == pcm.MASTER_SEED:
        raise ValueError("The pilot seed must be a nonnegative integer distinct from the final experiment seed.")
    cells = build_experiments()
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "skcure"):
        versions[package] = importlib.metadata.version(package)
    distribution = importlib.metadata.distribution("skcure")
    provenance = distribution.read_text("direct_url.json")
    skcure_origin = json.loads(provenance) if provenance else None
    if skcure_origin and "vcs_info" in skcure_origin:
        skcure_origin = {
            "url": skcure_origin.get("url"),
            "commit_id": skcure_origin["vcs_info"].get("commit_id"),
        }
    source_files = (
        "evaluate/cross_validation.py", "evaluate/util_cross_validation.py",
        "evaluate/simulated_data.py", "evaluate/metrics.py",
        "models/gg_primal.py", "models/splines.py",
        "scripts/experiment_utils/models.py", "scripts/run_simulation_pilot.py",
    )
    design = {
        "schema_version": SCHEMA_VERSION,
        "study": "primary-simulation-pilot",
        "replications": replications,
        "master_seed": seed,
        "final_master_seed": pcm.MASTER_SEED,
        "test_size": pcm.TEST_SIZE,
        "n_inner_splits": pcm.N_INNER_SPLITS,
        "inner_grid_points": pcm.INNER_GRID_POINTS,
        "outer_grid_points": pcm.OUTER_GRID_POINTS,
        "binomial_k": pcm.BINOMIAL_K,
        "surface_grid_points": 41,
        "software_versions": versions,
        "skcure_origin": skcure_origin,
        "source_sha256": {name: _digest(PROJECT_ROOT / name) for name in source_files},
        "models": {
            name: _factory_settings(pcm.MODELS[name])
            for name in dict.fromkeys(cell["model_name"] for cell in cells)
        },
        "cells": [
            {**cell, "parameter_grid": pcm.parameter_grid(cell["model_name"], cell["kernel"])}
            for cell in cells
        ],
    }
    # Final allocation/precision targets are deliberately excluded: the same raw
    # pilot can be reassessed at another final R without fitting it again.
    encoded = json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**design, "study_id": hashlib.sha256(encoded.encode()).hexdigest()}


def cell_paths(output_dir, cell):
    kernel = cell["kernel"] or "parametric"
    stem = (
        f"pilot_task{cell['task_id']:03d}_{cell['model_name']}_n{cell['n']}"
        f"_method{cell['method']}_gen{cell['generating_name']}_kernel{kernel}"
    )
    folder = Path(output_dir)
    return folder / f"replications_{stem}.csv", folder / f"metadata_{stem}.json"


def validate_output_dir(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    final_results = (PROJECT_ROOT / "data" / "results").resolve()
    if output_dir == final_results or final_results in output_dir.parents:
        raise ValueError("Pilot outputs must be separate from the final data/results directory.")
    return output_dir


def write_design(output_dir, design):
    path = Path(output_dir) / "pilot_design.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("study_id") != design["study_id"]:
            raise ValueError("This directory contains a different pilot design or software version. Use a separate --output-dir.")
    else:
        atomic_write_json(path, design)


def _write_csv(path, frame):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cell_result(output_dir, cell, study_id):
    """Accept completed cells only when the study identity and CSV checksum match."""
    csv_path, metadata_path = cell_paths(output_dir, cell)
    if not csv_path.exists() and not metadata_path.exists():
        return None
    if not csv_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Incomplete pilot cell {cell['task_id']}: CSV and metadata must both exist.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("study_id") != study_id or metadata.get("task_id") != cell["task_id"]:
        raise ValueError(f"Pilot cell {cell['task_id']} has mismatched study metadata.")
    if metadata.get("csv_sha256") != _digest(csv_path):
        raise ValueError(f"Pilot CSV checksum mismatch: {csv_path}")
    return pd.read_csv(csv_path)


def run_cell(cell, *, design, output_dir, overwrite=False):
    """Use the final simulation engine and tuning settings with independent seeds."""
    output_dir = validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_design(output_dir, design)
    csv_path, metadata_path = cell_paths(output_dir, cell)
    if not overwrite and csv_path.exists() and metadata_path.exists():
        frame = load_cell_result(output_dir, cell, design["study_id"])
        ids = pd.to_numeric(frame["replication"], errors="raise").sort_values().tolist()
        if ids != list(range(design["replications"])):
            raise ValueError(f"Pilot cell {cell['task_id']} does not contain all planned replication IDs.")
        print(f"Reusing completed pilot cell {cell['task_id']}: {csv_path}", flush=True)
        return csv_path
    results = pcm.monte_carlo_pcm(
        n=cell["n"], method=cell["method"],
        generating_distribution=cell["generating_distribution"],
        generating_phi=cell["generating_phi"], generating_K_bin=design["binomial_k"],
        estimator=pcm.MODELS[cell["model_name"]], estimator_name=cell["model_name"],
        kernel=cell["kernel"], parameter_grid=pcm.parameter_grid(cell["model_name"], cell["kernel"]),
        replications=design["replications"], test_size=design["test_size"],
        n_inner_splits=design["n_inner_splits"], inner_grid_points=design["inner_grid_points"],
        outer_grid_points=design["outer_grid_points"], random_state=design["master_seed"],
        return_diagnostics=False, surface_grid_points=design["surface_grid_points"],
    )
    ids = pd.to_numeric(results["replication"], errors="raise").sort_values().tolist()
    if ids != list(range(design["replications"])):
        raise ValueError("The simulation engine did not return every planned pilot attempt.")
    _write_csv(csv_path, results)
    atomic_write_json(metadata_path, {
        "schema_version": SCHEMA_VERSION, "study_id": design["study_id"],
        "task_id": cell["task_id"], "csv_file": csv_path.name,
        "csv_sha256": _digest(csv_path), "planned_replications": design["replications"],
        "master_seed": design["master_seed"],
    })
    print(f"Saved pilot cell {cell['task_id']}: {csv_path}", flush=True)
    return csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-id", type=int, help="One pilot cell; defaults to PBS_ARRAY_INDEX.")
    selection.add_argument("--all", action="store_true", help="Run all pilot cells sequentially.")
    parser.add_argument("--replications", type=int, default=PILOT_REPLICATIONS)
    parser.add_argument("--seed", type=int, default=PILOT_MASTER_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Recompute completed cells of the same pilot design.")
    args = parser.parse_args(argv)
    try:
        design = build_pilot_design(replications=args.replications, seed=args.seed)
        task_id = args.task_id
        if not args.all and task_id is None:
            if "PBS_ARRAY_INDEX" not in os.environ:
                raise ValueError("Select --task-id, --all, or use run_workflow.py --stage pilot.")
            task_id = int(os.environ["PBS_ARRAY_INDEX"])
        cells = design["cells"]
        if not args.all:
            if not 0 <= task_id < len(cells):
                raise ValueError(f"Task index must be between 0 and {len(cells) - 1}.")
            cells = [cells[task_id]]
        for cell in cells:
            run_cell(cell, design=design, output_dir=args.output_dir, overwrite=args.overwrite)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
