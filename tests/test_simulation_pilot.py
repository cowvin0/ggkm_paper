"""Exercise pilot persistence and reporting without fitting simulation models."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import run_pcm_experiment as pcm
from scripts import run_simulation_pilot as pilot
from scripts import summarize_simulation_pilot as summary
from scripts.table_utils.data import ALL_METHODS, PRIMARY_SCENARIOS, Method, Scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_design(monkeypatch):
    cells = pilot.build_experiments()[:2]
    monkeypatch.setattr(pilot, "build_experiments", lambda: deepcopy(cells))
    return pilot.build_pilot_design()


def _frame(cell, *, failures=1, scale=1.0):
    """Include all attempts, matched patient seeds, and a deliberately different RMSE."""
    successful = 5 - failures
    status = ["ok"] * successful + ["failed"] * failures
    frame = pd.DataFrame({
        "replication": range(5),
        "status": status,
        "n": cell["n"], "method": cell["method"],
        "generating_distribution": cell["generating_distribution"],
        "generating_phi": cell["generating_phi"],
        "model": cell["model_name"], "kernel": cell["kernel"],
        "generation_seed": np.arange(5) + 10000,
        "split_seed": np.arange(5) + 20000,
        "inner_seed": np.arange(5) + 30000,
        "test_ibs": np.asarray([0.20, 0.21, 0.19, 0.205, 0.195]) * scale,
        "cure_mse": np.asarray([0.020, 0.030, 0.015, 0.025, 0.018]) * scale,
        "cure_rmse": [0.01, 0.2, 0.3, 0.4, 0.5],
        "test_auc": [0.70, 0.71, 0.69, 0.705, 0.695],
        "error": [""] * successful + ["synthetic fit failure"] * failures,
    })
    frame.loc[frame["status"].eq("failed"), ["test_ibs", "cure_mse", "test_auc"]] = np.nan
    return frame


def _save_cells(monkeypatch, design, results_dir, *, failures=1, scale=1.0, limit=None):
    cells = design["cells"] if limit is None else design["cells"][:limit]
    lookup = {(cell["model_name"], cell["kernel"]): cell for cell in design["cells"]}

    def fit(**kwargs):
        cell = lookup[(kwargs["estimator_name"], kwargs["kernel"])]
        return _frame(cell, failures=failures, scale=scale)

    monkeypatch.setattr(pcm, "monte_carlo_pcm", fit)
    for cell in cells:
        pilot.run_cell(cell, design=design, output_dir=results_dir)


def test_pilot_uses_every_primary_scenario_and_reported_configuration_exactly_once():
    cells = pilot.build_experiments()
    combinations = [
        (Scenario(cell["method"], cell["generating_name"], cell["n"]),
         Method(cell["model_name"], cell["kernel"] or "parametric"))
        for cell in cells
    ]
    expected = {(scenario, method) for scenario in PRIMARY_SCENARIOS for method in ALL_METHODS}

    assert len(cells) == 324
    assert len(PRIMARY_SCENARIOS) == len(ALL_METHODS) == 18
    assert set(combinations) == expected
    assert all(count == 1 for count in Counter(combinations).values())
    assert [cell["task_id"] for cell in cells] == list(range(324))
    source = pcm.build_experiments()
    for cell in cells:
        assert {key: cell[key] for key in source[cell["source_task_id"]]} == source[cell["source_task_id"]]


def test_pilot_design_records_independent_seed_and_current_fit_settings(small_design):
    design = small_design

    assert design["replications"] == 5
    assert design["master_seed"] == 20260818
    assert design["master_seed"] != pcm.MASTER_SEED
    assert design["final_master_seed"] == pcm.MASTER_SEED
    for recorded, setting in [
        ("test_size", pcm.TEST_SIZE), ("n_inner_splits", pcm.N_INNER_SPLITS),
        ("inner_grid_points", pcm.INNER_GRID_POINTS), ("outer_grid_points", pcm.OUTER_GRID_POINTS),
        ("binomial_k", pcm.BINOMIAL_K),
    ]:
        assert design[recorded] == setting
    assert design["software_versions"]["skcure"]
    assert design["models"]
    for relative, digest in design["source_sha256"].items():
        assert digest == hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
    encoded = json.dumps(
        {key: value for key, value in design.items() if key != "study_id"},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    assert design["study_id"] == hashlib.sha256(encoded.encode()).hexdigest()


@pytest.mark.parametrize("seed", [-1, pcm.MASTER_SEED, True])
def test_pilot_rejects_invalid_or_final_experiment_seed(seed):
    with pytest.raises(ValueError, match="distinct from the final experiment seed"):
        pilot.build_pilot_design(seed=seed)


@pytest.mark.parametrize("replications", [0, 1, 1.5, True])
def test_pilot_requires_enough_planned_replications(replications):
    with pytest.raises(ValueError, match="at least two"):
        pilot.build_pilot_design(replications=replications)


@pytest.mark.parametrize("failures", [0, 1, 5])
def test_runner_forwards_final_fit_settings_and_retains_every_attempt(monkeypatch, small_design, tmp_path, failures):
    design = small_design
    cell = design["cells"][0]
    observed = []
    frame = _frame(cell, failures=failures)

    def fit(**kwargs):
        observed.append(kwargs)
        return frame.copy()

    monkeypatch.setattr(pcm, "monte_carlo_pcm", fit)
    csv_path = pilot.run_cell(cell, design=design, output_dir=tmp_path)

    assert len(observed) == 1
    assert observed[0] == {
        "n": cell["n"], "method": cell["method"],
        "generating_distribution": cell["generating_distribution"],
        "generating_phi": cell["generating_phi"], "generating_K_bin": pcm.BINOMIAL_K,
        "estimator": pcm.MODELS[cell["model_name"]], "estimator_name": cell["model_name"],
        "kernel": cell["kernel"], "parameter_grid": pcm.parameter_grid(cell["model_name"], cell["kernel"]),
        "replications": 5, "test_size": pcm.TEST_SIZE, "n_inner_splits": pcm.N_INNER_SPLITS,
        "inner_grid_points": pcm.INNER_GRID_POINTS, "outer_grid_points": pcm.OUTER_GRID_POINTS,
        "random_state": pilot.PILOT_MASTER_SEED, "return_diagnostics": False,
        "surface_grid_points": 41,
    }
    saved = pd.read_csv(csv_path)
    assert saved["replication"].tolist() == list(range(5))
    assert saved["status"].eq("failed").sum() == failures
    assert saved.loc[saved["status"].eq("failed"), "error"].eq("synthetic fit failure").all()
    _, metadata_path = pilot.cell_paths(tmp_path, cell)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["study_id"] == design["study_id"]
    assert metadata["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert metadata["planned_replications"] == 5
    assert metadata["master_seed"] != pcm.MASTER_SEED
    assert json.loads((tmp_path / "pilot_design.json").read_text()) == design


def test_completed_matching_cell_is_reused_without_refitting(monkeypatch, small_design, tmp_path):
    _save_cells(monkeypatch, small_design, tmp_path, limit=1)
    cell = small_design["cells"][0]
    csv_path, _ = pilot.cell_paths(tmp_path, cell)
    original = csv_path.read_bytes()
    monkeypatch.setattr(pcm, "monte_carlo_pcm", lambda **kwargs: pytest.fail("Completed pilot cell was refitted"))

    assert pilot.run_cell(cell, design=small_design, output_dir=tmp_path) == csv_path
    assert csv_path.read_bytes() == original


def test_another_design_cannot_reuse_existing_output_directory(monkeypatch, small_design, tmp_path):
    _save_cells(monkeypatch, small_design, tmp_path, limit=1)
    modified = {**small_design, "study_id": "another-independent-design"}

    with pytest.raises(ValueError, match="different pilot design"):
        pilot.run_cell(modified["cells"][0], design=modified, output_dir=tmp_path)


def test_resume_rejects_changed_csv_checksum(monkeypatch, small_design, tmp_path):
    _save_cells(monkeypatch, small_design, tmp_path, limit=1)
    cell = small_design["cells"][0]
    csv_path, _ = pilot.cell_paths(tmp_path, cell)
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="CSV checksum mismatch"):
        pilot.run_cell(cell, design=small_design, output_dir=tmp_path)


def test_resume_rejects_cell_from_another_study(monkeypatch, small_design, tmp_path):
    _save_cells(monkeypatch, small_design, tmp_path, limit=1)
    cell = small_design["cells"][0]
    _, metadata_path = pilot.cell_paths(tmp_path, cell)
    metadata = json.loads(metadata_path.read_text())
    metadata["study_id"] = "unrelated"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="mismatched study metadata"):
        pilot.load_cell_result(tmp_path, cell, small_design["study_id"])


@pytest.mark.parametrize("suffix", ["", "pilot", "nested/pilot"])
def test_pilot_output_cannot_overlap_final_result_directory(suffix):
    with pytest.raises(ValueError, match="separate from the final"):
        pilot.validate_output_dir(PROJECT_ROOT / "data" / "results" / suffix)


def test_pilot_output_protection_resolves_symbolic_links(tmp_path):
    alias = tmp_path / "final-alias"
    alias.symlink_to(PROJECT_ROOT / "data" / "results", target_is_directory=True)

    with pytest.raises(ValueError, match="separate from the final"):
        pilot.validate_output_dir(alias / "pilot")


def test_summary_reports_raw_cure_mse_and_writes_all_three_reports(monkeypatch, small_design, tmp_path):
    results_dir, report_dir = tmp_path / "cells", tmp_path / "reports"
    _save_cells(monkeypatch, small_design, results_dir)
    final_before = pcm.N_REPLICATIONS, pcm.MASTER_SEED

    report = summary.generate_report(results_dir=results_dir, output_dir=report_dir)

    assert report["ready_for_final"]
    assert report["n_complete_cells"] == report["n_expected_cells"] == 2
    assert report["final_replications"] == 250
    assert report["metrics"]["cure_mse"]["max_sd"] == pytest.approx(np.std([0.020, 0.030, 0.015, 0.025], ddof=1))
    assert report["metrics"]["test_ibs"]["max_sd"] == pytest.approx(np.std([0.20, 0.21, 0.19, 0.205], ddof=1))
    assert report["metrics"]["test_auc"]["projected_mcse"] == pytest.approx(np.std([0.70, 0.71, 0.69, 0.705], ddof=1) / np.sqrt(250))
    assert report["metrics"]["test_auc"]["required_replications"] is None
    assert report["allocation_metrics"] == ["test_ibs", "cure_mse"]
    assert sorted(path.name for path in report_dir.iterdir()) == [
        "pilot_cell_summary.csv", "pilot_precision_report.json", "pilot_precision_report.txt",
    ]
    saved = json.loads((report_dir / "pilot_precision_report.json").read_text())
    assert saved == report
    cells = pd.read_csv(report_dir / "pilot_cell_summary.csv")
    assert cells["attempted_replications"].tolist() == [5, 5]
    assert cells["successful_replications"].tolist() == [4, 4]
    text = (report_dir / "pilot_precision_report.txt").read_text()
    assert "not pooled with final estimates" in text
    assert "historical manuscript values are not imposed" in text
    assert (pcm.N_REPLICATIONS, pcm.MASTER_SEED) == final_before == (250, 20260817)


def test_summary_cli_returns_failure_when_precision_target_is_not_met(monkeypatch, small_design, tmp_path):
    _save_cells(monkeypatch, small_design, tmp_path, scale=100)

    exit_status = summary.main(["--results-dir", str(tmp_path)])

    assert exit_status == 2
    report = json.loads((tmp_path / "pilot_precision_report.json").read_text())
    assert report["complete"]
    assert not report["ready_for_final"]
    assert report["decision"] == "insufficient_precision"
    assert report["required_replications"] > pcm.N_REPLICATIONS
    assert pcm.N_REPLICATIONS == 250


def test_summary_cli_returns_failure_when_any_planned_cell_is_missing(monkeypatch, small_design, tmp_path):
    _save_cells(monkeypatch, small_design, tmp_path, limit=1)

    assert summary.main(["--results-dir", str(tmp_path)]) == 2
    report = json.loads((tmp_path / "pilot_precision_report.json").read_text())
    assert report["decision"] == "incomplete"
    assert report["required_replications"] is None
    assert report["n_recorded_cells"] == 1


def test_summary_cli_returns_failure_when_fewer_than_four_attempts_succeed(monkeypatch, small_design, tmp_path):
    _save_cells(monkeypatch, small_design, tmp_path, failures=2)

    assert summary.main(["--results-dir", str(tmp_path)]) == 2
    report = json.loads((tmp_path / "pilot_precision_report.json").read_text())
    assert not report["complete"]
    assert any("at least 4 required" in problem for problem in report["problems"])


def test_summary_rejects_saved_design_changed_without_new_identity(monkeypatch, small_design, tmp_path):
    _save_cells(monkeypatch, small_design, tmp_path)
    manifest = tmp_path / "pilot_design.json"
    changed = json.loads(manifest.read_text())
    changed["inner_grid_points"] += 1
    manifest.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="differs from the current simulation settings"):
        summary.generate_report(results_dir=tmp_path)


@pytest.mark.parametrize("filename", ["run_simulation_pilot.py", "summarize_simulation_pilot.py"])
def test_pilot_cli_help_works_outside_project(filename, tmp_path):
    process = subprocess.run(
        [sys.executable, "-B", str(PROJECT_ROOT / "scripts" / filename), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )

    assert process.returncode == 0, process.stderr
    assert "usage:" in process.stdout
    assert "--output-dir" in process.stdout


def test_runner_can_select_pbs_cell_without_affecting_final_settings(monkeypatch, small_design, tmp_path):
    monkeypatch.setattr(pilot, "build_pilot_design", lambda **kwargs: deepcopy(small_design))
    monkeypatch.setenv("PBS_ARRAY_INDEX", "1")
    monkeypatch.setattr(pcm, "monte_carlo_pcm", lambda **kwargs: _frame(small_design["cells"][1]))

    assert pilot.main(["--output-dir", str(tmp_path)]) == 0
    assert pilot.cell_paths(tmp_path, small_design["cells"][1])[0].is_file()
    assert not pilot.cell_paths(tmp_path, small_design["cells"][0])[0].exists()
    assert (pcm.N_REPLICATIONS, pcm.MASTER_SEED) == (250, 20260817)


def test_runner_rejects_incomplete_attempts_before_persisting_results(monkeypatch, small_design, tmp_path):
    cell = small_design["cells"][0]
    monkeypatch.setattr(pcm, "monte_carlo_pcm", lambda **kwargs: _frame(cell).iloc[:4])

    with pytest.raises(ValueError, match="every planned pilot attempt"):
        pilot.run_cell(cell, design=small_design, output_dir=tmp_path)
    assert not pilot.cell_paths(tmp_path, cell)[0].exists()
