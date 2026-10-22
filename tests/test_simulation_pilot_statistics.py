"""Allocation uses independent within-cell pilot variability, never pooled means."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts.pilot_utils import summarize_pilot_cells


def _cell(task_id=0, **changes):
    cell = {
        "task_id": task_id, "source_task_id": 99 + task_id,
        "n": 300, "method": 1, "generating_name": "poisson",
        "generating_distribution": "poisson", "generating_phi": None,
        "model_name": "poisson", "kernel": "rbf",
    }
    cell.update(changes)
    return cell


def _frame(cell=None, *, ibs_sd=0.01, mse_sd=0.02, auc_sd=0.03, offset=0):
    cell = cell or _cell()
    standardized = np.arange(-2, 3, dtype=float) / np.std(np.arange(-2, 3), ddof=1)
    return pd.DataFrame({
        "replication": np.arange(5), "status": ["ok"] * 5,
        "generation_seed": np.arange(5) + 100,
        "split_seed": np.arange(5) + 200,
        "inner_seed": np.arange(5) + 300,
        "n": cell["n"], "method": cell["method"],
        "generating_distribution": cell["generating_distribution"],
        "generating_phi": cell["generating_phi"],
        "model": cell["model_name"], "kernel": cell["kernel"],
        "test_ibs": 0.2 + offset + standardized * ibs_sd,
        "cure_mse": 0.1 + offset + standardized * mse_sd,
        "test_auc": 0.7 + standardized * auc_sd,
        "cure_rmse": np.linspace(0, 100, 5),
    })


def test_quoted_standard_deviations_reproduce_allocation_and_projections():
    summary, report = summarize_pilot_cells(
        [_cell()], {0: _frame(ibs_sd=0.0418, mse_sd=0.0332, auc_sd=0.1211)},
    )
    assert report["decision"] == "ready"
    assert report["required_replications"] == 195
    assert report["metrics"]["test_ibs"]["required_replications"] == 195
    assert report["metrics"]["cure_mse"]["required_replications"] == 45
    assert report["metrics"]["test_auc"]["required_replications"] is None
    assert report["metrics"]["test_auc"]["meets_target"] is None
    assert report["metrics"]["test_ibs"]["projected_mcse"] == pytest.approx(0.0418 / math.sqrt(250))
    assert report["metrics"]["cure_mse"]["projected_mcse"] == pytest.approx(0.0332 / math.sqrt(250))
    assert report["metrics"]["test_auc"]["projected_mcse"] == pytest.approx(0.1211 / math.sqrt(250))
    assert summary.loc[0, "test_ibs_sd"] == pytest.approx(0.0418)
    assert summary.loc[0, "cure_mse_count"] == 5
    json.dumps(report, allow_nan=False)


def test_uses_maximum_within_cell_sample_sd_not_between_cell_variation():
    cells = [_cell(), _cell(1, kernel="linear")]
    frames = {0: _frame(cells[0], ibs_sd=0.01), 1: _frame(cells[1], ibs_sd=0.02, offset=1000)}
    summary, report = summarize_pilot_cells(cells, frames)
    assert summary["test_ibs_sd"].tolist() == pytest.approx([0.01, 0.02])
    assert report["metrics"]["test_ibs"]["max_sd"] == pytest.approx(0.02)
    assert report["metrics"]["test_ibs"]["worst_cells"] == [1]


def test_a_single_failed_attempt_is_counted_but_excluded_from_every_metric():
    frame = _frame()
    frame.loc[4, "status"] = "failed"
    frame.loc[4, ["test_ibs", "cure_mse", "test_auc"]] = 100000
    summary, report = summarize_pilot_cells([_cell()], {0: frame})
    assert report["complete"]
    assert summary.loc[0, "attempted_replications"] == 5
    assert summary.loc[0, "successful_replications"] == 4
    assert summary.loc[0, "failed_replications"] == 1
    assert summary.loc[0, "test_ibs_count"] == 4
    assert summary.loc[0, "test_ibs_sd"] == pytest.approx(frame.loc[:3, "test_ibs"].std(ddof=1))


@pytest.mark.parametrize("metric", ["test_ibs", "cure_mse", "test_auc"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_every_metric_requires_at_least_four_finite_successes(metric, bad_value):
    frame = _frame()
    frame.loc[[3, 4], metric] = bad_value
    summary, report = summarize_pilot_cells([_cell()], {0: frame})
    assert not report["complete"]
    assert not report["ready_for_final"]
    assert report["decision"] == "incomplete"
    assert report["required_replications"] is None
    assert summary.loc[0, f"{metric}_count"] == 3
    assert any(metric in problem for problem in report["problems"])
    json.dumps(report, allow_nan=False)


def test_missing_cells_remain_in_summary_and_prevent_a_final_decision():
    cells = [_cell(), _cell(1, kernel="linear")]
    summary, report = summarize_pilot_cells(cells, {0: _frame()})
    assert len(summary) == 2
    assert summary.loc[1, "attempted_replications"] == 0
    assert report["n_expected_cells"] == 2
    assert report["n_recorded_cells"] == 1
    assert report["n_complete_cells"] == 1
    assert report["decision"] == "incomplete"
    assert report["required_replications"] is None
    assert "missing replication file" in report["problems"][0]
    json.dumps(report, allow_nan=False)


def test_no_recorded_cells_produces_json_safe_incomplete_report():
    summary, report = summarize_pilot_cells([_cell()], {})
    assert summary.loc[0, "test_ibs_sd"] is None
    assert report["metrics"]["test_ibs"]["max_sd"] is None
    assert report["metrics"]["test_ibs"]["provisional"]
    assert report["metrics"]["test_ibs"]["meets_target"] is None
    assert report["decision"] == "incomplete"
    json.dumps(report, allow_nan=False)


def test_missing_planned_replication_is_incomplete_even_with_four_successes():
    summary, report = summarize_pilot_cells([_cell()], {0: _frame().iloc[:4]})
    assert summary.loc[0, "successful_replications"] == 4
    assert not report["complete"]
    assert any("missing replication IDs [4]" in problem for problem in report["problems"])


def test_complete_pilot_can_reject_proposed_final_count():
    _, report = summarize_pilot_cells([_cell()], {0: _frame(ibs_sd=0.1)}, final_replications=250)
    assert report["complete"]
    assert report["decision"] == "insufficient_precision"
    assert report["required_replications"] == 1112
    assert not report["ready_for_final"]


def test_cure_allocation_uses_mse_column_even_when_rmse_is_very_different():
    _, report = summarize_pilot_cells([_cell()], {0: _frame(mse_sd=0.1)})
    assert report["metrics"]["cure_mse"]["required_replications"] == 400
    assert report["required_replications"] == 400


@pytest.mark.parametrize("seed_column", ["generation_seed", "split_seed", "inner_seed"])
def test_paired_configurations_must_share_all_replication_seeds(seed_column):
    cells = [_cell(), _cell(1, model_name="bn", kernel="linear")]
    frames = {0: _frame(cells[0]), 1: _frame(cells[1])}
    frames[1].loc[3, seed_column] += 1
    with pytest.raises(ValueError, match="inconsistent shared replication seeds"):
        summarize_pilot_cells(cells, frames)


def test_different_scenarios_can_have_different_seeds_and_rows_can_be_unordered():
    cells = [_cell(), _cell(1, method=2)]
    frames = {0: _frame(cells[0]).iloc[::-1], 1: _frame(cells[1])}
    frames[1]["generation_seed"] += 1000
    _, report = summarize_pilot_cells(cells, frames)
    assert report["complete"]


@pytest.mark.parametrize("column,value", [
    ("n", 600), ("method", 2), ("generating_distribution", "negative_binomial"),
    ("generating_phi", 0.25), ("model", "bn"), ("kernel", "wavelet"),
])
def test_cell_identity_must_match_planned_design(column, value):
    frame = _frame()
    frame.loc[2, column] = value
    with pytest.raises(ValueError, match="does not match the planned cell"):
        summarize_pilot_cells([_cell()], {0: frame})


def test_negative_binomial_phi_and_baseline_missing_kernel_are_validated():
    cell = _cell(model_name="mixture_cure", kernel=None, generating_name="nb025",
                 generating_distribution="negative_binomial", generating_phi=0.25)
    frame = _frame(cell)
    frame["kernel"] = np.nan
    _, report = summarize_pilot_cells([cell], {0: frame})
    assert report["complete"]
    frame.loc[0, "generating_phi"] = 1.0
    with pytest.raises(ValueError, match="generating_phi"):
        summarize_pilot_cells([cell], {0: frame})


@pytest.mark.parametrize("column,value,error", [
    ("replication", 1, "duplicate replication"),
    ("replication", 5, "must be in 0..4"),
    ("replication", -1, "nonnegative integer"),
    ("replication", 0.5, "nonnegative integer"),
    ("replication", np.nan, "nonnegative integer"),
    ("status", "unknown", "status must be"),
    ("status", None, "status must be"),
    ("generation_seed", np.nan, "nonnegative integer"),
    ("split_seed", 2.5, "nonnegative integer"),
    ("inner_seed", -1, "nonnegative integer"),
    ("cure_mse", "invalid", "nonnumeric values"),
])
def test_malformed_rows_are_rejected(column, value, error):
    frame = _frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        summarize_pilot_cells([_cell()], {0: frame})


def test_incomplete_schema_duplicate_headers_and_unplanned_tasks_are_rejected():
    frame = _frame()
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_pilot_cells([_cell()], {0: frame.drop(columns="cure_mse")})
    with pytest.raises(ValueError, match="duplicate column names"):
        summarize_pilot_cells([_cell()], {0: pd.concat([frame, frame[["replication"]]], axis=1)})
    with pytest.raises(ValueError, match="unplanned task IDs"):
        summarize_pilot_cells([_cell()], {1: frame})
    with pytest.raises(ValueError, match="duplicate task IDs"):
        summarize_pilot_cells([_cell(), _cell()], {})


@pytest.mark.parametrize("config", [
    {"planned_replications": 0}, {"planned_replications": 5.5},
    {"minimum_successful": 1}, {"minimum_successful": 6},
    {"final_replications": -1}, {"final_replications": True},
    {"ibs_target": 0}, {"ibs_target": np.nan}, {"ibs_target": np.inf},
    {"cure_mse_target": -1}, {"cure_mse_target": False},
])
def test_configuration_requires_valid_counts_and_finite_positive_targets(config):
    with pytest.raises(ValueError):
        summarize_pilot_cells([_cell()], {0: _frame()}, **config)


def test_constant_successful_metrics_require_at_least_one_final_replication():
    _, report = summarize_pilot_cells([_cell()], {0: _frame(ibs_sd=0, mse_sd=0, auc_sd=0)})
    assert report["required_replications"] == 1
    assert report["metrics"]["test_ibs"]["projected_mcse"] == 0
