"""Validate simulation table summaries against synthetic PCM replication files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.table_utils.data import Method, Results, Scenario


SCENARIO = Scenario(1, "nb025", 300)
METHOD = Method("bn", "rbf")


def _replications():
    return pd.DataFrame({
        "replication": [2, 0, 1, 3],
        "status": ["ok", "ok", "failed", "ok"],
        "test_ibs": [0.1, 0.3, 999, np.nan],
        "cure_rmse": [0.2, 0.4, 999, 0.6],
        "test_auc": [0.8, 0.6, 999, 0.7],
        "converged": ["False", "True", "False", "1"],
        "runtime_seconds": [1.0, 2.0, 100.0, 4.0],
        "fitted_a": [1e-5, 1, np.nan, 2],
        "fitted_d": [1, 1, np.nan, 1],
        "fitted_p": [1, 1, np.nan, 1],
        "fitted_phi": [1e-8, 1.5e-8, np.nan, 0.3],
        "n": [300] * 4,
        "method": [1] * 4,
        "model": ["bn"] * 4,
        "kernel": ["rbf"] * 4,
        "generating_distribution": ["negative_binomial"] * 4,
        "generating_phi": [0.25] * 4,
        "generation_seed": [1002, 1000, 1001, 1003],
        "split_seed": [2002, 2000, 2001, 2003],
        "inner_seed": [3002, 3000, 3001, 3003],
    })


def _write_replications(directory: Path, frame=None, *, task=0):
    path = directory / f"replications_task{task}_bn_n300_method1_gennb025_kernelrbf.csv"
    (frame if frame is not None else _replications()).to_csv(path, index=False)
    return path


def test_metric_retains_finite_successes_and_replication_alignment(tmp_path):
    _write_replications(tmp_path)
    results = Results(tmp_path)

    assert results.metric(SCENARIO, METHOD, "test_ibs").to_dict() == {0: 0.3, 2: 0.1}
    assert results.summary(SCENARIO, METHOD, "test_ibs") == pytest.approx((0.2, 0.1, 2))
    assert results.summary(SCENARIO, METHOD, "cure_rmse") == pytest.approx(
        (0.4, np.std([0.2, 0.4, 0.6], ddof=1) / np.sqrt(3), 3),
    )


@pytest.mark.parametrize("values, expected_count", [([np.inf, np.nan, 99, -np.inf], 0), ([0.2, np.nan, 99, np.inf], 1)])
def test_sparse_metrics_report_undefined_mcse(tmp_path, values, expected_count):
    frame = _replications()
    frame["test_ibs"] = values
    _write_replications(tmp_path, frame)

    mean, mcse, count = Results(tmp_path).summary(SCENARIO, METHOD, "test_ibs")

    assert count == expected_count
    assert np.isnan(mcse)
    if expected_count:
        assert mean == pytest.approx(0.2)
    else:
        assert np.isnan(mean)


def test_loading_discards_large_json_fields_but_preserves_pairing_seeds(tmp_path):
    frame = _replications()
    for column in ("test_tpr", "test_fpr", "best_params", "error"):
        frame[column] = '["unused large JSON payload"]'
    _write_replications(tmp_path, frame)

    loaded = Results(tmp_path).frame(SCENARIO, METHOD)

    assert not set(("test_tpr", "test_fpr", "best_params", "error")) & set(loaded.columns)
    for column in ("generation_seed", "split_seed", "inner_seed"):
        pd.testing.assert_series_equal(loaded[column], frame[column])


@pytest.mark.parametrize("column", ["replication", "status", "test_ibs", "cure_rmse", "test_auc"])
def test_loading_rejects_missing_essential_columns(tmp_path, column):
    _write_replications(tmp_path, _replications().drop(columns=column))

    with pytest.raises(ValueError, match=f"missing required columns: {column}"):
        Results(tmp_path).frame(SCENARIO, METHOD)


@pytest.mark.parametrize("bad_ids", [[2, 0, 1, 2], [2, np.nan, 1, 3], [2, -1, 1, 3], [2, 0.5, 1, 3]])
def test_loading_rejects_ambiguous_replication_ids(tmp_path, bad_ids):
    frame = _replications()
    frame["replication"] = bad_ids
    _write_replications(tmp_path, frame)

    with pytest.raises(ValueError, match="replication IDs"):
        Results(tmp_path).frame(SCENARIO, METHOD)


def test_index_rejects_duplicate_experiment_configurations(tmp_path):
    _write_replications(tmp_path, task=0)
    _write_replications(tmp_path, task=99)

    with pytest.raises(ValueError, match="Duplicate PCM replication files"):
        Results(tmp_path)


def test_index_ignores_models_outside_the_tables(tmp_path):
    _write_replications(tmp_path)
    unsupported = tmp_path / "replications_task1_bernoulli_n300_method1_gennb025_kernelrbf.csv"
    unsupported.write_text("malformed unused result\n", encoding="utf-8")

    assert len(Results(tmp_path).frame(SCENARIO, METHOD)) == 4


@pytest.mark.parametrize("column, bad_value", [
    ("n", 600), ("method", 2), ("model", "poisson"), ("kernel", "linear"),
    ("generating_distribution", "poisson"), ("generating_phi", 1.0),
])
def test_loading_rejects_metadata_inconsistent_with_filename(tmp_path, column, bad_value):
    frame = _replications()
    frame.loc[0, column] = bad_value
    _write_replications(tmp_path, frame)

    with pytest.raises(ValueError, match="metadata does not match the filename"):
        Results(tmp_path).frame(SCENARIO, METHOD)


def test_loading_rejects_unrecognized_status_and_nonnumeric_metrics(tmp_path):
    frame = _replications()
    frame.loc[0, "status"] = "success"
    _write_replications(tmp_path, frame)
    with pytest.raises(ValueError, match="status must be explicitly"):
        Results(tmp_path).frame(SCENARIO, METHOD)

    frame.loc[0, "status"] = "ok"
    frame["test_auc"] = frame["test_auc"].astype(object)
    frame.loc[0, "test_auc"] = "bad measurement"
    _write_replications(tmp_path, frame)
    with pytest.raises(ValueError, match="contains nonnumeric values"):
        Results(tmp_path).frame(SCENARIO, METHOD)


def test_loading_rejects_duplicate_csv_headers(tmp_path):
    path = _write_replications(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("replication,status", "replication,replication", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate CSV column names"):
        Results(tmp_path).frame(SCENARIO, METHOD)


@pytest.mark.parametrize("convergence", [
    ["False", "True", "False", "True"], ["0", "1", "0", "1"],
    ["0.0", "1.0", "0.0", "1.0"], [False, True, False, True],
])
def test_diagnostics_parse_false_explicitly(tmp_path, convergence):
    frame = _replications()
    frame["converged"] = convergence
    _write_replications(tmp_path, frame)

    diagnostics = Results(tmp_path).diagnostics([SCENARIO], METHOD)

    assert diagnostics["converged"] == 2
    assert diagnostics["nonconverged"] == 1
    assert diagnostics["failed"] == 1
    assert diagnostics["total"] == 4


def test_diagnostics_count_bounds_with_tight_tolerance_and_runtime_for_failures(tmp_path):
    _write_replications(tmp_path)

    diagnostics = Results(tmp_path).diagnostics([SCENARIO], METHOD)

    assert diagnostics["latency_bound"] == 1
    # A dispersion of 1.5e-8 is not a contact with the 1e-8 bound.
    assert diagnostics["phi_bound"] == 1
    assert diagnostics["runtime_mean"] == pytest.approx((1 + 2 + 100 + 4) / 4)
    assert diagnostics["runtime_max"] == 100


@pytest.mark.parametrize("column", ["converged", "runtime_seconds", "fitted_a", "fitted_phi"])
def test_diagnostics_reject_missing_columns(tmp_path, column):
    _write_replications(tmp_path, _replications().drop(columns=column))

    with pytest.raises(ValueError, match="missing diagnostic columns"):
        Results(tmp_path).diagnostics([SCENARIO], METHOD)


@pytest.mark.parametrize("column, row, expected", [
    ("converged", 0, "missing convergence"),
    ("runtime_seconds", 2, "invalid runtime_seconds"),
    ("fitted_a", 0, "invalid latency parameters"),
    ("fitted_phi", 0, "invalid fitted_phi"),
])
def test_diagnostics_reject_missing_values_instead_of_inventing_counts(tmp_path, column, row, expected):
    frame = _replications()
    frame.loc[row, column] = np.nan
    _write_replications(tmp_path, frame)

    with pytest.raises(ValueError, match=expected):
        Results(tmp_path).diagnostics([SCENARIO], METHOD)


def test_loading_rejects_unknown_convergence_values(tmp_path):
    frame = _replications()
    frame.loc[0, "converged"] = "undetermined"
    _write_replications(tmp_path, frame)

    with pytest.raises(ValueError, match="invalid convergence value"):
        Results(tmp_path).frame(SCENARIO, METHOD)
