"""Check PCM plotting against the experiment's saved CSV and NPZ formats."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("matplotlib")

PAIRED_SCRIPTS = (
    ("create_paired_plot_1_2_3_methods", "calculate_paired_difference", "integrated_brier"),
    ("create_paired_plot_interaction_quadratic_surfaces", "paired_difference", "ibs"),
)


@pytest.mark.parametrize("script_name, function_name, metric", PAIRED_SCRIPTS)
def test_paired_metrics_match_replication_ids_and_drop_nonfinite_pairs(
    tmp_path, script_name, function_name, metric,
):
    script = importlib.import_module(f"scripts.{script_name}")
    focal = tmp_path / "focal.csv"
    comparator = tmp_path / "comparator.csv"
    pd.DataFrame({
        "replication": [10, 20, 30, 40],
        "status": ["ok", "ok", "failed", "ok"],
        "test_ibs": [0.2, 0.5, np.nan, 0.7],
    }).to_csv(focal, index=False)
    pd.DataFrame({
        "replication": [20, 40, 10, 30],
        "status": ["ok", "ok", "ok", "failed"],
        "test_ibs": [0.2, np.inf, 0.1, np.nan],
    }).to_csv(comparator, index=False)

    result = getattr(script, function_name)(focal, comparator, metric=metric)

    assert result["mcse"] == pytest.approx(0.1)
    if script_name == "create_paired_plot_1_2_3_methods":
        assert result["pairs"] == 2
        assert result["mean_difference_a_minus_b"] == pytest.approx(0.2)
        assert result["ci95_lower"] == pytest.approx(0.004)
        assert result["ci95_upper"] == pytest.approx(0.396)
    else:
        assert result["paired_n"] == 2
        assert result["mean_difference"] == pytest.approx(0.2)
        assert result["halfwidth_95"] == pytest.approx(0.196)


@pytest.mark.parametrize("script_name, function_name, metric", PAIRED_SCRIPTS)
def test_pairing_rejects_duplicate_replications(
    tmp_path, script_name, function_name, metric,
):
    script = importlib.import_module(f"scripts.{script_name}")
    focal = tmp_path / "focal.csv"
    comparator = tmp_path / "comparator.csv"
    pd.DataFrame({"replication": [1, 1], "test_ibs": [0.1, 0.2]}).to_csv(
        focal, index=False,
    )
    pd.DataFrame({"replication": [1, 2], "test_ibs": [0.2, 0.3]}).to_csv(
        comparator, index=False,
    )

    with pytest.raises(pd.errors.MergeError):
        getattr(script, function_name)(focal, comparator, metric=metric)


@pytest.mark.parametrize("script_name, function_name, metric", PAIRED_SCRIPTS)
def test_filename_parser_accepts_pcm_spline_output(
    script_name, function_name, metric,
):
    script = importlib.import_module(f"scripts.{script_name}")
    path = Path("replications_task959_spline_bn_n600_method5_gennb100_kernelparametric.csv")
    parsed = script.parse_filename(path)

    assert parsed["model"] == "spline_bn"
    assert parsed["n"] == 600
    assert parsed["method"] == 5
    assert parsed["generating"] == "nb100"
    assert parsed["kernel"] == "parametric"


@pytest.mark.parametrize("method", [1, 2, 3])
def test_surface_truth_agrees_with_pcm_generator(method):
    from evaluate.simulated_data import cure_probability
    from scripts.recover_cure_surfaces_figure import true_cure_surface

    axis = np.linspace(-1.0, 1.0, 11)
    x1, x2 = np.meshgrid(axis, axis)

    np.testing.assert_allclose(
        true_cure_surface(method, x1, x2),
        cure_probability(x1, x2, method),
        rtol=1e-14,
    )


@pytest.mark.parametrize("has_successful_fits", [True, False])
def test_recovery_reads_current_pcm_diagnostics(tmp_path, has_successful_fits):
    from scripts import recover_cure_surfaces_figure as script

    diagnostics_dir = tmp_path / "diagnostics"
    diagnostics_dir.mkdir()
    stem = "task21_poisson_n600_method2_genpoisson_kernellinear"
    axis = np.linspace(-1.0, 1.0, 41)
    x1, x2 = np.meshgrid(axis, axis)
    truth = script.true_cure_surface(2, x1, x2)
    grid = np.column_stack((x1.ravel(), x2.ravel()))
    permutation = np.random.default_rng(3).permutation(len(grid))

    if has_successful_fits:
        predictions = np.stack((truth.ravel() + 0.04, truth.ravel() - 0.04))
        replication = np.array([0, 2])
        status = ["ok", "failed", "ok"]
    else:
        predictions = np.empty((0, 41**2))
        replication = np.array([], dtype=int)
        status = ["failed", "failed", "failed"]

    np.savez_compressed(
        diagnostics_dir / f"diagnostics_{stem}.npz",
        replication=replication,
        surface_axis=axis,
        surface_grid_raw=grid[permutation],
        true_cure_surface=truth.ravel()[permutation],
        predicted_cure_surface=predictions[:, permutation],
    )
    pd.DataFrame({"replication": [0, 1, 2], "status": status}).to_csv(
        tmp_path / f"replications_{stem}.csv", index=False,
    )

    recovered = script.recovery_field(
        model="poisson", kernel="linear", surface=2, results_dir=tmp_path,
    )

    np.testing.assert_allclose(recovered["true_cure"], truth, atol=1e-12)
    if has_successful_fits:
        np.testing.assert_allclose(recovered["mean_cure"], truth, atol=1e-12)
        np.testing.assert_allclose(recovered["rmse"], 0.04, atol=1e-12)
        assert np.all(recovered["n_valid"] == 2)
        assert recovered["n_total"] == 2
    else:
        assert np.isnan(recovered["mean_cure"]).all()
        assert np.isnan(recovered["rmse"]).all()
        assert np.all(recovered["n_valid"] == 0)
        assert recovered["n_total"] == 0
