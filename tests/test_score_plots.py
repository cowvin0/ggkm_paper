"""Standalone score exports preserve saved metrics and original model choices."""

import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import create_score_plots as plots
from scripts.table_utils.clinical import methods


def _clinical_results(directory, dataset):
    directory.mkdir()
    prefix = dataset.replace("-", "_")
    for task, method in enumerate(methods()):
        suffix = "" if method.kernel == "parametric" else f"_kernel_{method.kernel}"
        pd.DataFrame({
            "split": [0, 1, 2, 3], "status": ["ok", "failed", "ok", "ok"],
            "converged": [True, False, True, False], "horizon": [2.0] * 4,
            "model_name": [method.model] * 4, "kernel": [method.kernel] * 4,
            "integrated_brier": [0.2, 999, 0.3, np.nan],
            "uno_c_index": [0.6, 999, np.nan, 0.8],
            "cure_auc": np.array([0.7, 0.5, 0.8, 0.75]) + (0.1 if method.kernel == "wavelet" else 0),
            "mean_predicted_cure": [0.4] * 4, "runtime_seconds": [1.0] * 4,
            "best_params": ["{}"] * 4,
            "train_index_hash": ["train0", "train1", "train2", "train3"],
            "test_index_hash": ["test0", "test1", "test2", "test3"],
        }).to_csv(directory / f"splits_{prefix}_task{task}_{method.model}{suffix}.csv", index=False)


def _assert_figure_formats(paths, count):
    images = [path for path in paths if path.suffix in {".png", ".pdf", ".svg"}]
    assert len(images) == count
    for path in images:
        assert path.stat().st_size > 100
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF-")
        elif path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        else:
            assert "<svg" in path.read_text()


@pytest.mark.parametrize("dataset", ["melanoma", "breast-cancer", "fosp-melanoma"])
def test_clinical_exports_metrics_separately_without_archives_or_overwriting_auc(tmp_path, dataset):
    directory, output = tmp_path / "results", tmp_path / "figures"
    _clinical_results(directory, dataset)
    output.mkdir()
    prefix = dataset.replace("-", "_")
    existing = output / f"{prefix}_predictive_assessment_ibs_auc.png"
    existing.write_bytes(b"original IBS and AUC plot")

    paths = plots.generate_clinical_score_plots(dataset, results_dir=directory, output_dir=output)

    _assert_figure_formats(paths, 12)
    assert {path.stem for path in paths if path.suffix == ".pdf"} == {
        f"{prefix}_predictive_assessment_ibs", f"{prefix}_predictive_assessment_uno_c_index",
        f"{prefix}_ggkm_kernel_comparison_ibs", f"{prefix}_ggkm_kernel_comparison_uno_c_index",
    }
    assert existing.read_bytes() == b"original IBS and AUC plot"
    values = pd.read_csv(paths[-1])
    for metric, expected, splits in (("ibs", [0.2, 0.3], [0, 2]), ("uno_c_index", [0.6, 0.8], [0, 3])):
        group = values.loc[values.metric.eq(metric) & values.comparison.eq("models")]
        assert len(group) == 12  # Six models, two successful finite scores each.
        selected = group.loc[group.model_name.eq("poisson")]
        assert selected.kernel.tolist() == ["wavelet", "wavelet"]
        assert selected.split.tolist() == splits
        np.testing.assert_array_equal(selected.value, expected)
    assert len(values.loc[values.comparison.eq("kernels")]) == 56


@pytest.mark.parametrize("damage", ["missing_cindex", "undefined_cindex", "unpaired"])
def test_invalid_scores_fail_before_creating_outputs(tmp_path, damage):
    directory, output = tmp_path / "results", tmp_path / "figures"
    _clinical_results(directory, "melanoma")
    path = next(directory.glob("*mixture_cure.csv"))
    frame = pd.read_csv(path)
    if damage == "missing_cindex":
        frame = frame.drop(columns="uno_c_index")
    elif damage == "undefined_cindex":
        frame["uno_c_index"] = np.nan
    else:
        frame.loc[0, "test_index_hash"] = "unpaired"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        plots.generate_clinical_score_plots("melanoma", results_dir=directory, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("script_name,function,metric", [
    ("create_paired_plot_1_2_3_methods", "calculate_paired_difference", "uno_c_index"),
    ("create_paired_plot_interaction_quadratic_surfaces", "paired_difference", "cindex"),
])
def test_simulation_cindex_differences_keep_pair_ids_and_sign(tmp_path, script_name, function, metric):
    module = importlib.import_module(f"scripts.{script_name}")
    focal, comparator = tmp_path / "focal.csv", tmp_path / "comparator.csv"
    pd.DataFrame({"replication": [10, 20, 30], "test_cindex": [0.8, 0.9, np.nan]}).to_csv(focal, index=False)
    pd.DataFrame({"replication": [20, 30, 10], "test_cindex": [0.6, 0.1, 0.7]}).to_csv(comparator, index=False)
    result = getattr(module, function)(focal, comparator, metric=metric)
    mean_key = "mean_difference_a_minus_b" if metric == "uno_c_index" else "mean_difference"
    assert result[mean_key] == pytest.approx(0.2)
    assert result["mcse"] == pytest.approx(0.1)


def test_simulation_exports_standalone_primary_and_highdim_figures(tmp_path):
    from scripts.table_utils.data import ALL_METHODS, PRIMARY_SCENARIOS, HIGHDIM_SCENARIOS

    directory, output = tmp_path / "results", tmp_path / "figures"
    directory.mkdir()
    for scenario in (*PRIMARY_SCENARIOS, *HIGHDIM_SCENARIOS):
        for task, method in enumerate(ALL_METHODS):
            delta = 0.0 if method.kernel == "rbf" else 0.02
            pd.DataFrame({
                "replication": [0, 1, 2], "status": ["ok", "ok", "failed"],
                "test_ibs": [0.1 + delta, 0.15 + delta, np.nan],
                "test_cindex": [0.8 - delta, 0.9 - delta, np.nan],
            }).to_csv(directory / (
                f"replications_task{task}_{method.model}_n{scenario.n}_"
                f"method{scenario.method}_gen{scenario.generating}_kernel{method.kernel}.csv"
            ), index=False)

    paths = plots.generate_simulation_score_plots(results_dir=directory, output_dir=output)

    _assert_figure_formats(paths, 12)
    assert len(paths) == 17
    primary = pd.read_csv(output / "simulation_rbf_paired_differences_uno_c_index.csv")
    highdim = pd.read_csv(output / "highdim_rbf_vs_linear_spline_polynomial_paired_differences_uno_c_index.csv")
    np.testing.assert_allclose(primary.mean_difference_a_minus_b, 0.02)
    np.testing.assert_allclose(highdim.mean_difference, 0.02)
    assert set(primary.pairs) == set(highdim.paired_n) == {2}
    assert primary.metric.unique().tolist() == ["uno_c_index"]
    assert not list(output.glob("*with_auc*"))


def test_standalone_cli_help_works_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(plots.__file__).resolve()), "--help"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout
    assert "--results-dir" in result.stdout
