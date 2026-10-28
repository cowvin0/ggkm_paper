"""Keep breast-cancer fitting comparable with the melanoma experiment."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from scripts import run_breast_cancer_experiment as breast
from scripts import run_melanoma_experiment as melanoma


def test_breast_tasks_match_melanoma_without_sigmoid():
    assert breast.KERNELS == melanoma.KERNELS
    assert "sigmoid" not in breast.KERNELS
    assert breast.build_experiments() == melanoma.build_experiments()
    assert len(breast.build_experiments()) == 32
    with pytest.raises(ValueError, match="Unknown kernel: sigmoid"):
        breast.kernel_parameter_grid("poisson", "sigmoid")


@pytest.mark.parametrize("model_name", sorted(melanoma.KERNEL_MODELS))
@pytest.mark.parametrize("kernel", melanoma.KERNELS)
def test_breast_kernel_candidates_match_melanoma(model_name, kernel):
    assert breast.kernel_parameter_grid(model_name, kernel) == melanoma.kernel_parameter_grid(
        model_name, kernel
    )


@pytest.mark.parametrize("model_name", list(melanoma.MODELS))
def test_breast_model_parameters_match_melanoma(model_name):
    assert breast.MODELS[model_name]().get_params(deep=False) == melanoma.MODELS[
        model_name
    ]().get_params(deep=False)


def test_breast_validation_settings_match_melanoma_with_cohort_specific_holdout():
    for setting in (
        "N_OUTER_SPLITS", "N_INNER_SPLITS", "INNER_GRID_POINTS", "OUTER_GRID_POINTS",
        "OBSERVED_TIME_QUANTILE", "MINIMUM_CENSORING_SURVIVAL", "MASTER_SEED",
        "KERNEL_PENALTIES", "KERNEL_GAMMAS", "POLYNOMIAL_DEGREES",
        "POLYNOMIAL_COEF0", "WAVELET_A_VALUES", "BINOMIAL_K", "SPLINE_PARAMETER_GRID",
    ):
        assert getattr(breast, setting) == getattr(melanoma, setting), setting
    assert breast.OUTER_TEST_SIZE == 0.30


def test_breast_runner_plots_and_table_share_the_updated_results_directory():
    from scripts import breast_cancer_plot_utils, create_clinical_performance_tables

    assert breast.OUTPUT_DIR == breast_cancer_plot_utils.RESULTS_DIR
    assert breast.OUTPUT_DIR == create_clinical_performance_tables.BREAST_CANCER_RESULTS_DIR
    assert breast.OUTPUT_DIR.name == "grid_search_paired_50_melanoma_design"


def test_breast_pbs_array_covers_the_complete_model_grid():
    project_root = Path(__file__).resolve().parents[1]
    script = (project_root / "pbs_jobs" / "run_breast_cancer_experiment.pbs").read_text()
    match = re.search(r"^#PBS -J (\d+)-(\d+)$", script, re.MULTILINE)

    assert match is not None
    assert (int(match[1]), int(match[2])) == (0, len(breast.build_experiments()) - 1)


@pytest.fixture
def breast_checkpoint(tmp_path):
    plot_file = tmp_path / "split_000.npz"
    np.savez_compressed(plot_file, cure_probability_test=np.array([0.3, 0.7]))
    row = {
        "split": 0, "model_name": "poisson", "kernel": "rbf", "status": "ok",
        "train_index_hash": "train", "test_index_hash": "test",
        "plot_data_file": str(plot_file),
    }
    path = tmp_path / "split_000.json"
    return path, row


def _load_checkpoint(path):
    return breast.load_valid_checkpoint(
        path, split_index=0, model_name="poisson", kernel="rbf",
        train_hash="train", test_hash="test",
    )


def test_legacy_checkpoint_is_not_reused_after_tuning_design_changes(breast_checkpoint):
    path, row = breast_checkpoint
    breast.atomic_write_json(path, row)
    assert _load_checkpoint(path) is None


def test_matching_checkpoint_is_reused(breast_checkpoint):
    path, row = breast_checkpoint
    row["experiment_settings_hash"] = breast.experiment_settings_hash()
    breast.atomic_write_json(path, row)
    assert _load_checkpoint(path) == row


@pytest.mark.parametrize("setting,new_value", [
    ("N_INNER_SPLITS", 4),
    ("WAVELET_A_VALUES", [0.5, 1.0, 2.0]),
    ("MASTER_SEED", 20260818),
    ("SPLINE_PARAMETER_GRID", [{"n_knots": 3, "degree": 3, "lambda_reg": 3.0}]),
])
def test_changed_design_cannot_reuse_cached_fits(
    monkeypatch, breast_checkpoint, setting, new_value,
):
    path, row = breast_checkpoint
    row["experiment_settings_hash"] = breast.experiment_settings_hash()
    breast.atomic_write_json(path, row)
    monkeypatch.setattr(breast, setting, new_value)
    assert _load_checkpoint(path) is None


def test_failed_holdout_records_current_design(monkeypatch, tmp_path):
    def fail_tuning(*args, **kwargs):
        raise RuntimeError("deliberate tuning failure")

    monkeypatch.setattr(breast, "tune_model", fail_tuning)
    path = tmp_path / "checkpoints" / "split_000.json"
    row = breast.run_outer_split_worker(
        pd.DataFrame({"delta": [0, 1] * 4}), "promotion_time", None, 0,
        np.arange(6), np.arange(6, 8), path, tmp_path / "split_000.npz",
    )

    assert row["status"] == "failed"
    assert "deliberate tuning failure" in row["failure_reason"]
    assert row["experiment_settings_hash"] == breast.experiment_settings_hash()
    assert json.loads(path.read_text())["experiment_settings_hash"] == row[
        "experiment_settings_hash"
    ]
