"""Check FOSP's paired design and training-only survival target encoding."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from scripts import run_breast_cancer_experiment as breast
from scripts import run_fosp_melanoma_experiment as fosp
from scripts import run_melanoma_experiment as melanoma
from scripts import fosp_melanoma_plot_utils as plots
from scripts.fosp_melanoma_data import load_fosp_melanoma
from scripts.preprocessing import FOSPMelanomaTargetEncodedPreprocessor


@pytest.fixture
def cohort():
    size = 40
    return pd.DataFrame({
        "id": np.arange(size) + 100,
        "idade": np.arange(size) + 20,
        "anodiag": 2000 + np.arange(size) % 15,
        "female": np.arange(size) % 2,
        "cirurgia": np.arange(size) % 2,
        "radioterapia": np.arange(size) % 3 == 0,
        "quimioterapia": np.arange(size) % 4 == 0,
        "hormonio": np.zeros(size),
        "imuno": np.zeros(size),
        "estadio": np.tile(["I", "II", "III", "IV"], size // 4),
        "topo": np.tile(["C440", "C441"], size // 2),
        "descmorfo": [f"morphology-{index}" for index in range(size)],
        "tempo": np.arange(size, dtype=float) + 0.5,
        "status": (np.arange(size) + 1) % 2,
        "tempo_ext": np.arange(size, dtype=float) + 1.0,
        "status_ext": np.arange(size) % 2,
    })


def test_loader_and_plots_use_extended_outcomes_and_preserve_rows(cohort, tmp_path):
    path = tmp_path / "fosp.csv"
    cohort.to_csv(path, index=False)
    loaded = load_fosp_melanoma(path)
    np.testing.assert_array_equal(loaded.id, cohort.id)
    np.testing.assert_array_equal(loaded.delta, cohort.status_ext)
    times, events = plots.full_cohort_times(loaded)
    np.testing.assert_array_equal(times, cohort.tempo_ext)
    np.testing.assert_array_equal(events, cohort.status_ext)
    assert len(plots.construct_outer_splits(loaded)[0][1]) == 12


def test_tuning_candidates_and_estimator_parameters_match_existing_cohorts():
    assert fosp.build_experiments() == melanoma.build_experiments() == breast.build_experiments()
    assert len(fosp.build_experiments()) == 32
    for name in melanoma.MODELS:
        assert fosp.MODELS[name]().get_params() == melanoma.MODELS[name]().get_params()
        for kernel in melanoma.KERNELS:
            assert fosp.kernel_parameter_grid(name, kernel) == melanoma.kernel_parameter_grid(name, kernel)
    for setting in (
        "N_OUTER_SPLITS", "N_INNER_SPLITS", "INNER_GRID_POINTS", "OUTER_GRID_POINTS",
        "OBSERVED_TIME_QUANTILE", "MINIMUM_CENSORING_SURVIVAL", "MASTER_SEED",
        "SPLINE_PARAMETER_GRID", "WAVELET_A_VALUES",
    ):
        assert getattr(fosp, setting) == getattr(melanoma, setting)
    assert fosp.OUTER_TEST_SIZE == breast.OUTER_TEST_SIZE == 0.30
    assert fosp.WAVELET_A_VALUES == [0.5, 1, 2, 10, 15, 20]


def test_preprocessing_encodes_three_categories_and_keeps_binary_features(cohort):
    preprocessor = FOSPMelanomaTargetEncodedPreprocessor()
    features, times, events = preprocessor.fit(cohort)
    assert features.shape == (len(cohort), 11)
    assert preprocessor.get_feature_names_out().tolist()[-3:] == [
        "estadio_rmst", "topo_rmst", "descmorfo_rmst",
    ]
    np.testing.assert_array_equal(features[:, 2:8], cohort[preprocessor.binary_cols_])
    np.testing.assert_array_equal(times, cohort.tempo_ext)
    np.testing.assert_array_equal(events, cohort.status_ext)
    assert np.isfinite(features).all()
    for encoder in preprocessor.categorical_encoders_.values():
        assert (encoder.smooth, encoder.cv, encoder.random_state) == (10.0, 5, 42)


def test_training_row_cannot_encode_its_own_outcome(cohort):
    original = FOSPMelanomaTargetEncodedPreprocessor().fit(cohort)[0]
    changed = cohort.copy()
    # Keep the overall maximum follow-up fixed so the common RMST horizon is unchanged.
    changed.loc[0, "tempo_ext"] = 2.5
    changed.loc[0, "status_ext"] = 1
    encoded = FOSPMelanomaTargetEncodedPreprocessor().fit(changed)[0]
    np.testing.assert_allclose(encoded[0, -3:], original[0, -3:])


def test_validation_outcomes_and_unseen_categories_cannot_change_fitted_maps(cohort):
    preprocessor = FOSPMelanomaTargetEncodedPreprocessor()
    preprocessor.fit(cohort.iloc[:30])
    test = cohort.iloc[30:].copy()
    test["topo"] = "UNSEEN"
    test["descmorfo"] = "UNSEEN"
    original = preprocessor.transform(test)[0]
    test["tempo_ext"] = 1000.0
    test["status_ext"] = 1 - test.status_ext
    changed = preprocessor.transform(test)[0]
    np.testing.assert_array_equal(changed, original)
    for index, column in ((-2, "topo"), (-1, "descmorfo")):
        np.testing.assert_allclose(
            changed[:, index], preprocessor.categorical_encoders_[column].global_rmst_,
        )
    assert preprocessor.scaler_.mean_[0] == pytest.approx(cohort.iloc[:30].idade.mean())


def test_fosp_pbs_and_artifact_paths_cover_its_own_design():
    from scripts import create_clinical_performance_tables as tables

    assert fosp.OUTPUT_DIR == plots.RESULTS_DIR == tables.FOSP_MELANOMA_RESULTS_DIR
    script = (Path(fosp.PROJECT_ROOT) / "pbs_jobs/run_fosp_melanoma_experiment.pbs").read_text()
    bounds = re.search(r"^#PBS -J (\d+)-(\d+)$", script, re.MULTILINE)
    assert bounds and (int(bounds[1]), int(bounds[2])) == (0, 31)
    assert fosp.experiment_prefix(0, "poisson", "rbf") == "fosp_melanoma_task0_poisson_kernel_rbf"


@pytest.mark.parametrize("runner", [melanoma, fosp])
def test_changed_wavelet_grid_invalidates_saved_checkpoint(runner, monkeypatch, tmp_path):
    archive = tmp_path / "split.npz"
    np.savez_compressed(archive, prediction=[0.5])
    checkpoint = tmp_path / "split.json"
    row = {
        "split": 0, "model_name": "poisson", "kernel": "wavelet", "status": "ok",
        "train_index_hash": "train", "test_index_hash": "test",
        "plot_data_file": str(archive), "experiment_settings_hash": runner.experiment_settings_hash(),
    }
    checkpoint.write_text(json.dumps(row))
    arguments = dict(split_index=0, model_name="poisson", kernel="wavelet", train_hash="train", test_hash="test")
    assert runner.load_valid_checkpoint(checkpoint, **arguments) == row
    monkeypatch.setattr(runner, "WAVELET_A_VALUES", [0.5, 1, 2, 10])
    assert runner.load_valid_checkpoint(checkpoint, **arguments) is None
