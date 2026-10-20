"""Verify saved melanoma predictions, patient summaries, and optional refits."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from scripts import melanoma_auc_plot as auc_plot
from scripts import melanoma_individual_cure_variation as individual_plot
from scripts import melanoma_refit_utils as refit
from scripts.melanoma_plot_utils import index_hash, read_split_file


@pytest.fixture
def saved_holdouts(tmp_path):
    results = tmp_path / "results"
    archive_dir = results / "plot_data" / "melanoma_task0_poisson_kernel_rbf"
    archive_dir.mkdir(parents=True)
    splits = {
        0: (np.array([0, 2]), np.array([3, 1])),
        1: (np.array([0, 1]), np.array([2, 3])),
    }
    rows = []
    for split, (train, test) in splits.items():
        grid = np.array([0.0, 1.0, 2.0]) if split == 0 else np.array([0.0, 2.0, 4.0])
        curve = [1, 0.8, 0.6] if split == 0 else [1, 0.5, 0.2]
        archive = archive_dir / f"split_{split:03d}.npz"
        np.savez_compressed(
            archive, split=split, model_name="poisson", kernel="rbf",
            train_indices=train, test_indices=test,
            evaluation_time_grid=grid, evaluation_horizon=grid[-1],
            survival_test=np.tile(curve, (len(test), 1)),
            cure_probability_test=[0.8, 0.2] if split == 0 else [0.6, 0.4],
        )
        rows.append({
            "split": split, "model_name": "poisson", "kernel": "rbf",
            "status": "ok", "converged": True, "horizon": grid[-1],
            "integrated_brier": 0.1 + 0.2 * split, "cure_auc": 0.6 + 0.2 * split,
            "best_params": '{"lambda_reg": 3}',
            "train_index_hash": index_hash(train), "test_index_hash": index_hash(test),
            "plot_data_file": str(archive),
        })
    path = results / "splits_melanoma_task0_poisson_kernel_rbf.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    spec = {
        "name": "GG-KM P", "model_name": "poisson", "kernel": "rbf", "path": path,
        "label": "GG-KM\nP", "box_label": "GG-KM\nP", "color": "#0072B2",
        "hatch": None, "linestyle": "-",
    }
    df = pd.DataFrame({"time": [100, 200, 300, 400], "delta": [0, 1, 0, 1], "x": [0, 1, 2, 3]})
    return results, splits, spec, df


def test_individual_summaries_use_original_patient_ids_and_sample_sd(saved_holdouts):
    results, splits, spec, df = saved_holdouts

    predictions = individual_plot.collect_individual_predictions(
        [spec], df=df, results_dir=results, outer_splits=splits,
    )
    statistics = individual_plot.summarize_individual_predictions(
        predictions, n_patients=len(df),
    ).set_index("patient_index")

    assert predictions["patient_index"].tolist() == [3, 1, 2, 3]
    assert statistics.loc[3, "n_test_appearances"] == 2
    assert statistics.loc[3, "mean_predicted_cure"] == pytest.approx(0.6)
    assert statistics.loc[3, "sd_predicted_cure"] == pytest.approx(np.sqrt(0.08))
    assert statistics.loc[1, "mean_predicted_cure"] == pytest.approx(0.2)
    assert statistics.loc[2, "mean_predicted_cure"] == pytest.approx(0.6)
    assert np.isnan(statistics.loc[1, "sd_predicted_cure"])
    assert statistics.loc[0, "n_test_appearances"] == 0
    assert np.isnan(statistics.loc[0, "mean_predicted_cure"])
    assert np.isnan(statistics.loc[0, "sd_predicted_cure"])


def test_duplicate_patient_predictions_are_not_counted_twice():
    predictions = pd.DataFrame({
        "method": ["P", "P"], "patient_index": [3, 3], "split": [0, 0],
        "cure_probability": [0.2, 0.3],
    })
    with pytest.raises(ValueError, match="Duplicate patient predictions"):
        individual_plot.summarize_individual_predictions(predictions)


def test_survival_curves_use_common_supported_horizon_and_saved_scores(saved_holdouts):
    results, splits, spec, df = saved_holdouts

    reconstructed = auc_plot.reconstruct_results(
        df, [spec], results_dir=results, outer_splits=splits,
    )

    assert reconstructed["common_horizon"] == 2
    assert len(reconstructed["t_grid"]) == 250
    np.testing.assert_allclose(
        reconstructed["survival_curves"][spec["name"]],
        1 - 0.225 * reconstructed["t_grid"],
    )
    np.testing.assert_allclose(reconstructed["ibs_values"][spec["name"]], [0.1, 0.3])
    np.testing.assert_allclose(reconstructed["auc_values"][spec["name"]], [0.6, 0.8])


def test_kaplan_meier_accounts_for_tied_events_and_censoring():
    time, survival = auc_plot.kaplan_meier([1, 1, 2, 3], [1, 0, 1, 0], 3)

    np.testing.assert_array_equal(time, [0, 1, 2, 3])
    np.testing.assert_allclose(survival, [1, 0.75, 0.375, 0.375])


def test_existing_archives_do_not_trigger_refits(saved_holdouts, monkeypatch):
    results, splits, spec, df = saved_holdouts
    row = read_split_file(spec["path"]).iloc[0]
    monkeypatch.setattr(refit, "_refit_predictions", lambda *args, **kwargs: pytest.fail("unexpected refit"))

    predictions = refit.obtain_split_predictions(
        row, df=df, model_name="poisson", kernel="rbf", outer_splits=splits,
        results_dir=results, refit_missing=True,
    )

    np.testing.assert_array_equal(predictions["test_indices"], [3, 1])


def test_legacy_results_require_explicit_refit_option(saved_holdouts, monkeypatch):
    results, splits, spec, df = saved_holdouts
    row = read_split_file(spec["path"]).iloc[0]
    (results / "plot_data" / "melanoma_task0_poisson_kernel_rbf" / "split_000.npz").unlink()
    monkeypatch.setattr(refit, "_refit_predictions", lambda *args, **kwargs: pytest.fail("unexpected refit"))

    with pytest.raises(FileNotFoundError, match="--refit-missing"):
        refit.obtain_split_predictions(
            row, df=df, model_name="poisson", kernel="rbf", outer_splits=splits,
            results_dir=results,
        )


def test_bad_archives_are_not_silently_refitted(saved_holdouts, monkeypatch):
    results, splits, spec, df = saved_holdouts
    row = read_split_file(spec["path"]).iloc[0]
    archive = results / "plot_data" / "melanoma_task0_poisson_kernel_rbf" / "split_000.npz"
    archive.write_bytes(b"not a numpy archive")
    monkeypatch.setattr(refit, "_refit_predictions", lambda *args, **kwargs: pytest.fail("unexpected refit"))

    with pytest.raises(ValueError, match="Could not read melanoma prediction archive"):
        refit.obtain_split_predictions(
            row, df=df, model_name="poisson", kernel="rbf", outer_splits=splits,
            results_dir=results, refit_missing=True,
        )


def test_predictions_must_match_the_requested_method(saved_holdouts, monkeypatch):
    results, splits, spec, df = saved_holdouts
    row = read_split_file(spec["path"]).iloc[0]
    monkeypatch.setattr(refit, "_refit_predictions", lambda *args, **kwargs: pytest.fail("unexpected refit"))

    with pytest.raises(ValueError, match="do not match the requested model"):
        refit.obtain_split_predictions(
            row, df=df, model_name="bn", kernel="rbf", outer_splits=splits,
            results_dir=results, refit_missing=True,
        )


def test_refits_use_saved_parameters_and_training_only_preprocessing(saved_holdouts, monkeypatch):
    results, splits, spec, df = saved_holdouts
    row = read_split_file(spec["path"]).iloc[0]
    (results / "plot_data" / "melanoma_task0_poisson_kernel_rbf" / "split_000.npz").unlink()
    calls = {}

    class Preprocessor:
        def fit(self, training):
            calls["preprocessor_training_rows"] = training["x"].tolist()
            return training[["x"]].to_numpy(), training["time"].to_numpy(), training["delta"].to_numpy()

        def transform(self, testing):
            calls["preprocessor_test_rows"] = testing["x"].tolist()
            return testing[["x"]].to_numpy(), testing["time"].to_numpy(), testing["delta"].to_numpy()

    class Model:
        def predict_survival(self, X, grid):
            return np.tile(np.exp(-grid), (len(X), 1))

        def predict_cure_probability(self, X):
            return np.full(len(X), 0.4)

    def build_model(estimator, kernel, params):
        calls["estimator"] = estimator
        calls["kernel"] = kernel
        calls["params"] = params
        return Model()

    def fit_model(model, X, time, event):
        calls["fit_X"] = X.copy()
        calls["fit_time"] = time.copy()

    runner = SimpleNamespace(
        MelanomaSurvivalPreprocessor=Preprocessor, MODELS={"poisson": "registered-model"},
        build_model=build_model, fit_cure_model=fit_model,
    )
    monkeypatch.setattr(refit.importlib, "import_module", lambda name: runner)
    grid = np.array([0.0, 0.5, 1.0])

    predictions = refit.obtain_split_predictions(
        row, df=df, model_name="poisson", kernel="rbf", outer_splits=splits,
        results_dir=results, refit_missing=True, prediction_grid=grid,
    )

    assert calls["preprocessor_training_rows"] == [0, 2]
    assert calls["preprocessor_test_rows"] == [3, 1]
    assert calls["params"] == {"lambda_reg": 3}
    assert calls["kernel"] == "rbf"
    assert calls["estimator"] == "registered-model"
    np.testing.assert_array_equal(calls["fit_X"][:, 0], [0, 2])
    np.testing.assert_array_equal(calls["fit_time"], [100, 300])
    np.testing.assert_array_equal(predictions["test_indices"], [3, 1])
    np.testing.assert_array_equal(predictions["evaluation_time_grid"], grid)
    assert predictions["survival_test"].shape == (2, 3)
