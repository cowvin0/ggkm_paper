"""Verify the breast-cancer plots use the breast cohort and holdout design."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedShuffleSplit

pytest.importorskip("matplotlib")

from scripts import breast_cancer_auc_plot as predictive
from scripts import breast_cancer_individual_cure_variation as individual
from scripts import breast_cancer_plot_utils as data
from scripts import breast_cancer_refit_utils as refit


DROP_COLUMNS = (
    "RhcDataDeNascimento", "Limitacao", "DiffEmDiasEntreRhcDtNascimentoDtObito",
    "Instrução", "EstCong", "HistFamCan", "SimDataDeNascimento", "CosemsCodIbge",
    "Renda", "Macroregião", "DIFFTRATCONS1", "DIFFDIAGTRAT", "DIFFDIAGCONS1",
    "Diagnostico", "Raça", "AnoCateg", "Idade", "Ano", "Estadiamento",
)


def test_breast_loader_retains_event_coding_deduplication_and_zero_time(tmp_path):
    raw = pd.DataFrame({
        **{column: [None] * 4 for column in DROP_COLUMNS},
        "Status": ["Morte por cancer de Mama", "Alive", "Other cause", "Morte por cancer de Mama"],
        "Tempo": [0.0, 12.0, 24.0, 0.0],
        "SimRecordID": [None] * 4, "RhcRecordID": [None] * 4,
        "SimCausaBasicaCategoria": [None] * 4, "SimCausaBasica": [None] * 4,
    })
    path = tmp_path / "cohort.csv"
    raw.to_csv(path, index=False)

    cohort = data.load_breast_cancer(path)

    assert cohort["delta"].tolist() == [1, 0, 0]
    assert cohort["Tempo"].tolist() == [0.5, 12.0, 24.0]
    assert cohort.index.tolist() == [0, 1, 2]
    assert cohort["SimRecordID"].tolist() == ["vivo"] * 3
    assert not set(DROP_COLUMNS) & set(cohort.columns)
    times, events = data.full_cohort_times(cohort)
    np.testing.assert_array_equal(times, [0.5, 12, 24])
    np.testing.assert_array_equal(events, [1, 0, 0])


@pytest.fixture
def saved_breast_holdouts(tmp_path):
    cohort = pd.DataFrame({"Tempo": np.arange(1, 21, dtype=float), "delta": [0, 1] * 10})
    splits = data.construct_outer_splits(cohort)
    results = tmp_path / "results"
    directory = results / "plot_data" / "breast_cancer_task7_poisson_kernel_rbf"
    directory.mkdir(parents=True)
    rows = []
    for split in (0, 1):
        train, test = splits[split]
        horizon = 12.0 + split
        grid = np.linspace(0, horizon, 4)
        cure = 0.1 + 0.6 * test / len(cohort) + 0.02 * split
        archive = directory / f"split_{split:03d}.npz"
        np.savez_compressed(
            archive, split=split, model_name="poisson", kernel="rbf",
            train_indices=train, test_indices=test,
            evaluation_time_grid=grid, evaluation_horizon=horizon,
            survival_test=np.tile(np.exp(-grid / 20), (len(test), 1)),
            cure_probability_test=cure,
        )
        rows.append({
            "split": split, "status": "ok", "converged": True,
            "model_name": "poisson", "kernel": "rbf", "horizon": horizon,
            "integrated_brier": 0.1 + 0.01 * split, "cure_auc": 0.7 + 0.01 * split,
            "best_params": json.dumps({"lambda_reg": 3.0}),
            "train_index_hash": data.index_hash(train), "test_index_hash": data.index_hash(test),
            "plot_data_file": f"/old/cluster/results/plot_data/{directory.name}/{archive.name}",
        })
    csv = results / "splits_breast_cancer_task7_poisson_kernel_rbf.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    spec = {
        "name": "GG-KM P", "label": "GG-KM\nP", "model_name": "poisson",
        "kernel": "rbf", "path": csv, "color": "#0072B2", "hatch": None,
    }
    return cohort, results, splits, spec


def test_breast_holdouts_use_thirty_percent_of_any_cohort(saved_breast_holdouts):
    cohort, _, splits, _ = saved_breast_holdouts
    expected = StratifiedShuffleSplit(n_splits=50, test_size=0.30, random_state=20260817)

    assert len(splits) == 50
    for split, (train, test) in enumerate(expected.split(cohort, cohort["delta"])):
        np.testing.assert_array_equal(splits[split][0], train)
        np.testing.assert_array_equal(splits[split][1], test)
        assert len(train) == 14
        assert len(test) == 6


def test_saved_breast_archives_and_individual_stats_keep_original_patient_ids(saved_breast_holdouts):
    cohort, results, splits, spec = saved_breast_holdouts
    predictions = individual.collect_individual_predictions(
        [spec], df=cohort, results_dir=results, outer_splits=splits,
    )
    statistics = individual.summarize_individual_predictions(
        predictions, n_patients=len(cohort),
    ).set_index("patient_index")

    assert predictions["patient_index"].tolist() == np.concatenate([splits[i][1] for i in (0, 1)]).tolist()
    for patient in range(len(cohort)):
        values = [0.1 + 0.6 * patient / len(cohort) + 0.02 * split for split in (0, 1) if patient in splits[split][1]]
        assert statistics.loc[patient, "n_test_appearances"] == len(values)
        if values:
            assert statistics.loc[patient, "mean_predicted_cure"] == pytest.approx(np.mean(values))
        else:
            assert np.isnan(statistics.loc[patient, "mean_predicted_cure"])
        if len(values) >= 2:
            assert statistics.loc[patient, "sd_predicted_cure"] == pytest.approx(np.std(values, ddof=1))
        else:
            assert np.isnan(statistics.loc[patient, "sd_predicted_cure"])


def test_breast_survival_reconstruction_preserves_native_horizon(saved_breast_holdouts):
    cohort, results, splits, spec = saved_breast_holdouts

    reconstructed = predictive.reconstruct_results(
        cohort, [spec], results_dir=results, outer_splits=splits,
    )

    assert reconstructed["common_horizon"] == 12
    assert reconstructed["t_grid"][-1] == 12
    assert reconstructed["summary"]["GG-KM P"]["reconstructed"] == 2
    np.testing.assert_allclose(reconstructed["ibs_values"]["GG-KM P"], [0.1, 0.11])
    np.testing.assert_allclose(reconstructed["auc_values"]["GG-KM P"], [0.7, 0.71])


def test_sigmoid_is_a_candidate_for_breast_auc_selection(saved_breast_holdouts):
    _, results, _, spec = saved_breast_holdouts
    template = pd.read_csv(spec["path"])
    for index, kernel in enumerate(data.KERNELS):
        frame = template.copy()
        frame["kernel"] = kernel
        frame["cure_auc"] = 0.9 if kernel == "sigmoid" else 0.6
        frame.to_csv(results / f"splits_breast_cancer_task{index}_poisson_kernel_{kernel}.csv", index=False)
    # The original RBF filename is one of the files just written (task7 differs).
    original = spec["path"]
    generated_rbf = results / f"splits_breast_cancer_task{data.KERNELS.index('rbf')}_poisson_kernel_rbf.csv"
    if original != generated_rbf:
        original.unlink()

    assert data.choose_best_kernel("poisson", results)["kernel"] == "sigmoid"
    ibs, auc = predictive.collect_kernel_score_data("poisson", results)
    assert len(ibs) == len(auc) == 8


def test_legacy_breast_refits_use_breast_preprocessing_and_saved_parameters(
    saved_breast_holdouts, monkeypatch,
):
    cohort, results, splits, spec = saved_breast_holdouts
    row = data.read_split_file(spec["path"]).iloc[0]
    (results / "plot_data" / "breast_cancer_task7_poisson_kernel_rbf" / "split_000.npz").unlink()
    calls = {}

    class Preprocessor:
        def fit(self, frame):
            calls["training_times"] = frame["Tempo"].to_numpy()
            return frame[["Tempo"]].to_numpy(), frame["Tempo"].to_numpy(), frame["delta"].to_numpy()

        def transform(self, frame):
            calls["test_times"] = frame["Tempo"].to_numpy()
            return frame[["Tempo"]].to_numpy(), frame["Tempo"].to_numpy(), frame["delta"].to_numpy()

    class Model:
        def predict_survival(self, X, grid):
            return np.tile(np.exp(-grid / 20), (len(X), 1))

        def predict_cure_probability(self, X):
            return np.full(len(X), 0.5)

    def build_model(estimator, kernel, params):
        calls["model"] = (estimator, kernel, params)
        return Model()

    def fit_model(model, X, time, event):
        calls["fit_times"] = time

    def import_runner(name):
        assert name == "scripts.run_breast_cancer_experiment"
        return SimpleNamespace(
            BreastCancerSurvivalPreprocessor=Preprocessor,
            MODELS={"poisson": "breast-model"}, build_model=build_model, fit_cure_model=fit_model,
        )

    monkeypatch.setattr(refit.importlib, "import_module", import_runner)
    predictions = refit.obtain_split_predictions(
        row, df=cohort, model_name="poisson", kernel="rbf", outer_splits=splits,
        results_dir=results, refit_missing=True, prediction_grid=[0, 6, 12],
    )

    np.testing.assert_array_equal(calls["training_times"], cohort.iloc[splits[0][0]]["Tempo"])
    np.testing.assert_array_equal(calls["test_times"], cohort.iloc[splits[0][1]]["Tempo"])
    np.testing.assert_array_equal(calls["fit_times"], calls["training_times"])
    assert calls["model"] == ("breast-model", "rbf", {"lambda_reg": 3.0})
    assert predictions["survival_test"].shape == (6, 3)
    assert predictions["cure_probability_test"].shape == (6,)
