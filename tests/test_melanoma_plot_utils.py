"""Saved melanoma results retain their identities when used for plotting."""

import numpy as np
import pandas as pd
import pytest

from scripts import melanoma_plot_utils as plots


@pytest.fixture
def saved_run(tmp_path):
    results = tmp_path / "copied_results"
    prefix = "melanoma_task4_poisson_kernel_rbf"
    archive_path = results / "plot_data" / prefix / "split_003.npz"
    archive_path.parent.mkdir(parents=True)
    train = np.array([4, 1, 8], dtype=np.int64)
    test = np.array([6, 0], dtype=np.int64)
    arrays = {
        "split": np.asarray(3, dtype=np.int64),
        "model_name": np.asarray("poisson"), "kernel": np.asarray("rbf"),
        "train_indices": train, "test_indices": test,
        "evaluation_time_grid": np.array([0.0, 1.0, 2.0]),
        "evaluation_horizon": np.asarray(2.0),
        "survival_test": np.array([[1.0, 0.8, 0.5], [1.0, 0.9, 0.88]]),
        "cure_probability_test": np.array([0.12, 0.87]),
        "train_time": np.array([2.0, 3.0, 4.0]),
        "train_event": np.array([1.0, 0.0, 1.0]),
        "test_time": np.array([3.0, 2.0]), "test_event": np.array([1.0, 0.0]),
    }
    np.savez_compressed(archive_path, **arrays)
    row = {
        "split": 3, "model_name": "poisson", "kernel": "rbf", "status": "ok",
        "converged": True, "horizon": 2.0, "integrated_brier": 0.14,
        "cure_auc": 0.8, "best_params": "{}",
        "train_index_hash": plots.index_hash(train),
        "test_index_hash": plots.index_hash(test),
        "plot_data_file": f"/cluster/project/data/old_results/plot_data/{prefix}/split_003.npz",
    }
    csv_path = results / f"splits_{prefix}.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    return results, archive_path, arrays, plots.read_split_file(csv_path).iloc[0]


def test_relocated_archive_preserves_original_patient_order(saved_run):
    results, _, arrays, row = saved_run
    # Drop source metadata to exercise the remapping of the old absolute path.
    row = row.drop("_result_file")
    loaded = plots.load_split_predictions(
        row, results_dir=results,
        expected_train_indices=arrays["train_indices"],
        expected_test_indices=arrays["test_indices"],
    )
    np.testing.assert_array_equal(loaded["test_indices"], [6, 0])
    np.testing.assert_array_equal(loaded["cure_probability_test"], [0.12, 0.87])
    assert dict(zip(loaded["test_indices"], loaded["cure_probability_test"])) == {
        6: 0.12, 0: 0.87,
    }


def test_selected_results_directory_takes_precedence_over_existing_old_archive(saved_run, tmp_path):
    results, _, arrays, row = saved_run
    old_path = tmp_path / "old_project" / "plot_data" / "old_run" / "split_003.npz"
    old_path.parent.mkdir(parents=True)
    np.savez_compressed(old_path, **{**arrays, "cure_probability_test": np.array([0.4, 0.5])})
    row["plot_data_file"] = str(old_path)
    loaded = plots.load_split_predictions(row, results_dir=results)
    np.testing.assert_array_equal(loaded["cure_probability_test"], [0.12, 0.87])


def test_csv_prefix_finds_archive_when_plot_data_column_is_absent(saved_run):
    results, _, _, row = saved_run
    loaded = plots.load_split_predictions(row.drop("plot_data_file"), results_dir=results)
    np.testing.assert_array_equal(loaded["test_indices"], [6, 0])


@pytest.mark.parametrize("column", ["train_index_hash", "test_index_hash"])
def test_archive_patient_hash_must_match_csv(saved_run, column):
    results, _, _, row = saved_run
    row[column] = "incorrect_hash"
    with pytest.raises(ValueError, match="CSV hash"):
        plots.load_split_predictions(row, results_dir=results)


@pytest.mark.parametrize("column,value", [
    ("split", np.asarray(4)), ("model_name", np.asarray("bn")),
    ("kernel", np.asarray("linear")),
])
def test_archive_identity_must_match_csv(saved_run, column, value):
    results, path, arrays, row = saved_run
    np.savez_compressed(path, **{**arrays, column: value})
    with pytest.raises(ValueError, match="(identifier|metadata).*CSV row"):
        plots.load_split_predictions(row, results_dir=results)


def test_reconstructed_patient_order_must_match_exactly(saved_run):
    results, _, arrays, row = saved_run
    with pytest.raises(ValueError, match="do not match the outer split"):
        plots.load_split_predictions(
            row, results_dir=results, expected_test_indices=arrays["test_indices"][::-1],
        )


@pytest.mark.parametrize("column,value,match", [
    ("cure_probability_test", np.array([[0.12], [0.87]]), "shape"),
    ("survival_test", np.ones((3, 2)), "shape"),
    ("cure_probability_test", np.array([0.2, np.nan]), "finite"),
    ("survival_test", np.array([[1.0, 0.8, 0.5], [1.0, 0.9, 1.2]]), "probabilities"),
    ("cure_probability_test", np.array([-0.1, 0.4]), "probabilities"),
    ("evaluation_time_grid", np.array([0.0, 2.0, 1.0]), "strictly increasing"),
    ("evaluation_horizon", np.asarray(1.5), "inconsistent"),
    ("test_indices", np.array([6, 6]), "unique"),
])
def test_archive_rejects_inconsistent_prediction_arrays(saved_run, column, value, match):
    results, path, arrays, row = saved_run
    np.savez_compressed(path, **{**arrays, column: value})
    with pytest.raises(ValueError, match=match):
        plots.load_split_predictions(row, results_dir=results)


def test_missing_archive_gives_actionable_refit_message(saved_run):
    results, path, _, row = saved_run
    path.unlink()
    with pytest.raises(FileNotFoundError, match="--refit-missing"):
        plots.load_split_predictions(row, results_dir=results)


def test_read_split_file_only_uses_successful_finite_scores(tmp_path):
    path = tmp_path / "splits.csv"
    pd.DataFrame({
        "split": [0, 1, 2, 3], "status": [" OK ", "failed", "ok", "ok"],
        "converged": [True, False, False, True], "horizon": [2, 2, 2, 2],
        "integrated_brier": [0.1, 0.2, np.inf, "invalid"],
        "cure_auc": [0.8, 0.9, 0.7, np.nan], "best_params": ["{}"] * 4,
        "train_index_hash": ["1234567890123456"] * 4,
        "test_index_hash": ["0123456789012345"] * 4,
    }).to_csv(path, index=False)
    frame = plots.read_split_file(path)
    assert frame["valid_ibs"].tolist() == [True, False, False, False]
    assert frame["valid_auc"].tolist() == [True, False, True, False]
    assert frame["test_index_hash"].iloc[0] == "0123456789012345"


@pytest.mark.parametrize("identifiers", [[0, 0], [0, 0.5], [0, -1]])
def test_read_split_file_rejects_ambiguous_split_identifiers(saved_run, identifiers):
    results, _, _, row = saved_run
    path = results / "invalid.csv"
    frame = pd.DataFrame([row, row])
    frame["split"] = identifiers
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="split identifiers"):
        plots.read_split_file(path)


@pytest.mark.parametrize("scores,expected", [
    ({"rbf": (0.8, 0.1), "linear": (0.9, 0.3), "cauchy": (0.7, 0.05)}, "linear"),
    ({"rbf": (0.9, 0.2), "linear": (0.9, 0.1), "cauchy": (0.8, 0.05)}, "linear"),
    ({"rbf": (0.9, 0.1), "linear": (0.9, 0.1), "cauchy": (0.9, 0.1)}, "cauchy"),
])
def test_kernel_selection_uses_auc_then_ibs_then_name(tmp_path, monkeypatch, scores, expected):
    monkeypatch.setattr(plots, "KERNELS", list(scores))
    for kernel, (auc, ibs) in scores.items():
        pd.DataFrame({
            "split": [0, 1], "status": ["ok", "failed"], "converged": [True, False],
            "horizon": [2.0, 2.0], "integrated_brier": [ibs, 0.0],
            "cure_auc": [auc, 1.0], "best_params": ["{}", "{}"],
            "train_index_hash": ["hash_a", "hash_b"],
            "test_index_hash": ["hash_c", "hash_d"],
        }).to_csv(tmp_path / f"splits_melanoma_task1_poisson_kernel_{kernel}.csv", index=False)
    selected = plots.choose_best_kernel("poisson", results_dir=tmp_path)
    assert selected["kernel"] == expected
    assert selected["mean_auc"] == scores[expected][0]
    assert selected["mean_ibs"] == scores[expected][1]
    assert selected["path"] == tmp_path / f"splits_melanoma_task1_poisson_kernel_{expected}.csv"


def test_split_reconstruction_and_hash_validation():
    frame = pd.DataFrame({"delta": [0, 1, 0, 1, 0, 1]})
    splits = plots.construct_outer_splits(frame, n_splits=2, test_size=2, random_state=7)
    rows = pd.DataFrame([
        {"split": split, "train_index_hash": plots.index_hash(train),
         "test_index_hash": plots.index_hash(test)}
        for split, (train, test) in splits.items()
    ])
    plots.verify_split_hashes(rows, splits, "synthetic")
    rows.loc[0, "test_index_hash"] = "wrong"
    with pytest.raises(RuntimeError, match="Outer test split mismatch"):
        plots.verify_split_hashes(rows, splits, "synthetic")


@pytest.mark.parametrize("value", [None, np.nan, pd.NA, "", "none", "NaN", "null"])
def test_parse_best_params_accepts_empty_csv_values(value):
    assert plots.parse_best_params(value) == {}


def test_parse_best_params_preserves_values_and_copies_dictionaries():
    params = {"penalty": 0.3, "normalize": True}
    parsed = plots.parse_best_params(params)
    assert parsed == params and parsed is not params
    assert plots.parse_best_params('{"penalty": 0.3}') == {"penalty": 0.3}


@pytest.mark.parametrize("value,error", [("{broken", ValueError), ("[1, 2]", TypeError)])
def test_parse_best_params_rejects_invalid_payloads(value, error):
    with pytest.raises(error):
        plots.parse_best_params(value)
