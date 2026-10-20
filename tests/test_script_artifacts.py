"""Clinical entry points preserve resumable checkpoints and output formats."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=["breast_cancer", "melanoma"])
def clinical_script(request, monkeypatch, tmp_path):
    script = importlib.import_module(
        f"scripts.run_{request.param}_experiment"
        # f"ggkm_paper.scripts.run_{request.param}_experiment"
    )
    project_root = tmp_path / "project"
    monkeypatch.setattr(script, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(script, "OUTPUT_DIR", project_root / "saved")
    return script, request.param


@pytest.mark.parametrize("kernel", [None, "rbf"])
def test_clinical_checkpoint_round_trip_and_resume_validation(clinical_script, kernel):
    script, dataset = clinical_script
    model_name = "mixture_cure" if kernel is None else "poisson"
    prefix = f"{dataset}_task4_{model_name}"
    if kernel is not None:
        prefix += f"_kernel_{kernel}"

    checkpoint = script.checkpoint_path(4, model_name, kernel, 3)
    archive = script.plot_data_path(4, model_name, kernel, 3)
    assert checkpoint == script.OUTPUT_DIR / "checkpoints" / prefix / "split_003.json"
    assert archive == script.OUTPUT_DIR / "plot_data" / prefix / "split_003.npz"

    arrays = {
        "time_grid": np.array([0.0, 1.0, 2.0]),
        "predicted_survival_test": np.array([[1.0, 0.8, 0.6], [1.0, 0.7, 0.4]]),
    }
    script.atomic_write_npz(archive, **arrays)
    with np.load(archive, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for name, array in arrays.items():
            np.testing.assert_array_equal(saved[name], array)

    relative_archive = script._portable_path(archive)
    assert relative_archive == str(archive.relative_to(script.PROJECT_ROOT))
    assert script._resolve_saved_path(relative_archive) == archive
    assert script._resolve_saved_path(str(archive)) == archive

    expected = {
        "split_index": 3,
        "model_name": model_name,
        "kernel": kernel,
        "train_hash": script.index_hash([0, 2, 4]),
        "test_hash": script.index_hash([1, 3]),
    }
    row = {
        "split": 3,
        "model_name": model_name,
        "kernel": kernel if kernel is not None else "parametric",
        "train_index_hash": expected["train_hash"],
        "test_index_hash": expected["test_hash"],
        "status": "ok",
        "plot_data_file": relative_archive,
    }
    script.atomic_write_json(checkpoint, row)
    assert json.loads(checkpoint.read_text()) == row
    assert script.load_valid_checkpoint(checkpoint, **expected) == row

    for key, value in (
        ("split_index", 4),
        ("model_name", "other_model"),
        ("kernel", "linear"),
        ("train_hash", "different_train_indices"),
        ("test_hash", "different_test_indices"),
    ):
        assert (
            script.load_valid_checkpoint(checkpoint, **{**expected, key: value}) is None
        )

    archive.unlink()
    assert script.load_valid_checkpoint(checkpoint, **expected) is None

    row.update(status="failed", plot_data_file=None)
    script.atomic_write_json(checkpoint, row)
    assert script.load_valid_checkpoint(checkpoint, **expected) == row


def test_clinical_resume_ignores_missing_and_corrupted_json(clinical_script):
    script, _ = clinical_script
    checkpoint = script.checkpoint_path(4, "poisson", "rbf", 3)
    expected = dict(
        split_index=3,
        model_name="poisson",
        kernel="rbf",
        train_hash="train",
        test_hash="test",
    )
    assert script.load_valid_checkpoint(checkpoint, **expected) is None

    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{incomplete JSON", encoding="utf-8")
    assert script.load_valid_checkpoint(checkpoint, **expected) is None


def test_clinical_csv_names_schema_and_successful_split_summary(clinical_script):
    script, dataset = clinical_script
    results = pd.DataFrame(
        {
            "split": [0, 1, 2],
            "model_name": ["poisson"] * 3,
            "kernel": ["rbf"] * 3,
            "status": ["ok", "ok", "failed"],
            "converged": [True, False, False],
            "integrated_brier": [0.1, 0.3, 99.0],
            "uno_c_index": [0.6, 0.8, 99.0],
            "cure_auc": [0.7, 0.9, 99.0],
            "mean_predicted_cure": [0.2, 0.4, 99.0],
            "runtime_seconds": [2.0, 4.0, 999.0],
        }
    )

    split_file, summary_file, summary = script.save_results(
        results, 4, "poisson", "rbf"
    )
    prefix = f"{dataset}_task4_poisson_kernel_rbf"
    assert split_file == script.OUTPUT_DIR / f"splits_{prefix}.csv"
    assert summary_file == script.OUTPUT_DIR / f"summary_{prefix}.csv"
    pd.testing.assert_frame_equal(pd.read_csv(split_file), results)

    expected = {
        "model_name": "poisson",
        "kernel": "rbf",
        "splits": 3,
        "successful": 2,
        "failed": 1,
        "converged": 1,
        "mean_integrated_brier": 0.2,
        "split_sd_integrated_brier": np.sqrt(0.02),
        "mean_uno_c_index": 0.7,
        "split_sd_uno_c_index": np.sqrt(0.02),
        "mean_cure_auc": 0.8,
        "split_sd_cure_auc": np.sqrt(0.02),
        "mean_predicted_cure": 0.3,
        "split_sd_mean_predicted_cure": np.sqrt(0.02),
        "mean_runtime_seconds": 3.0,
        "maximum_runtime_seconds": 4.0,
    }
    assert summary == pytest.approx(expected)
    pd.testing.assert_frame_equal(pd.read_csv(summary_file), pd.DataFrame([summary]))
