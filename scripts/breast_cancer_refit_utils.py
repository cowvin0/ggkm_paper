"""Load breast-cancer predictions, with an explicit legacy refit fallback."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np

from scripts.breast_cancer_plot_utils import (
    PROJECT_ROOT, index_hash, load_split_predictions, parse_best_params,
)


def _value(row, name):
    return row.get(name) if hasattr(row, "get") else getattr(row, name, None)


def _refit_predictions(row, *, df, model_name, kernel, train_idx, test_idx, prediction_grid):
    """Fit the runner's breast-cancer preprocessing/model with saved parameters."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    runner = importlib.import_module("scripts.run_breast_cancer_experiment")
    params = parse_best_params(_value(row, "best_params"))
    preprocessor = runner.BreastCancerSurvivalPreprocessor()
    X_train, t_train, d_train = preprocessor.fit(df.iloc[train_idx].reset_index(drop=True))
    X_test, t_test, d_test = preprocessor.transform(df.iloc[test_idx].reset_index(drop=True))
    model = runner.build_model(runner.MODELS[model_name], kernel, params)
    print(f"Refitting breast-cancer {model_name}, kernel={kernel}, split={int(_value(row, 'split'))}")
    runner.fit_cure_model(model, X_train, t_train, d_train)
    if prediction_grid is None:
        horizon = float(_value(row, "horizon"))
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("Saved split has no positive finite prediction horizon.")
        prediction_grid = np.linspace(0.0, horizon, 250)
    grid = np.asarray(prediction_grid, dtype=float).reshape(-1)
    if grid.size < 2 or not np.isfinite(grid).all() or grid[0] < 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("Prediction times must be a finite, increasing nonnegative grid.")
    survival = np.asarray(model.predict_survival(X_test, grid), dtype=float)
    cure = np.asarray(model.predict_cure_probability(X_test), dtype=float).reshape(-1)
    if survival.shape != (len(test_idx), len(grid)) or cure.shape != (len(test_idx),):
        raise ValueError("Refitted predictions have unexpected patient or time dimensions.")
    if not np.isfinite(survival).all() or not np.isfinite(cure).all():
        raise ValueError("Refitted predictions contain nonfinite values.")
    if np.any((survival < 0) | (survival > 1)) or np.any((cure < 0) | (cure > 1)):
        raise ValueError("Refitted survival and cure probabilities must lie in [0, 1].")
    return {
        "split": np.asarray(int(_value(row, "split"))), "model_name": np.asarray(model_name),
        "kernel": np.asarray(kernel if kernel is not None else "parametric"),
        "train_indices": train_idx, "test_indices": test_idx,
        "train_time": np.asarray(t_train), "train_event": np.asarray(d_train),
        "test_time": np.asarray(t_test), "test_event": np.asarray(d_test),
        "evaluation_time_grid": grid, "evaluation_horizon": np.asarray(grid[-1]),
        "survival_test": survival, "cure_probability_test": cure,
    }


def obtain_split_predictions(
    row, *, df, model_name, kernel, outer_splits, results_dir,
    project_root=PROJECT_ROOT, refit_missing=False, prediction_grid=None,
):
    """Validate and load original held-out predictions; never tune a model."""
    if str(_value(row, "status")).strip().lower() != "ok":
        raise ValueError("Predictions may only be loaded for successful saved splits.")
    split = int(_value(row, "split"))
    if split not in outer_splits:
        raise ValueError(f"Saved split {split} is outside the reconstructed holdout design.")
    train_idx, test_idx = (np.asarray(indices, dtype=np.int64) for indices in outer_splits[split])
    for column, indices in (("train_index_hash", train_idx), ("test_index_hash", test_idx)):
        if str(_value(row, column)) != index_hash(indices):
            raise ValueError(f"Saved {column} does not match reconstructed split {split}.")
    try:
        predictions = load_split_predictions(
            row, results_dir=Path(results_dir), project_root=Path(project_root),
            expected_train_indices=train_idx, expected_test_indices=test_idx,
        )
        expected_kernel = kernel if kernel not in (None, "parametric") else "parametric"
        if (
            str(predictions["model_name"].item()) != model_name
            or str(predictions["kernel"].item()) != expected_kernel
        ):
            raise ValueError(
                f"Saved predictions for split {split} do not match the requested "
                f"model {model_name}, kernel {expected_kernel}."
            )
        return predictions
    except FileNotFoundError as error:
        if not refit_missing:
            raise FileNotFoundError(
                f"{error}\nFor legacy CSV-only results, pass --refit-missing "
                "to reconstruct predictions with the current breast-cancer model code."
            ) from error
    return _refit_predictions(
        row, df=df, model_name=model_name, kernel=None if kernel in (None, "parametric") else kernel,
        train_idx=train_idx, test_idx=test_idx, prediction_grid=prediction_grid,
    )
