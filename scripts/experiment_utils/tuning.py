"""Shared inner cross-validation and clinical grid-search routines.

The runners supply their settings and callbacks so each experiment retains its
own preprocessing, model construction, and scoring behavior.
"""

from __future__ import annotations

import numpy as np


def prepare_inner_splits(
    df_outer_train,
    *,
    inner_seed,
    n_inner_splits,
    preprocessor_factory,
    make_evaluation_grid,
    inner_grid_points,
    splitter_factory,
):
    """Fit preprocessing within each training fold and prepare its time grid."""
    inner_cv = splitter_factory(
        n_splits=n_inner_splits,
        shuffle=True,
        random_state=inner_seed,
    )
    prepared = []

    for tr_idx, val_idx in inner_cv.split(df_outer_train, df_outer_train["delta"]):
        df_tr = df_outer_train.iloc[tr_idx].reset_index(drop=True)
        df_val = df_outer_train.iloc[val_idx].reset_index(drop=True)
        preprocessor = preprocessor_factory()
        X_tr, t_tr, d_tr = preprocessor.fit(df_tr)
        X_val, t_val, d_val = preprocessor.transform(df_val)
        t_grid_inner, _ = make_evaluation_grid(
            t_tr,
            d_tr,
            points=inner_grid_points,
        )
        prepared.append(
            (X_tr, t_tr, d_tr, X_val, t_val, d_val, t_grid_inner)
        )

    return prepared


def score_params_on_inner_splits(
    inner_splits,
    *,
    estimator,
    kernel,
    params,
    n_inner_splits,
    build_model,
    fit_model,
    integrated_brier_score,
):
    """Return the mean validation integrated Brier score across all folds."""
    fold_scores = []

    for X_tr, t_tr, d_tr, X_val, t_val, d_val, t_grid_inner in inner_splits:
        model = build_model(estimator, kernel, params)
        fit_model(model, X_tr, t_tr, d_tr)
        S_val = model.predict_survival(X_val, t_grid_inner)
        ibs = integrated_brier_score(
            S_val,
            t_val,
            d_val,
            t_tr,
            d_tr,
            t_grid_inner,
        )

        if not np.isfinite(ibs):
            raise FloatingPointError("Non-finite inner integrated Brier score.")

        fold_scores.append(float(ibs))

    if len(fold_scores) != n_inner_splits:
        raise RuntimeError("Inner tuning did not evaluate every fold.")

    return float(np.mean(fold_scores))


def _score_grid(
    candidates,
    inner_splits,
    *,
    estimator,
    kernel,
    score_params_on_inner_splits,
    failure_message,
    report_failures,
):
    scores = []
    failure_reasons = []

    for candidate in candidates:
        try:
            score = score_params_on_inner_splits(
                inner_splits,
                estimator=estimator,
                kernel=kernel,
                params=candidate,
            )
            failure_reason = None

        except Exception as error:
            score = np.inf
            failure_reason = (
                f"{type(error).__name__}: {error}" if report_failures else None
            )

        scores.append(float(score))
        failure_reasons.append(failure_reason)

    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)

    if not np.any(finite):
        first_failure = next(
            (reason for reason in failure_reasons if reason is not None),
            None,
        )

        if first_failure is not None:
            raise RuntimeError(f"{failure_message} First failure: {first_failure}")

        raise RuntimeError(failure_message)

    best_index = int(np.argmin(scores))
    return (
        dict(candidates[best_index]),
        float(scores[best_index]),
        len(candidates),
        int(np.sum(finite)),
    )


def tune_kernel_grid(
    df_outer_train,
    *,
    estimator,
    model_name,
    kernel,
    inner_seed,
    prepare_inner_splits,
    kernel_parameter_grid,
    score_params_on_inner_splits,
):
    """Evaluate the runner's kernel candidates on its prepared inner folds."""
    inner_splits = prepare_inner_splits(df_outer_train, inner_seed=inner_seed)
    candidates = kernel_parameter_grid(model_name, kernel)
    return _score_grid(
        candidates,
        inner_splits,
        estimator=estimator,
        kernel=kernel,
        score_params_on_inner_splits=score_params_on_inner_splits,
        failure_message="All kernel-grid candidates failed during inner tuning.",
        report_failures=True,
    )


def tune_spline_grid(
    df_outer_train,
    *,
    estimator,
    inner_seed,
    prepare_inner_splits,
    score_params_on_inner_splits,
    spline_parameter_grid,
    report_failures=True,
):
    """Evaluate spline candidates, retaining each runner's failure reporting.

    Breast-cancer tuning copies candidate dictionaries and reports the first
    failure. Melanoma tuning uses the original candidates and its shorter error.
    """
    inner_splits = prepare_inner_splits(df_outer_train, inner_seed=inner_seed)

    if report_failures:
        candidates = [dict(candidate) for candidate in spline_parameter_grid]
        failure_message = "All spline-grid candidates failed during inner tuning."
    else:
        candidates = spline_parameter_grid
        failure_message = "All spline parameter candidates failed during inner tuning."

    return _score_grid(
        candidates,
        inner_splits,
        estimator=estimator,
        kernel=None,
        score_params_on_inner_splits=score_params_on_inner_splits,
        failure_message=failure_message,
        report_failures=report_failures,
    )
