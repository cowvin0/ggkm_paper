import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from skcure.utils.estimator import fit_cure_model
from evaluate.metrics import integrated_brier_score


def _make_evaluation_grid(
    time_values,
    event_values,
    points,
    observed_time_quantile=0.90,
    minimum_censoring_survival=0.05,
):
    quantile_horizon = float(
        np.quantile(
            time_values,
            observed_time_quantile,
        )
    )

    censoring_horizon = _reliable_censoring_horizon(
        time_values,
        event_values,
        minimum_survival=minimum_censoring_survival,
    )

    horizon = min(
        quantile_horizon,
        censoring_horizon,
    )

    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError(
            "training data provide no positive reliable evaluation horizon"
        )

    return np.linspace(
        0.0,
        horizon,
        points,
    )


def _build_estimator(
    estimator,
    kernel,
    params,
):
    if kernel is None:
        return estimator(**params)

    return estimator(
        kernel=kernel,
        **params,
    )


def _select_grid_parameters(
    X,
    t,
    delta,
    estimator,
    kernel,
    parameter_grid,
    *,
    n_inner_splits,
    t_grid_points,
    random_state,
):
    splitter = StratifiedKFold(
        n_splits=n_inner_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = list(
        splitter.split(
            X,
            delta,
        )
    )

    best_params = None
    best_score = np.inf

    for params in parameter_grid:
        fold_scores = []

        for train_index, validation_index in splits:
            X_train_raw = X[train_index]
            X_validation_raw = X[validation_index]

            t_train = t[train_index]
            d_train = delta[train_index]

            t_validation = t[validation_index]
            d_validation = delta[validation_index]

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train_raw)
            X_validation = scaler.transform(X_validation_raw)

            try:
                t_grid = _make_evaluation_grid(
                    t_train,
                    d_train,
                    points=t_grid_points,
                )

                model = _build_estimator(
                    estimator,
                    kernel,
                    params,
                )

                fit_cure_model(
                    model,
                    X_train,
                    t_train,
                    d_train,
                )

                survival = model.predict_survival(
                    X_validation,
                    t_grid,
                )

                ibs = integrated_brier_score(
                    survival,
                    t_validation,
                    d_validation,
                    t_train,
                    d_train,
                    t_grid,
                )

                if not np.isfinite(ibs):
                    raise FloatingPointError("non-finite inner IBS")

                fold_scores.append(float(ibs))

            except Exception:
                fold_scores = []
                break

        if len(fold_scores) != n_inner_splits:
            continue

        mean_score = float(np.mean(fold_scores))

        if mean_score < best_score:
            best_score = mean_score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("all grid-search candidates failed")

    return best_params, best_score


def _reliable_censoring_horizon(
    time_values,
    event_values,
    minimum_survival=0.05,
):
    time_values = np.asarray(time_values, dtype=float)
    event_values = np.asarray(event_values, dtype=int)

    order = np.argsort(time_values, kind="mergesort")

    sorted_time = time_values[order]
    sorted_event = event_values[order]

    unique_time, first_index, counts = np.unique(
        sorted_time,
        return_index=True,
        return_counts=True,
    )

    survival = 1.0
    reliable_horizon = 0.0
    n = len(sorted_time)

    for current_time, start, count in zip(
        unique_time,
        first_index,
        counts,
    ):
        at_risk = n - start

        censoring_count = np.sum(1 - sorted_event[start : start + count])

        if censoring_count > 0:
            survival *= 1.0 - censoring_count / at_risk

        if survival >= minimum_survival:
            reliable_horizon = current_time

    return float(reliable_horizon)
