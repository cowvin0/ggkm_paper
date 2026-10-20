"""Training-data horizons and time grids for the clinical experiments."""

from __future__ import annotations

import numpy as np


def censoring_reliable_horizon(t, delta, minimum_survival):
    """Return the last time with sufficiently large censoring survival."""
    t = np.asarray(t, dtype=float).reshape(-1)
    delta = np.asarray(delta, dtype=float).reshape(-1)

    if t.size == 0:
        raise ValueError("Cannot construct censoring survival from empty data.")

    if t.shape != delta.shape:
        raise ValueError("t and delta must have the same shape.")

    if np.any(~np.isfinite(t)):
        raise ValueError("Survival times must be finite.")

    if np.any(t < 0.0):
        raise ValueError("Survival times must be non-negative.")

    if not np.all(np.isin(delta, [0.0, 1.0])):
        raise ValueError("delta must contain only 0 and 1.")

    if not (0.0 < minimum_survival < 1.0):
        raise ValueError("minimum_survival must lie in (0, 1).")

    order = np.argsort(t, kind="mergesort")
    sorted_time = t[order]
    sorted_delta = delta[order]
    censoring_event = 1.0 - sorted_delta

    unique_time, first_index, counts = np.unique(
        sorted_time,
        return_index=True,
        return_counts=True,
    )
    survival_after = np.empty(unique_time.size, dtype=float)
    survival = 1.0
    sample_size = sorted_time.size

    for index, (start, count) in enumerate(zip(first_index, counts, strict=True)):
        at_risk = sample_size - int(start)
        censoring_count = int(np.sum(censoring_event[start : start + count]))

        if censoring_count > 0:
            survival *= 1.0 - censoring_count / at_risk

        survival_after[index] = survival

    reliable = survival_after >= minimum_survival

    if not np.any(reliable):
        return 0.0

    reliable_index = np.flatnonzero(reliable)[-1]
    return float(unique_time[reliable_index])


def make_evaluation_grid(
    t,
    delta,
    *,
    points,
    observed_time_quantile,
    minimum_censoring_survival,
    censoring_horizon,
):
    """Build the clinical evaluation grid using the supplied horizon policy."""
    t = np.asarray(t, dtype=float).reshape(-1)
    delta = np.asarray(delta, dtype=float).reshape(-1)

    quantile_horizon = float(np.quantile(t, observed_time_quantile))
    reliable_horizon = censoring_horizon(
        t,
        delta,
        minimum_survival=minimum_censoring_survival,
    )
    horizon = min(quantile_horizon, reliable_horizon)

    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("Training data provide no positive reliable IPCW horizon.")

    t_grid = np.linspace(0.0, horizon, points, dtype=float)
    return t_grid, horizon
