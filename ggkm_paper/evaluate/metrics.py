import numpy as np


def harrell_c_index(risk_score, time, event) -> float:
    risk_score = np.asarray(risk_score, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = len(time)

    permissible = 0.0
    concordant = 0.0
    tied = 0.0

    for i in range(n - 1):
        j_idx = np.arange(i + 1, n)
        t_i, t_j = time[i], time[j_idx]
        e_i, e_j = event[i], event[j_idx]

        smaller_i = (t_i < t_j) & (e_i == 1)
        smaller_j = (t_j < t_i) & (e_j == 1)
        tied_time_both_events = (t_i == t_j) & (e_i == 1) & (e_j == 1)

        comparable = smaller_i | smaller_j | tied_time_both_events
        if not np.any(comparable):
            continue

        diff = np.where(
            smaller_i,
            risk_score[i] - risk_score[j_idx],
            np.where(smaller_j, risk_score[j_idx] - risk_score[i], 0.0),
        )
        is_tied_pair = tied_time_both_events | (diff == 0)

        permissible += comparable.sum()
        concordant += np.sum(comparable & ~is_tied_pair & (diff > 0))
        tied += np.sum(comparable & is_tied_pair)

    if permissible == 0:
        return float("nan")
    return (concordant + 0.5 * tied) / permissible


def uno_c_index_rmst(
    S_pred,
    t_eval,
    delta_eval,
    t_train,
    delta_train,
    t_grid,
    tau=None,
    *,
    minimum_probability=1.0e-8,
    tied_tol=1.0e-8,
):
    S_pred = np.asarray(S_pred, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)
    delta_eval = np.asarray(delta_eval, dtype=float)
    t_train = np.asarray(t_train, dtype=float)
    delta_train = np.asarray(delta_train, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    if t_eval.ndim != 1 or delta_eval.ndim != 1:
        raise ValueError("t_eval and delta_eval must be one-dimensional")
    if t_train.ndim != 1 or delta_train.ndim != 1:
        raise ValueError("t_train and delta_train must be one-dimensional")
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be one-dimensional")
    if len(t_eval) != len(delta_eval):
        raise ValueError("t_eval and delta_eval must have the same length")
    if len(t_train) != len(delta_train):
        raise ValueError("t_train and delta_train must have the same length")
    if S_pred.shape != (len(t_eval), len(t_grid)):
        raise ValueError("S_pred must have shape " "(len(t_eval), len(t_grid))")
    if not np.all(np.isfinite(S_pred)):
        return np.nan
    if not np.all(np.isfinite(t_eval)):
        raise ValueError("t_eval must contain only finite values")
    if not np.all(np.isfinite(t_train)):
        raise ValueError("t_train must contain only finite values")
    if not np.all(np.isfinite(t_grid)):
        raise ValueError("t_grid must contain only finite values")
    if not np.all(np.isin(delta_eval, [0.0, 1.0])):
        raise ValueError("delta_eval must contain only 0 and 1")
    if not np.all(np.isin(delta_train, [0.0, 1.0])):
        raise ValueError("delta_train must contain only 0 and 1")
    if np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing")
    if not 0.0 < minimum_probability < 1.0:
        raise ValueError("minimum_probability must lie strictly between 0 and 1")
    if tied_tol < 0.0:
        raise ValueError("tied_tol must be non-negative")

    numerical_tolerance = 2.0e-12
    if np.any(S_pred < -numerical_tolerance) or np.any(
        S_pred > 1.0 + numerical_tolerance
    ):
        raise ValueError("S_pred must contain survival probabilities in [0, 1]")

    S_pred = np.clip(S_pred, 0.0, 1.0)
    if tau is None:
        tau = float(t_grid[-1])
    else:
        tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
    if tau <= t_grid[0]:
        raise ValueError("tau must be greater than the first grid time")
    if tau > t_grid[-1]:
        raise ValueError("tau cannot exceed the last value of t_grid")

    before_tau = t_grid < tau
    rmst_grid = t_grid[before_tau]
    rmst_pred = S_pred[:, before_tau]
    exact_tau = np.flatnonzero(np.isclose(t_grid, tau, rtol=0.0, atol=1.0e-12))

    if exact_tau.size > 0:
        tau_index = exact_tau[0]

        rmst_grid = np.append(
            rmst_grid,
            tau,
        )

        rmst_pred = np.column_stack(
            (
                rmst_pred,
                S_pred[:, tau_index],
            )
        )

    else:
        right = np.searchsorted(
            t_grid,
            tau,
            side="right",
        )

        left = right - 1

        t_left = t_grid[left]
        t_right = t_grid[right]

        fraction = (tau - t_left) / (t_right - t_left)

        S_tau = S_pred[:, left] + fraction * (S_pred[:, right] - S_pred[:, left])

        rmst_grid = np.append(
            rmst_grid,
            tau,
        )

        rmst_pred = np.column_stack(
            (
                rmst_pred,
                S_tau,
            )
        )

    rmst = np.trapezoid(
        rmst_pred,
        rmst_grid,
        axis=1,
    )

    risk_score = -rmst
    G = censoring_survival_function(
        t_train,
        delta_train,
    )

    G_event = G(
        t_eval,
        left_limit=True,
    )
    G_event = np.maximum(
        G_event,
        minimum_probability,
    )
    weights = 1.0 / (G_event**2)
    numerator = 0.0
    denominator = 0.0

    n = len(t_eval)

    for i in range(n):
        if delta_eval[i] != 1.0:
            continue
        if t_eval[i] >= tau:
            continue
        comparable = t_eval > t_eval[i]

        if not np.any(comparable):
            continue

        risk_difference = risk_score[i] - risk_score[comparable]
        concordant = risk_difference > tied_tol
        tied = np.abs(risk_difference) <= tied_tol
        pair_score = concordant.astype(float) + 0.5 * tied.astype(float)
        weight = weights[i]
        numerator += weight * np.sum(pair_score)
        denominator += weight * np.sum(comparable)

    if denominator <= 0.0:
        return np.nan

    return float(numerator / denominator)


def kaplan_meier(t, delta):
    t = np.asarray(t, dtype=float)
    delta = np.asarray(delta, dtype=float)

    if t.ndim != 1 or delta.ndim != 1:
        raise ValueError("t and delta must be one-dimensional")
    if len(t) != len(delta):
        raise ValueError("t and delta must have the same length")
    if len(t) == 0:
        raise ValueError("t and delta cannot be empty")
    if not np.all(np.isfinite(t)):
        raise ValueError("t must contain only finite values")
    if np.any(t < 0.0):
        raise ValueError("t must be non-negative")
    if not np.all(np.isin(delta, [0.0, 1.0])):
        raise ValueError("delta must contain only 0 and 1")

    order = np.argsort(t)
    t_ord = t[order]
    d_ord = delta[order]

    unique_t = np.unique(t_ord)

    surv = 1.0

    km_times = []
    km_surv = []

    if unique_t[0] > 0.0:
        km_times.append(0.0)
        km_surv.append(1.0)

    for ti in unique_t:
        at_time = t_ord == ti

        events = np.sum(d_ord[at_time])
        at_risk = np.sum(t_ord >= ti)

        if events > 0:
            surv *= 1.0 - events / at_risk

        km_times.append(ti)
        km_surv.append(surv)

    return (
        np.asarray(km_times, dtype=float),
        np.asarray(km_surv, dtype=float),
    )


def censoring_survival_function(t_train, delta_train):
    t_train = np.asarray(t_train, dtype=float)
    delta_train = np.asarray(delta_train, dtype=float)

    if t_train.ndim != 1 or delta_train.ndim != 1:
        raise ValueError("t_train and delta_train must be one-dimensional")
    if t_train.shape[0] != delta_train.shape[0]:
        raise ValueError("t_train and delta_train must have the same length")
    if len(t_train) == 0:
        raise ValueError("training data cannot be empty")
    if not np.all(np.isfinite(t_train)):
        raise ValueError("t_train must contain only finite values")
    if np.any(t_train < 0.0):
        raise ValueError("t_train must be non-negative")
    if not np.all(np.isin(delta_train, [0.0, 1.0])):
        raise ValueError("delta_train must contain only 0 and 1")

    censoring_event = 1.0 - delta_train

    km_times, km_surv = kaplan_meier(
        t_train,
        censoring_event,
    )

    km_times = np.asarray(km_times, dtype=float)
    km_surv = np.asarray(km_surv, dtype=float)

    if km_times.ndim != 1 or km_surv.ndim != 1:
        raise ValueError("kaplan_meier must return one-dimensional arrays")

    if km_times.shape[0] != km_surv.shape[0]:
        raise ValueError("kaplan_meier returned incompatible time and survival arrays")

    if len(km_times) == 0:

        def G(t_query, left_limit=False):
            t_query = np.asarray(t_query, dtype=float)
            result = np.ones_like(t_query, dtype=float)

            if result.ndim == 0:
                return float(result)

            return result

        return G

    def G(t_query, left_limit=False):
        t_query = np.asarray(t_query, dtype=float)

        if np.any(~np.isfinite(t_query)):
            raise ValueError("t_query must contain only finite values")

        if left_limit:
            idx = (
                np.searchsorted(
                    km_times,
                    t_query,
                    side="left",
                )
                - 1
            )
        else:
            idx = (
                np.searchsorted(
                    km_times,
                    t_query,
                    side="right",
                )
                - 1
            )

        result = np.ones_like(t_query, dtype=float)

        valid = idx >= 0
        result[valid] = km_surv[idx[valid]]

        if result.ndim == 0:
            return float(result)

        return result

    return G


def integrated_brier_score(
    S_pred,
    t_eval,
    delta_eval,
    t_train,
    delta_train,
    t_grid,
    *,
    minimum_probability=1.0e-8,
):
    S_pred = np.asarray(S_pred, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)
    delta_eval = np.asarray(delta_eval, dtype=float)
    t_train = np.asarray(t_train, dtype=float)
    delta_train = np.asarray(delta_train, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    if t_eval.ndim != 1:
        raise ValueError("t_eval must be one-dimensional")
    if delta_eval.ndim != 1:
        raise ValueError("delta_eval must be one-dimensional")
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be one-dimensional")
    if t_train.ndim != 1:
        raise ValueError("t_train must be one-dimensional")
    if delta_train.ndim != 1:
        raise ValueError("delta_train must be one-dimensional")
    if len(t_eval) != len(delta_eval):
        raise ValueError("t_eval and delta_eval must have the same length")
    if len(t_train) != len(delta_train):
        raise ValueError("t_train and delta_train must have the same length")
    if len(t_eval) == 0:
        raise ValueError("evaluation data cannot be empty")
    if len(t_train) == 0:
        raise ValueError("training data cannot be empty")
    if len(t_grid) < 2:
        raise ValueError("t_grid must contain at least two time points")
    if S_pred.shape != (len(t_eval), len(t_grid)):
        raise ValueError("S_pred must have shape " "(len(t_eval), len(t_grid))")
    if not np.all(np.isfinite(S_pred)):
        return np.inf
    if not np.all(np.isfinite(t_eval)):
        raise ValueError("t_eval must contain only finite values")
    if not np.all(np.isfinite(t_train)):
        raise ValueError("t_train must contain only finite values")
    if not np.all(np.isfinite(t_grid)):
        raise ValueError("t_grid must contain only finite values")
    if np.any(t_eval < 0.0):
        raise ValueError("t_eval must be non-negative")
    if np.any(t_train < 0.0):
        raise ValueError("t_train must be non-negative")
    if not np.all(np.isin(delta_eval, [0.0, 1.0])):
        raise ValueError("delta_eval must contain only 0 and 1")
    if not np.all(np.isin(delta_train, [0.0, 1.0])):
        raise ValueError("delta_train must contain only 0 and 1")
    if np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing")
    if not 0.0 < minimum_probability < 1.0:
        raise ValueError("minimum_probability must lie strictly between 0 and 1")

    numerical_tolerance = 2.0e-12
    if np.any(S_pred < -numerical_tolerance) or np.any(
        S_pred > 1.0 + numerical_tolerance
    ):
        raise ValueError("S_pred must contain survival probabilities in [0, 1]")

    S_pred = np.clip(S_pred, 0.0, 1.0)

    G = censoring_survival_function(
        t_train,
        delta_train,
    )
    G_y_minus = G(
        t_eval,
        left_limit=True,
    )
    G_y_minus = np.maximum(
        G_y_minus,
        minimum_probability,
    )
    G_grid = G(
        t_grid,
        left_limit=False,
    )
    G_grid = np.maximum(
        G_grid,
        minimum_probability,
    )
    bs_grid = np.empty(len(t_grid), dtype=float)

    for j, t in enumerate(t_grid):

        event_before = (t_eval <= t) & (delta_eval == 1.0)
        alive_after = t_eval > t
        event_term = event_before * S_pred[:, j] ** 2 / G_y_minus
        risk_term = alive_after * (1.0 - S_pred[:, j]) ** 2 / G_grid[j]
        bs_grid[j] = np.mean(event_term + risk_term)

    duration = float(t_grid[-1] - t_grid[0])
    integrated_score = (
        np.trapezoid(
            bs_grid,
            t_grid,
        )
        / duration
    )

    return float(integrated_score)
