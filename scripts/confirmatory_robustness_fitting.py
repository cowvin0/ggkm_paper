"""Research adapters for auditable GGKM multistart fits; skcure is unchanged."""

from __future__ import annotations

from itertools import product
import warnings

import numpy as np

from skcure.ggkm import GGKM
from scripts.experiment_utils.artifacts import finite_or_none
from scripts.run_latent_robustness_experiment import _KnownLatencyOptimizer


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class InitializedGGKM(GGKM):
    """Set an estimated NB dispersion's initial value without fixing it.

    Public ``phi=value`` fixes dispersion in skcure. The reset hook below
    instead changes the starting state while leaving ``estimate_phi=True``.
    Private hooks are covered by integration tests and frozen source hashes.
    """

    def _reset_parameters_for_fit(self):
        super()._reset_parameters_for_fit()
        if self._latent_model.name == self._NB and self.phi is None:
            self._latent_model.phi = float(self.dispersion_start)
        if self.freeze_latency:
            self._latency_optimizer = _KnownLatencyOptimizer()


class MultiStartGGKM:
    """Factory-compatible wrapper used in both inner folds and final refits."""

    def __init__(self, *, fitting, fixed_latency=None, **parameters):
        self.fitting = fitting
        self.fixed_latency = fixed_latency
        self.parameters = parameters
        self.attempts_ = []

    def fit(self, X, time, event):
        # This wrapper deliberately uses the legacy (X,t,event) fit protocol.
        from evaluate.util_cross_validation import fit_cure_model

        nb = self.parameters.get("latent_distribution") == "negative_binomial"
        free_phi = nb and self.parameters.get("phi") is None
        phi_starts = self.fitting["dispersion_starts"] if free_phi else [None]
        latency_starts = [self.fixed_latency] if self.fixed_latency else self.fitting["latency_starts"]
        candidates = []
        self.attempts_ = []
        for start_id, (latency, phi_start) in enumerate(product(latency_starts, phi_starts)):
            initial = {**self.parameters, **dict(zip(("a", "d", "p"), latency, strict=True))}
            initial["max_em_iter"] = int(self.parameters["max_em_iter"])
            # A retry restarts from the SAME prescribed initial state with a
            # larger iteration budget, rather than selecting using test data.
            for retry in range(self.fitting["retries"] + 1):
                parameters = {**initial, "max_em_iter": initial["max_em_iter"] * self.fitting["retry_multiplier"] ** retry}
                model = InitializedGGKM(**parameters)
                model.dispersion_start = phi_start
                model.freeze_latency = self.fixed_latency is not None
                record = {"start_id": start_id, "retry": retry, "latency_start": list(latency),
                          "dispersion_start": phi_start, "max_em_iter": parameters["max_em_iter"],
                          "status": "failed", "failure_reason": None, "converged": False}
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fit_cure_model(model, X, time, event)
                    record.update(
                        status="ok", converged=bool(model.converged_), n_iter=int(model.n_iter_),
                        final_loglik=finite_or_none(model.final_loglik_),
                        fitted_phi=finite_or_none(getattr(model, "phi_", None)),
                        fitted_latency=[float(model.a_), float(model.d_), float(model.p_)],
                        termination_reason=model.termination_reason_,
                        loglik_history=[finite_or_none(value) for value in model.loglik_history_],
                        optimizer_results=json_safe(model.optimizer_results_),
                        warnings=list(dict.fromkeys(str(item.message) for item in caught)),
                    )
                    if record["final_loglik"] is not None:
                        candidates.append((model.final_loglik_, model, len(self.attempts_)))
                except Exception as error:
                    record["failure_reason"] = f"{type(error).__name__}: {error}"
                self.attempts_.append(record)
                if record["converged"]:
                    break
        if not candidates:
            raise RuntimeError("All prescribed multistart fits failed or had nonfinite objectives")
        _, self.model_, self.selected_attempt_ = max(candidates, key=lambda item: (item[0], bool(item[1].converged_)))
        # Selection is by objective even if a higher-objective fit did not
        # converge. That fact remains visible for the sensitivity analysis.
        return self

    def predict_survival(self, X, times):
        return self.model_.predict_survival(X, times)

    def predict_cure_probability(self, X):
        return self.model_.predict_cure_probability(X)


def profile_dispersion(parameters, fitting, X, time, event, *, fixed_latency=None):
    """Reoptimize incidence and latency at each fixed phi and at Poisson.

    All points use the SAME kernel and penalty; this is an objective profile,
    not an invalid comparison of objectives across tuning configurations.
    """
    records = []
    for phi in fitting["profile_dispersion"]:
        values = dict(parameters)
        values.update(latent_distribution="poisson" if phi == 0 else "negative_binomial",
                      phi=None if phi == 0 else phi)
        fit = MultiStartGGKM(fitting=fitting, fixed_latency=fixed_latency, **values)
        record = {"phi": phi, "status": "failed", "final_loglik": None,
                  "converged": False, "failure_reason": None}
        try:
            fit.fit(X, time, event)
            model = fit.model_
            record.update(status="ok", final_loglik=finite_or_none(model.final_loglik_),
                          converged=bool(model.converged_),
                          fitted_latency=[float(model.a_), float(model.d_), float(model.p_)])
        except Exception as error:
            record["failure_reason"] = f"{type(error).__name__}: {error}"
        record["attempts"] = fit.attempts_
        records.append(record)
    finite = [row["final_loglik"] for row in records if row["final_loglik"] is not None]
    maximum = max(finite, default=None)
    for row in records:
        row["gap_from_profile_max"] = None if maximum is None or row["final_loglik"] is None else maximum - row["final_loglik"]
    return records
