"""Research adapters for auditable GGKM multistart fits; skcure is unchanged."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
import warnings

import numpy as np

from skcure.ggkm import GGKM
from skcure._em import _EstimatorEMState, _GGKMSteps, run_em
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


def _record_solution(model):
    latency = [float(model.a_), float(model.d_), float(model.p_)]
    phi = finite_or_none(getattr(model, "phi_", None))
    return dict(
        status="ok", converged=bool(model.converged_), n_iter=int(model.n_iter_),
        final_loglik=finite_or_none(model.final_loglik_), fitted_phi=phi,
        fitted_latency=latency, termination_reason=model.termination_reason_,
        loglik_history=[finite_or_none(value) for value in model.loglik_history_],
        optimizer_results=json_safe(model.optimizer_results_),
        latency_at_lower_bound=[name for name, value in zip(("a", "d", "p"), latency, strict=True)
                                if value <= 1e-5 * (1 + 1e-6)],
        dispersion_at_lower_bound=phi is not None and phi <= 1e-8 * (1 + 1e-6),
    )


def _resume_free_nb(seed, phi, time, event, max_em_iter):
    """Continue ECME from a full fitted state, without resetting its predictor.

    ``GGKM.fit(warm_start=True)`` still initializes beta0/alpha anew. Use the
    same library ECME blocks explicitly with the fitted coefficients instead.
    The Poisson endpoint is a seed at the library's positive NB lower bound;
    its initial objective is recomputed under NB, never relabeled as exact NB.
    """
    model = deepcopy(seed)
    model.latent_distribution, model.phi = "negative_binomial", None
    model._latent_model = model._make_latent_model(model._NB, None, model.K_bin)
    model._latent_model.phi = max(float(phi), 1e-8)
    model._latent_configuration = (model._NB, None, None)
    model.max_em_iter = int(max_em_iter)
    model.n_iter_, model.converged_, model._is_fitted = 0, False, False
    model.loglik_history_, model.optimizer_results_ = [], []
    model.termination_reason_ = "running"
    parameters = (float(seed.beta0_), seed.alpha_.copy())
    steps = _GGKMSteps(model, np.asarray(time), np.asarray(event, dtype=float))
    initial = float(steps.loglikelihood(parameters))
    model._validate_initial_loglik(initial)
    model.continuation_initial_loglik_ = initial
    best = initial

    def guarded_loglikelihood(values):
        nonlocal best
        value = float(steps.loglikelihood(values))
        # Allow floating-point noise only. A material decrease invalidates
        # this attempt; it cannot inherit the fixed-profile convergence flag.
        if value < best - 1e-8:
            raise RuntimeError("Free-NB continuation decreased the penalized objective")
        best = max(best, value)
        return value

    (model.beta0_, model.alpha_), model.final_loglik_ = run_em(
        parameters, e_step=steps.expectation,
        m_steps=(steps.maximize_predictor, steps.maximize_latency, steps.maximize_dispersion),
        loglikelihood=guarded_loglikelihood, state=_EstimatorEMState(model),
        tol=model.tol, rtol=model.em_rtol, max_iter=model.max_em_iter,
        validate_initial_loglik=model._validate_initial_loglik,
        nonfinite_message="Non-finite free-NB continuation objective.",
    )
    model._latent_model.finalize_fit()
    model._is_fitted = True
    times = np.unique(np.r_[0., np.quantile(time, [0., .25, .5, .75, 1.])])
    survival = model.predict_survival(model.X_train_, times)
    cure = model.predict_cure_probability(model.X_train_)
    if (not np.isfinite(survival).all() or not np.isfinite(cure).all()
            or np.any((survival < 0) | (survival > 1)) or np.any((cure < 0) | (cure > 1))
            or np.any(np.diff(survival, axis=1) > 1e-10)):
        raise FloatingPointError("Invalid training predictions after free-NB continuation")
    return model


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
        self.profile_ = []
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
                    record.update(_record_solution(model))
                    record["warnings"] = list(dict.fromkeys(str(item.message) for item in caught))
                    if record["final_loglik"] is not None:
                        candidates.append((model.final_loglik_, model, len(self.attempts_)))
                except Exception as error:
                    record["failure_reason"] = f"{type(error).__name__}: {error}"
                self.attempts_.append(record)
                if record["converged"]:
                    break
        if free_phi and self.fitting["profile_restart"]:
            self.profile_, solutions = profile_dispersion(
                self.parameters, self.fitting, X, time, event,
                fixed_latency=self.fixed_latency, return_models=True,
            )
            if solutions:
                phi, seed = max(solutions, key=lambda item: (item[1].final_loglik_, bool(item[1].converged_)))
                seed_coefficients = np.r_[seed.beta0_, seed.alpha_].astype("<f8")
                for retry in range(self.fitting["retries"] + 1):
                    budget = int(self.parameters["max_em_iter"]) * self.fitting["retry_multiplier"] ** retry
                    record = dict(source="profile_restart", retry=retry, profile_phi=phi,
                                  profile_loglik=float(seed.final_loglik_),
                                  latency_start=[float(seed.a_), float(seed.d_), float(seed.p_)],
                                  dispersion_start=max(float(phi), 1e-8),
                                  predictor_sha256=hashlib.sha256(seed_coefficients.tobytes()).hexdigest(),
                                  max_em_iter=budget, status="failed", failure_reason=None, converged=False)
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model = _resume_free_nb(seed, phi, time, event, budget)
                        record.update(_record_solution(model))
                        record.update(initial_loglik=model.continuation_initial_loglik_,
                                      training_predictions_valid=True,
                                      warnings=list(dict.fromkeys(str(item.message) for item in caught)))
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


def profile_dispersion(parameters, fitting, X, time, event, *, fixed_latency=None, return_models=False):
    """Reoptimize incidence and latency at each fixed phi and at Poisson.

    All points use the SAME kernel and penalty; this is an objective profile,
    not an invalid comparison of objectives across tuning configurations.
    """
    records, solutions = [], []
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
            if record["final_loglik"] is not None:
                solutions.append((phi, model))
        except Exception as error:
            record["failure_reason"] = f"{type(error).__name__}: {error}"
        record["attempts"] = fit.attempts_
        records.append(record)
    finite = [row["final_loglik"] for row in records if row["final_loglik"] is not None]
    maximum = max(finite, default=None)
    for row in records:
        row["gap_from_profile_max"] = None if maximum is None or row["final_loglik"] is None else maximum - row["final_loglik"]
    return (records, solutions) if return_models else records
