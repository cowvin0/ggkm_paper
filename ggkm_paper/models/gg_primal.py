"""L2-regularized linear generalized-gamma cure models used by the GGKM paper."""

from __future__ import annotations

import numpy as np

from skcure.models._em import (
    _EstimatorEMState,
    _LinearCureSteps,
    run_em,
)
from skcure.models._linear_cure import _BaseLinearCureModel
from skcure.models.latent import (
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
)


class _NegativeBinomialPrimalSteps(_LinearCureSteps):
    """Add the NB dispersion update to the shared linear EM steps."""

    def maximize_dispersion(
        self,
        coefficients,
        expected,
    ):
        eta = self.model._predictor_from_coefficients(
            coefficients,
        )

        self.model._latent_model.update_after_latency_in_log_space_from_eta(
            self.model._latency,
            eta,
            self.t,
            self.delta,
        )

        return coefficients


def _fit_primal_em(
    model,
    t,
    delta,
    *,
    update_dispersion=False,
    validate_initial_loglik=True,
):
    """Fit an L2-regularized linear cure model through the shared EM engine."""
    coefficients = model._initial_coefficients(
        delta,
    )

    if update_dispersion:
        steps = _NegativeBinomialPrimalSteps(
            model,
            t,
            delta,
        )

        m_steps = (
            steps.maximize_predictor,
            steps.maximize_latency,
            steps.maximize_dispersion,
        )
    else:
        steps = _LinearCureSteps(
            model,
            t,
            delta,
        )

        m_steps = (
            steps.maximize_predictor,
            steps.maximize_latency,
        )

    initial_validator = (
        model._validate_initial_loglik
        if validate_initial_loglik
        else None
    )

    coefficients, ll_new = run_em(
        coefficients,
        e_step=steps.expectation,
        m_steps=m_steps,
        loglikelihood=steps.loglikelihood,
        state=_EstimatorEMState(model),
        tol=model.tol,
        max_iter=model._max_em_iter,
        validate_initial_loglik=initial_validator,
        nonfinite_message="Non-finite penalized log-likelihood.",
    )

    model._set_fitted_coefficients(
        coefficients,
    )

    model.final_loglik_ = ll_new
    model._latent_model.finalize_fit()


class _BaseGGPrimal(_BaseLinearCureModel):
    """Shared L2-regularized linear predictor for paper primal models.

    The incidence distribution, generalized-gamma latency implementation,
    likelihood, E-step, population survival, cure probability, and latency
    optimization are inherited from the shared scikit-cure components.

    This paper-specific layer adds only an L2 penalty on slope coefficients.
    The intercept is never penalized.
    """

    _initialize_intercept_from_cure = True
    _default_max_em_iter = 1000
    _update_dispersion = False
    _validate_initial_primal_loglik = True

    # All primal models used in the paper contain an intercept.
    fit_intercept = True

    def _penalty_value_and_grad(
        self,
        coefficients,
    ):
        """Return L2 penalty and its gradient."""
        coefficients = np.asarray(
            coefficients,
            dtype=float,
        )

        gradient = np.zeros_like(
            coefficients,
        )

        if self.fit_intercept:
            slopes = coefficients[1:]
            gradient[1:] = self.lambda_reg * slopes
        else:
            slopes = coefficients
            gradient[:] = self.lambda_reg * slopes

        penalty = (
            self.lambda_reg / 2.0
        ) * float(slopes @ slopes)

        return penalty, gradient

    def _loglikelihood(
        self,
        coefficients,
        t,
        delta,
    ):
        """Return penalized observed log-likelihood."""
        loglik = super()._loglikelihood(
            coefficients,
            t,
            delta,
        )

        if not np.isfinite(loglik):
            return loglik

        penalty, _ = self._penalty_value_and_grad(
            coefficients,
        )

        value = loglik - penalty

        if not np.isfinite(value):
            return -np.inf

        return float(value)

    def _predictor_value_and_grad(
        self,
        coefficients,
        expected,
    ):
        """Return penalized incidence objective and gradient."""
        value, gradient = super()._predictor_value_and_grad(
            coefficients,
            expected,
        )

        if not np.isfinite(value):
            return value, gradient

        penalty, penalty_gradient = self._penalty_value_and_grad(
            coefficients,
        )

        value = value + penalty
        gradient = gradient + penalty_gradient

        if (
            not np.isfinite(value)
            or np.any(~np.isfinite(gradient))
        ):
            return (
                np.inf,
                np.zeros_like(
                    coefficients,
                    dtype=float,
                ),
            )

        return (
            float(value),
            gradient,
        )

    def _set_fitted_coefficients(
        self,
        coefficients,
    ):
        """Publish primal coefficients using the historical paper names."""
        coefficients = np.asarray(
            coefficients,
            dtype=float,
        ).copy()

        if self.fit_intercept:
            self.beta0_ = float(
                coefficients[0]
            )
            self.alpha_ = coefficients[1:].copy()
        else:
            self.beta0_ = 0.0
            self.alpha_ = coefficients.copy()

        self.intercept_ = self.beta0_
        self.coef_ = self.alpha_.copy()

    def _fitted_coefficients(self):
        """Return the complete predictor vector used by the shared base."""
        if self.fit_intercept:
            return np.concatenate(
                (
                    np.array(
                        [self.beta0_],
                        dtype=float,
                    ),
                    self.alpha_,
                )
            )

        return self.alpha_.copy()

    def fit(
        self,
        X,
        y,
    ):
        """Fit the paper primal model to a structured survival target."""
        X, t, delta = self._prepare_fit(
            X,
            y,
            nonnegative=("lambda_reg",),
            booleans=("fit_intercept",),
            default_max_em_iter=self._default_max_em_iter,
        )

        self._reset_components_for_fit()

        self.X_train_ = X
        self.W_train_ = self._design_matrix(
            X,
        )

        _fit_primal_em(
            self,
            t,
            delta,
            update_dispersion=self._update_dispersion,
            validate_initial_loglik=self._validate_initial_primal_loglik,
        )

        self._publish_latent_state()

        self._finish_fit(
            t,
        )

        return self

    def _publish_latent_state(self):
        """Publish optional latent-family fitted state."""
        return None


class GGPoissonPrimal(_BaseGGPrimal):
    """Poisson generalized-gamma cure model with a linear log-intensity.

    Parameters
    ----------
    a, d, p : float, default=1.0
        Positive initial generalized-gamma parameters. Fitting estimates their
        values without changing these constructor parameters.

    lambda_reg : float, default=1e-3
        Non-negative L2 penalty on the regression coefficients. The intercept
        is unpenalized.

    tol : float, default=1e-6
        Absolute change in penalized log-likelihood required for convergence.

    max_em_iter : int or None, default=1000
        Maximum number of EM iterations. None uses the default limit.

    warm_start : bool, default=False
        Reuse fitted generalized-gamma parameters on a subsequent fit.
        Regression coefficients and the intercept are initialized again.

    Attributes
    ----------
    a_, d_, p_ : float
        Fitted generalized-gamma parameters.

    alpha_ : ndarray of shape (n_features,)
        Fitted regression coefficients.

    beta0_ : float
        Fitted intercept.

    coef_ : ndarray of shape (n_features,)
        Copy of ``alpha_``.

    intercept_ : float
        Fitted intercept, also available as ``beta0_``.

    unique_times_ : ndarray
        Sorted unique observed training times.

    loglik_history_ : list of float
        Penalized log-likelihood after each EM iteration.

    converged_ : bool
        Whether the EM convergence tolerance was reached.

    n_iter_ : int
        Number of completed EM iterations.

    final_loglik_ : float
        Final penalized log-likelihood.

    Notes
    -----
    The model uses the shared ``_PoissonLatentModel``,
    ``GeneralizedGamma``, generalized-gamma latency optimizer, and EM engine
    from ``skcure``.

    ``predict(X)`` returns the linear log-intensity, with larger values
    indicating higher risk.
    """

    _default_max_em_iter = 1000

    _model_name = "penalized"
    _predictor_error = "Non-finite Poisson predictor."
    _predictor_optimization_error = (
        "Poisson predictor optimization failed: "
    )

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        lambda_reg=1e-3,
        tol=1e-6,
        max_em_iter=1000,
        warm_start=False,
    ):
        self.a = a
        self.d = d
        self.p = p
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start

        self.fit_intercept = True

    def _make_latent_model(self):
        """Create the shared Poisson latent-count strategy."""
        return _PoissonLatentModel()

    def predict_theta(
        self,
        X,
    ):
        """Return the predicted Poisson latent intensity."""
        eta = self.predict_eta(
            X,
        )

        return self._latent_model.predict_theta_from_eta(
            eta,
        )


class GGNBPrimal(_BaseGGPrimal):
    """Negative-binomial generalized-gamma cure model with a linear predictor.

    Parameters
    ----------
    a, d, p : float, default=1.0
        Positive initial generalized-gamma parameters. Fitting estimates their
        values without changing these constructor parameters.

    phi : float or None, default=None
        Positive fixed dispersion. If None, dispersion is estimated starting
        from 0.5 on the first fit.

    lambda_reg : float, default=1e-3
        Non-negative L2 penalty on the regression coefficients. The intercept
        is unpenalized.

    tol : float, default=1e-6
        Absolute change in penalized log-likelihood required for convergence.

    max_em_iter : int or None, default=10000
        Maximum number of EM iterations. None uses the paper model's default
        limit of 10000.

    warm_start : bool, default=False
        Reuse fitted generalized-gamma parameters and an estimated dispersion
        on a subsequent fit. Regression coefficients and the intercept are
        initialized again.

    Attributes
    ----------
    a_, d_, p_ : float
        Fitted generalized-gamma parameters.

    phi_ : float
        Fixed or estimated dispersion used by the fitted model.

    alpha_ : ndarray of shape (n_features,)
        Fitted regression coefficients.

    beta0_ : float
        Fitted intercept.

    coef_ : ndarray of shape (n_features,)
        Copy of ``alpha_``.

    intercept_ : float
        Fitted intercept, also available as ``beta0_``.

    unique_times_ : ndarray
        Sorted unique observed training times.

    loglik_history_ : list of float
        Penalized log-likelihood after each EM iteration.

    converged_ : bool
        Whether the EM convergence tolerance was reached.

    n_iter_ : int
        Number of completed EM iterations.

    final_loglik_ : float
        Final penalized log-likelihood.

    Notes
    -----
    The model uses the shared ``_NegativeBinomialLatentModel`` for the
    likelihood, E-step, incidence objective, dispersion optimization,
    survival prediction, and cure-probability prediction.

    The EM update order remains

    ``predictor -> generalized-gamma latency -> dispersion``.
    """

    _default_max_em_iter = 10000
    _update_dispersion = True

    # The historical NB paper implementation did not reject the initial
    # likelihood separately before entering the EM loop.
    _validate_initial_primal_loglik = False

    _model_name = "penalized"
    _predictor_error = "Non-finite negative-binomial predictor."
    _predictor_optimization_error = (
        "Negative-binomial predictor optimization failed: "
    )

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        phi=None,
        lambda_reg=1e-3,
        tol=1e-6,
        max_em_iter=10000,
        warm_start=False,
    ):
        self.a = a
        self.d = d
        self.p = p
        self.phi = phi
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start

        self.fit_intercept = True

    def _make_latent_model(self):
        """Create the shared negative-binomial latent-count strategy."""
        return _NegativeBinomialLatentModel(
            phi=self.phi,
        )

    def _reset_components_for_fit(self):
        """Reset NB state while preserving estimated dispersion on warm fit."""
        warm_phi = None

        if (
            self.phi is None
            and self.warm_start
            and hasattr(self, "phi_")
            and self.phi_ is not None
        ):
            warm_phi = float(
                self.phi_
            )

        self._latent_model = self._make_latent_model()
        self._latent_model.reset_for_fit()

        if (
            warm_phi is not None
            and self._latent_model.estimate_phi
        ):
            self._latent_model.phi = warm_phi

    def _validate_predictor_result(
        self,
        result,
    ):
        """Preserve the paper NB predictor optimizer acceptance rule."""
        if (
            not result.success
            or np.any(~np.isfinite(result.x))
        ):
            raise RuntimeError(
                "Negative-binomial predictor optimization failed: "
                f"{result.message}"
            )

    def _publish_latent_state(self):
        """Expose fitted NB dispersion through the estimator API."""
        self.phi_ = float(
            self._latent_model.phi_
        )


__all__ = [
    "GGPoissonPrimal",
    "GGNBPrimal",
]
