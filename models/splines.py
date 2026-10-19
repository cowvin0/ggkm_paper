"""Spline cure baselines used by the GGKM paper experiments."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_scalar

from skcure.base import _BaseParametricCureEstimator
from .gg_primal import GGNBPrimal, GGPoissonPrimal


class _SplineGGModel(_BaseParametricCureEstimator):
    """Shared spline feature mapping for the generalized-gamma primal models."""

    estimator_class = None
    _default_max_em_iter = 1000

    def __init__(
        self,
        n_knots=5,
        degree=3,
        lambda_reg=0.3,
        a=1.0,
        d=1.0,
        p=1.0,
        tol=1e-6,
        max_em_iter=1000,
        warm_start=False,
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.lambda_reg = lambda_reg
        self.a = a
        self.d = d
        self.p = p
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start

    def _make_spline(self):
        return SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
            sparse_output=False,
        )

    def fit(self, X, y):
        """Fit spline features and the primal model to a structured target."""
        model = self.__dict__.get("model_") if self.__sklearn_is_fitted__() else None
        X, t, _ = self._prepare_fit(
            X,
            y,
            nonnegative=("lambda_reg",),
            default_max_em_iter=self._default_max_em_iter,
        )
        for name, minimum in (("n_knots", 2), ("degree", 1)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
            check_scalar(value, name, numbers.Integral, min_val=minimum)

        spline = self._make_spline()
        X_spline = spline.fit_transform(X)
        parameters = {
            "a": self.a,
            "d": self.d,
            "p": self.p,
            "lambda_reg": self.lambda_reg,
            "tol": self.tol,
            "max_em_iter": self.max_em_iter,
            "warm_start": self.warm_start,
        }
        if hasattr(self, "phi"):
            parameters["phi"] = self.phi
        if model is None or not self.warm_start:
            model = self.estimator_class(**parameters)
        else:
            model.set_params(**parameters)
        model.fit(X_spline, y)

        self.spline_ = spline
        self.model_ = model
        self.X_train_ = X
        for name in (
            "a_", "d_", "p_", "phi_", "beta0_", "alpha_", "coef_",
            "intercept_", "loglik_history_", "converged_", "n_iter_",
            "final_loglik_",
        ):
            if hasattr(model, name):
                setattr(self, name, getattr(model, name))
        self._finish_fit(t)
        return self

    def _transform(self, X):
        X = self._validate_predict_data(X)
        return self.spline_.transform(X)

    def predict_eta(self, X):
        """Return the risk predictor after applying the fitted spline mapping."""
        X_spline = self._transform(X)
        return self.model_.predict_eta(X_spline)

    def predict_theta(self, X):
        """Return the latent intensity after applying the fitted spline mapping."""
        X_spline = self._transform(X)
        return self.model_.predict_theta(X_spline)

    def predict_survival(self, X, times):
        X_spline = self._transform(X)
        return self.model_.predict_survival(X_spline, times)

    def predict_cure_probability(self, X):
        X_spline = self._transform(X)
        return self.model_.predict_cure_probability(X_spline)


class SplinePoissonModel(_SplineGGModel):
    """Generalized-gamma Poisson cure model on quantile spline features.

    ``coef_`` describes the transformed spline features; ``n_features_in_``
    describes the original input features.
    """

    estimator_class = GGPoissonPrimal


class SplineNegativeBinomialModel(_SplineGGModel):
    """Generalized-gamma negative-binomial cure model on spline features.

    ``phi=None`` estimates dispersion; a positive ``phi`` holds it fixed.
    ``coef_`` describes the transformed spline features; ``n_features_in_``
    describes the original input features.
    """

    estimator_class = GGNBPrimal
    _default_max_em_iter = 10000

    def __init__(
        self,
        n_knots=5,
        degree=3,
        lambda_reg=0.3,
        a=1.0,
        d=1.0,
        p=1.0,
        phi=None,
        tol=1e-6,
        max_em_iter=10000,
        warm_start=False,
    ):
        super().__init__(
            n_knots=n_knots,
            degree=degree,
            lambda_reg=lambda_reg,
            a=a,
            d=d,
            p=p,
            tol=tol,
            max_em_iter=max_em_iter,
            warm_start=warm_start,
        )
        self.phi = phi
