"""Regression coverage for the paper models' scikit-cure adapter."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from models import (
    GGNBPrimal,
    GGPoissonPrimal,
    SplineNegativeBinomialModel,
    SplinePoissonModel,
)
from skcure.linear_model._base import _BaseLinearCureModel
from tests.test_paper_em_primal import _data


@pytest.mark.parametrize("estimator_class", [GGPoissonPrimal, GGNBPrimal])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_rescaled_incidence_keeps_penalty_on_original_coefficients(
    estimator_class, fit_intercept
):
    model = estimator_class(lambda_reg=0.4)
    model.fit_intercept = fit_intercept
    features = np.array(
        [[0.02, 40.0, 0.0], [-0.04, 80.0, 0.0],
         [0.06, -20.0, 0.0], [0.08, -60.0, 0.0]]
    )
    model.W_train_ = model._design_matrix(features)
    model._reset_components_for_fit()
    original = np.array([-0.3, 2.0, 0.005, 0.7])
    scales = np.array([2.0, 10.0, 0.5, 1.0])
    if not fit_intercept:
        original = original[1:]
        scales = scales[1:]
    design = model.W_train_ / scales
    working = original * scales
    expected_counts = np.array([0.5, 1.0, 2.0, 1.5])

    value, gradient = model._predictor_value_and_grad(
        working, expected_counts, design_matrix=design
    )
    original_value, original_gradient = model._predictor_value_and_grad(
        original, expected_counts
    )
    unpenalized_value, unpenalized_gradient = (
        _BaseLinearCureModel._predictor_value_and_grad(
            model, working, expected_counts, design_matrix=design
        )
    )
    slopes = original[1:] if fit_intercept else original
    penalty_gradient = 0.4 * original / scales
    if fit_intercept:
        penalty_gradient[0] = 0.0

    assert value == pytest.approx(original_value)
    assert value - unpenalized_value == pytest.approx(0.2 * (slopes @ slopes))
    np.testing.assert_allclose(gradient, original_gradient / scales)
    np.testing.assert_allclose(
        gradient - unpenalized_gradient, penalty_gradient, atol=1e-12
    )
    numerical_gradient = np.empty_like(working)
    for index in range(working.size):
        offset = np.zeros_like(working)
        offset[index] = 1e-6
        upper = model._predictor_value_and_grad(
            working + offset, expected_counts, design_matrix=design
        )[0]
        lower = model._predictor_value_and_grad(
            working - offset, expected_counts, design_matrix=design
        )[0]
        numerical_gradient[index] = (upper - lower) / 2e-6
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-6, atol=1e-7)


@pytest.mark.filterwarnings("ignore:EM did not converge:sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize(
    "estimator_class,expected_stages",
    [
        (SplinePoissonModel, {"incidence", "latency"}),
        (SplineNegativeBinomialModel, {"incidence", "latency", "dispersion"}),
    ],
)
def test_spline_forwards_new_estimator_controls_predictions_and_diagnostics(
    estimator_class, expected_stages
):
    X, y = _data()
    options = {
        "incidence": {"maxiter": 1000, "ftol": 1e-9},
        "latency": {"maxiter": 1000},
        "dispersion": {"maxiter": 200},
    }
    original_options = deepcopy(options)
    model = estimator_class(
        n_knots=3,
        degree=2,
        max_em_iter=2,
        em_rtol=0.01,
        solver_options=options,
    )
    cloned = clone(model)
    assert cloned.get_params()["solver_options"] == options
    assert cloned.get_params()["em_rtol"] == 0.01
    model.fit(X, y)

    assert options == original_options
    assert model.model_.solver_options == options
    assert model.model_.em_rtol == 0.01
    assert model.termination_reason_ == model.model_.termination_reason_
    assert model.termination_reason_ in {"converged", "max_em_iter"}
    assert model.optimizer_results_ == model.model_.optimizer_results_
    assert {result["stage"] for result in model.optimizer_results_} == expected_stages
    assert all(result["success"] for result in model.optimizer_results_)

    X_test = X[:3]
    transformed = model.spline_.transform(X_test)
    times = np.array([0.0, 1.0, 4.0])
    np.testing.assert_array_equal(
        model.predict_latent_mean(X_test),
        model.model_.predict_latent_mean(transformed),
    )
    np.testing.assert_array_equal(
        model.predict_log_survival(X_test, times),
        model.model_.predict_log_survival(transformed, times),
    )


@pytest.mark.filterwarnings("ignore:EM did not converge:sklearn.exceptions.ConvergenceWarning")
def test_spline_log_survival_and_cumulative_hazard_survive_probability_underflow():
    X, y = _data()
    model = SplinePoissonModel(n_knots=3, degree=2, max_em_iter=1).fit(X, y)
    model.model_.beta0_ = 20.0
    model.model_.alpha_ = np.zeros_like(model.model_.alpha_)

    times = np.array([0.0, 1e6])
    survival = model.predict_survival(X[:2], times)
    log_survival = model.predict_log_survival(X[:2], times)

    assert np.all(survival[:, -1] == 0.0)
    assert np.all(np.isfinite(log_survival))
    assert np.all(log_survival[:, -1] < -1000.0)
    np.testing.assert_allclose(np.exp(log_survival), survival)
    hazards = model.predict_cumulative_hazard_function(X[:2], return_array=True)
    assert np.all(np.isfinite(hazards))
    np.testing.assert_array_equal(
        hazards, -model.predict_log_survival(X[:2], model.unique_times_)
    )


def test_relative_em_tolerance_can_end_fit_before_iteration_limit():
    X, y = _data()
    model = GGPoissonPrimal(tol=1e-15, em_rtol=1e6, max_em_iter=3).fit(X, y)

    assert model.n_iter_ == 1
    assert model.converged_
    assert model.termination_reason_ == "converged"


@pytest.mark.parametrize(
    "estimator_class,hook",
    [
        (GGPoissonPrimal, "_reset_components_for_fit"),
        (GGNBPrimal, "_reset_components_for_fit"),
        (SplinePoissonModel, "_make_spline"),
        (SplineNegativeBinomialModel, "_make_spline"),
    ],
)
def test_failure_after_validation_marks_model_unfitted(
    estimator_class, hook, monkeypatch
):
    X, y = _data()
    model = estimator_class(max_em_iter=1)

    def failed(*args, **kwargs):
        raise ArithmeticError("injected initialization failure")

    monkeypatch.setattr(model, hook, failed)
    with pytest.raises(ArithmeticError, match="injected initialization failure"):
        model.fit(X, y)

    assert model.termination_reason_ == "failed"
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        model.predict_latent_mean(X[:2])


@pytest.mark.parametrize(
    "estimator_class", [SplinePoissonModel, SplineNegativeBinomialModel]
)
def test_spline_retains_partial_optimizer_progress_after_latency_failure(
    estimator_class, monkeypatch
):
    X, y = _data()
    model = estimator_class(n_knots=3, degree=2, max_em_iter=2)

    def failed(*args, **kwargs):
        raise ArithmeticError("injected latency gradient failure")

    monkeypatch.setattr(
        model.estimator_class, "_gg_log_value_and_grad", failed
    )
    with pytest.raises(ArithmeticError, match="injected latency gradient failure"):
        model.fit(X, y)

    assert model.termination_reason_ == "failed"
    assert model.n_iter_ == 1
    assert not model.__sklearn_is_fitted__()
    assert model.loglik_history_ == []
    incidence_results = [
        result for result in model.optimizer_results_
        if result["stage"] == "incidence"
    ]
    assert len(incidence_results) == 1
    assert incidence_results[0]["success"]
    with pytest.raises(NotFittedError):
        model.predict_survival(X[:2], [1.0])
