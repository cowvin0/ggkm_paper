"""Tests for the paper-specific spline cure-model wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from ggkm_paper.models import (
    GGNBPrimal,
    GGPoissonPrimal,
    SplineNegativeBinomialModel,
    SplinePoissonModel,
)
from ggkm_paper.tests.test_paper_em_primal import _data


@pytest.mark.parametrize(
    "estimator_class,inner_class,parameters",
    [
        (
            SplinePoissonModel,
            GGPoissonPrimal,
            {},
        ),
        (
            SplineNegativeBinomialModel,
            GGNBPrimal,
            {
                "phi": 0.7,
            },
        ),
        (
            SplineNegativeBinomialModel,
            GGNBPrimal,
            {
                "phi": None,
            },
        ),
    ],
)
def test_spline_models_use_paper_primal_estimators(
    estimator_class,
    inner_class,
    parameters,
):
    X, y = _data()

    model = estimator_class(
        n_knots=3,
        degree=2,
        max_em_iter=1,
        **parameters,
    ).fit(
        X,
        y,
    )

    assert isinstance(
        model.model_,
        inner_class,
    )


@pytest.mark.parametrize(
    "estimator_class,parameters",
    [
        (
            SplinePoissonModel,
            {},
        ),
        (
            SplineNegativeBinomialModel,
            {
                "phi": 0.7,
            },
        ),
        (
            SplineNegativeBinomialModel,
            {
                "phi": None,
            },
        ),
    ],
)
def test_spline_predictions_delegate_to_transformed_primal_model(
    estimator_class,
    parameters,
):
    X, y = _data()

    X_test = np.array(
        [
            [0.2, -0.4],
            [1.1, 0.7],
            [-0.3, 0.2],
        ],
        dtype=float,
    )

    model = estimator_class(
        n_knots=3,
        degree=2,
        max_em_iter=2,
        **parameters,
    ).fit(
        X,
        y,
    )

    X_spline = model.spline_.transform(
        X_test,
    )

    np.testing.assert_array_equal(
        model.predict_eta(X_test),
        model.model_.predict_eta(
            X_spline,
        ),
    )

    np.testing.assert_array_equal(
        model.predict_cure_probability(
            X_test,
        ),
        model.model_.predict_cure_probability(
            X_spline,
        ),
    )

    times = np.array(
        [
            0.0,
            1.0,
            3.1,
            5.5,
        ],
        dtype=float,
    )

    np.testing.assert_array_equal(
        model.predict_survival(
            X_test,
            times,
        ),
        model.model_.predict_survival(
            X_spline,
            times,
        ),
    )

    if hasattr(
        model.model_,
        "predict_theta",
    ):
        np.testing.assert_array_equal(
            model.predict_theta(
                X_test,
            ),
            model.model_.predict_theta(
                X_spline,
            ),
        )


@pytest.mark.parametrize(
    "estimator_class",
    [
        SplinePoissonModel,
        SplineNegativeBinomialModel,
    ],
)
def test_spline_representation_has_expected_dimensions(
    estimator_class,
):
    X, y = _data()

    model = estimator_class(
        n_knots=3,
        degree=2,
        max_em_iter=1,
    ).fit(
        X,
        y,
    )

    transformed = model.spline_.transform(
        X,
    )

    assert model.spline_.n_features_in_ == X.shape[1]

    assert transformed.shape == (
        X.shape[0],
        model.spline_.n_features_out_,
    )

    assert model.model_.n_features_in_ == transformed.shape[1]

    assert model.coef_.shape == (transformed.shape[1],)


@pytest.mark.parametrize(
    "estimator_class,parameters",
    [
        (
            SplinePoissonModel,
            {},
        ),
        (
            SplineNegativeBinomialModel,
            {
                "phi": 0.7,
            },
        ),
        (
            SplineNegativeBinomialModel,
            {
                "phi": None,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "warm_start",
    [
        False,
        True,
    ],
    ids=[
        "cold",
        "warm",
    ],
)
def test_spline_refit_preserves_wrapper_contract(
    estimator_class,
    parameters,
    warm_start,
):
    X, y = _data()

    model = estimator_class(
        n_knots=3,
        degree=2,
        lambda_reg=0.3,
        max_em_iter=1,
        warm_start=warm_start,
        **parameters,
    )

    assert (
        model.fit(
            X,
            y,
        )
        is model
    )

    first_design = model.spline_.transform(
        X,
    )

    assert (
        model.fit(
            X,
            y,
        )
        is model
    )

    second_design = model.spline_.transform(
        X,
    )

    np.testing.assert_array_equal(
        first_design,
        second_design,
    )

    assert model.coef_.shape == (second_design.shape[1],)


def test_spline_poisson_exposes_theta_prediction():
    X, y = _data()

    model = SplinePoissonModel(
        n_knots=3,
        degree=2,
        max_em_iter=1,
    ).fit(
        X,
        y,
    )

    theta = model.predict_theta(
        X[:3],
    )

    assert theta.shape == (3,)
    assert np.all(np.isfinite(theta))
    assert np.all(theta > 0.0)


def test_spline_negative_binomial_does_not_expose_theta_prediction():
    model = SplineNegativeBinomialModel()

    assert not hasattr(
        model.model_ if hasattr(model, "model_") else GGNBPrimal(),
        "predict_theta",
    )
