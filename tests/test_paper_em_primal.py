"""Architecture and behavioral tests for the paper primal cure models."""

from __future__ import annotations

import numpy as np
import pytest
from sksurv.util import Surv

from ggkm_paper.models import (
    GGNBPrimal,
    GGPoissonPrimal,
)
from skcure.distributions import GeneralizedGamma
from skcure.models._linear_cure import _BaseLinearCureModel
from skcure.models.latent import (
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
)


def _data():
    rng = np.random.default_rng(73)

    X = rng.normal(
        size=(24, 2),
    )

    time = np.round(
        rng.uniform(
            0.4,
            3.2,
            size=24,
        ),
        2,
    )

    event = np.tile(
        [
            True,
            False,
            True,
            True,
        ],
        6,
    )

    y = Surv.from_arrays(
        event,
        time,
    )

    return X, y


@pytest.mark.parametrize(
    "estimator_class,latent_class",
    [
        (
            GGPoissonPrimal,
            _PoissonLatentModel,
        ),
        (
            GGNBPrimal,
            _NegativeBinomialLatentModel,
        ),
    ],
)
def test_primal_models_use_shared_linear_cure_architecture(
    estimator_class,
    latent_class,
):
    X, y = _data()

    model = estimator_class(
        max_em_iter=1,
    ).fit(
        X,
        y,
    )

    assert isinstance(
        model,
        _BaseLinearCureModel,
    )

    assert isinstance(
        model._latency,
        GeneralizedGamma,
    )

    assert isinstance(
        model._latent_model,
        latent_class,
    )


@pytest.mark.parametrize(
    "estimator_class",
    [
        GGPoissonPrimal,
        GGNBPrimal,
    ],
)
def test_primal_coefficients_reproduce_linear_predictor(
    estimator_class,
):
    X, y = _data()

    model = estimator_class(
        max_em_iter=2,
    ).fit(
        X,
        y,
    )

    expected = model.beta0_ + X @ model.alpha_

    np.testing.assert_allclose(
        model.predict_eta(X),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_array_equal(
        model.coef_,
        model.alpha_,
    )

    assert model.intercept_ == model.beta0_


@pytest.mark.parametrize(
    "estimator_class",
    [
        GGPoissonPrimal,
        GGNBPrimal,
    ],
)
def test_l2_penalty_excludes_intercept(
    estimator_class,
):
    model = estimator_class(
        lambda_reg=0.4,
    )

    coefficients = np.array(
        [
            3.0,
            2.0,
            -1.0,
        ],
        dtype=float,
    )

    penalty, gradient = model._penalty_value_and_grad(
        coefficients,
    )

    expected_penalty = 0.4 / 2.0 * (2.0**2 + (-1.0) ** 2)

    assert penalty == pytest.approx(
        expected_penalty,
    )

    np.testing.assert_array_equal(
        gradient,
        np.array(
            [
                0.0,
                0.8,
                -0.4,
            ]
        ),
    )


def test_poisson_primal_delegates_theta_prediction():
    X, y = _data()

    model = GGPoissonPrimal(
        max_em_iter=1,
    ).fit(
        X,
        y,
    )

    eta = model.predict_eta(
        X,
    )

    expected = model._latent_model.predict_theta_from_eta(
        eta,
    )

    np.testing.assert_array_equal(
        model.predict_theta(X),
        expected,
    )


@pytest.mark.parametrize(
    "estimator_class,latent_class",
    [
        (
            GGPoissonPrimal,
            _PoissonLatentModel,
        ),
        (
            GGNBPrimal,
            _NegativeBinomialLatentModel,
        ),
    ],
)
def test_primal_fit_uses_shared_latent_entrypoints(
    estimator_class,
    latent_class,
    monkeypatch,
):
    calls = {
        "likelihood": 0,
        "e_step": 0,
        "incidence": 0,
        "survival": 0,
        "cure": 0,
    }

    original_likelihood = latent_class.observed_loglikelihood_from_eta
    original_e_step = latent_class.e_step_from_eta
    original_incidence = latent_class.incidence_value_and_grad
    original_survival = latent_class.predict_survival_from_eta
    original_cure = latent_class.predict_cure_probability_from_eta

    def likelihood(
        self,
        latency,
        eta,
        t,
        delta,
    ):
        calls["likelihood"] += 1

        return original_likelihood(
            self,
            latency,
            eta,
            t,
            delta,
        )

    def e_step(
        self,
        latency,
        eta,
        t,
        delta,
    ):
        calls["e_step"] += 1

        return original_e_step(
            self,
            latency,
            eta,
            t,
            delta,
        )

    def incidence(
        self,
        eta,
        expected,
    ):
        calls["incidence"] += 1

        return original_incidence(
            self,
            eta,
            expected,
        )

    def survival(
        self,
        latency,
        eta,
        times,
    ):
        calls["survival"] += 1

        return original_survival(
            self,
            latency,
            eta,
            times,
        )

    def cure(
        self,
        latency,
        eta,
    ):
        calls["cure"] += 1

        return original_cure(
            self,
            latency,
            eta,
        )

    monkeypatch.setattr(
        latent_class,
        "observed_loglikelihood_from_eta",
        likelihood,
    )
    monkeypatch.setattr(
        latent_class,
        "e_step_from_eta",
        e_step,
    )
    monkeypatch.setattr(
        latent_class,
        "incidence_value_and_grad",
        incidence,
    )
    monkeypatch.setattr(
        latent_class,
        "predict_survival_from_eta",
        survival,
    )
    monkeypatch.setattr(
        latent_class,
        "predict_cure_probability_from_eta",
        cure,
    )

    X, y = _data()

    model = estimator_class(
        max_em_iter=1,
    ).fit(
        X,
        y,
    )

    model.predict_survival(
        X[:2],
        [
            0.0,
            1.0,
        ],
    )

    model.predict_cure_probability(
        X[:2],
    )

    assert calls["likelihood"] >= 2
    assert calls["e_step"] >= 1
    assert calls["incidence"] >= 1
    assert calls["survival"] >= 1
    assert calls["cure"] >= 1


def test_estimated_nb_dispersion_uses_shared_latent_update(
    monkeypatch,
):
    calls = 0

    original = _NegativeBinomialLatentModel.update_after_latency_in_log_space_from_eta

    def update(
        self,
        latency,
        eta,
        t,
        delta,
    ):
        nonlocal calls
        calls += 1

        return original(
            self,
            latency,
            eta,
            t,
            delta,
        )

    monkeypatch.setattr(
        _NegativeBinomialLatentModel,
        "update_after_latency_in_log_space_from_eta",
        update,
    )

    X, y = _data()

    model = GGNBPrimal(
        phi=None,
        max_em_iter=2,
    ).fit(
        X,
        y,
    )

    assert calls >= 1
    assert model._latent_model.estimate_phi
    assert model.phi_ > 0.0
    assert np.isfinite(model.phi_)


def test_fixed_nb_dispersion_does_not_change():
    X, y = _data()

    model = GGNBPrimal(
        phi=0.7,
        max_em_iter=2,
    ).fit(
        X,
        y,
    )

    assert not model._latent_model.estimate_phi
    assert model.phi_ == 0.7


def test_cold_refit_resets_estimated_nb_dispersion(
    monkeypatch,
):
    X, y = _data()

    model = GGNBPrimal(
        phi=None,
        max_em_iter=1,
        warm_start=False,
    ).fit(
        X,
        y,
    )

    first_phi = model.phi_

    class InitializationChecked(Exception):
        pass

    original = _NegativeBinomialLatentModel.e_step_from_eta

    def check_initialization(
        latent,
        latency,
        eta,
        t,
        delta,
    ):
        assert latent.phi == 0.5
        raise InitializationChecked

    monkeypatch.setattr(
        _NegativeBinomialLatentModel,
        "e_step_from_eta",
        check_initialization,
    )

    with pytest.raises(
        InitializationChecked,
    ):
        model.fit(
            X,
            y,
        )

    assert first_phi > 0.0


def test_warm_refit_reuses_estimated_nb_dispersion(
    monkeypatch,
):
    X, y = _data()

    model = GGNBPrimal(
        phi=None,
        max_em_iter=1,
        warm_start=True,
    ).fit(
        X,
        y,
    )

    fitted_phi = model.phi_

    class InitializationChecked(Exception):
        pass

    def check_initialization(
        latent,
        latency,
        eta,
        t,
        delta,
    ):
        assert latent.phi == fitted_phi
        raise InitializationChecked

    monkeypatch.setattr(
        _NegativeBinomialLatentModel,
        "e_step_from_eta",
        check_initialization,
    )

    with pytest.raises(
        InitializationChecked,
    ):
        model.fit(
            X,
            y,
        )


@pytest.mark.parametrize(
    "estimator_class",
    [
        GGPoissonPrimal,
        GGNBPrimal,
    ],
)
def test_shared_e_step_failure_propagates(
    estimator_class,
    monkeypatch,
):
    X, y = _data()

    model = estimator_class(
        max_em_iter=1,
    )

    def failed(
        *args,
        **kwargs,
    ):
        raise ArithmeticError("injected E-step failure")

    monkeypatch.setattr(
        model,
        "_e_step",
        failed,
    )

    with pytest.raises(
        ArithmeticError,
        match="injected E-step failure",
    ):
        model.fit(
            X,
            y,
        )


@pytest.mark.parametrize(
    "estimator_class",
    [
        GGPoissonPrimal,
        GGNBPrimal,
    ],
)
def test_shared_latency_failure_propagates(
    estimator_class,
    monkeypatch,
):
    X, y = _data()

    model = estimator_class(
        max_em_iter=1,
    )

    def failed(
        *args,
        **kwargs,
    ):
        raise ArithmeticError("injected latency failure")

    monkeypatch.setattr(
        model,
        "_gg_log_value_and_grad",
        failed,
    )

    with pytest.raises(
        ArithmeticError,
        match="injected latency failure",
    ):
        model.fit(
            X,
            y,
        )
