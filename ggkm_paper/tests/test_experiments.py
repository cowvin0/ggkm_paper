"""The paper entry points continue to construct and evaluate relocated models."""

import importlib

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

pytest.importorskip("optuna")
pytest.importorskip("tqdm")

from ggkm_paper.evaluate.simulated_data import simulate_pcm
from ggkm_paper.models import SplineNegativeBinomialModel, SplinePoissonModel
from skcure.base import BaseCureEstimator
from skcure.models import GGKM, MixtureCureModel, PromotionTimeModel


SCRIPT_NAMES = (
    "run_pcm_experiment",
    "run_melanoma_experiment",
    "run_breast_cancer_experiment",
    "run_default_experiment",
    "run_melanoma_fosp_experiment",
)
SPLINE_CASES = (
    ("spline_poisson", SplinePoissonModel),
    ("spline_bn", SplineNegativeBinomialModel),
)
SMALL_PARAMS = {
    "n_knots": 3,
    "degree": 2,
    "lambda_reg": 3.0,
    "max_em_iter": 1,
}


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_paper_entrypoint_registries_import(script_name):
    script = importlib.import_module(f"ggkm_paper.scripts.{script_name}")
    for name, distribution in (
        ("bernoulli", "binomial"),
        ("binomial", "binomial"),
        ("poisson", "poisson"),
        ("bn", "negative_binomial"),
    ):
        model = script.MODELS[name]()
        assert isinstance(model, GGKM)
        assert model.latent_distribution == distribution

    if "mixture_cure" in script.MODELS:
        assert script.MODELS["mixture_cure"] is MixtureCureModel
        assert script.MODELS["promotion_time"] is PromotionTimeModel

    if "spline_poisson" in script.MODELS:
        for name, estimator in SPLINE_CASES:
            assert script.MODELS[name] is estimator
            experiments = script.build_experiments()
            assert any(
                experiment["model_name"] == name and experiment["kernel"] is None
                for experiment in experiments
            )


@pytest.mark.parametrize("model_name, estimator", SPLINE_CASES)
def test_simulation_entrypoint_fits_relocated_spline_models(
    monkeypatch, model_name, estimator,
):
    script = importlib.import_module("ggkm_paper.scripts.run_pcm_experiment")
    monkeypatch.setattr(script, "N_REPLICATIONS", 1)
    monkeypatch.setattr(script, "INNER_GRID_POINTS", 6)
    monkeypatch.setattr(script, "OUTER_GRID_POINTS", 8)
    monkeypatch.setattr(script, "parameter_grid", lambda *args: [
        dict(SMALL_PARAMS),
        {**SMALL_PARAMS, "lambda_reg": 1.0},
    ])

    results, diagnostics = script.run_experiment(
        model_name=model_name,
        n=100,
        method=1,
        generating_distribution="poisson",
        generating_phi=None,
        kernel=None,
    )

    assert results["status"].tolist() == ["ok"], results["error"].tolist()
    assert results["model"].tolist() == [model_name]
    assert results["n_iter"].tolist() == [1]
    assert np.all(np.isfinite(results["best_inner_ibs"]))
    assert np.all(np.isfinite(results["test_ibs"]))
    assert diagnostics["predicted_survival_test"].shape == (1, 30, 8)
    assert diagnostics["predicted_cure_test"].shape == (1, 30)
    assert diagnostics["predicted_cure_surface"].shape == (1, 41**2)
    assert issubclass(estimator, BaseCureEstimator)


@pytest.fixture(scope="module")
def synthetic_inner_splits():
    data = simulate_pcm(n=80, method=1, seed=53)
    X = data[["x1", "x2"]].to_numpy(dtype=float)
    time = data["time"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=int)
    splits = []
    for train, validation in StratifiedKFold(2, shuffle=True, random_state=5).split(X, event):
        scaler = StandardScaler()
        grid = np.linspace(0.0, np.quantile(time[train], 0.75), 8)
        splits.append((
            scaler.fit_transform(X[train]), time[train], event[train],
            scaler.transform(X[validation]), time[validation], event[validation], grid,
        ))
    return splits


@pytest.mark.parametrize(
    "script_name", ["run_melanoma_experiment", "run_breast_cancer_experiment"],
)
@pytest.mark.parametrize("model_name, estimator", SPLINE_CASES)
def test_clinical_spline_tuning_uses_relocated_models(
    monkeypatch, synthetic_inner_splits, script_name, model_name, estimator,
):
    script = importlib.import_module(f"ggkm_paper.scripts.{script_name}")
    monkeypatch.setattr(script, "N_INNER_SPLITS", len(synthetic_inner_splits))
    monkeypatch.setattr(script, "SPLINE_PARAMETER_GRID", [dict(SMALL_PARAMS)])
    monkeypatch.setattr(script, "prepare_inner_splits", lambda *args, **kwargs: synthetic_inner_splits)

    params, score, method, evaluations, valid_evaluations = script.tune_model(
        None,
        estimator=script.MODELS[model_name],
        model_name=model_name,
        kernel=None,
        inner_seed=5,
    )

    assert params == SMALL_PARAMS
    assert np.isfinite(score)
    assert (method, evaluations, valid_evaluations) == ("grid", 1, 1)
    model = script.build_model(estimator, None, params)
    assert isinstance(model, BaseCureEstimator)
