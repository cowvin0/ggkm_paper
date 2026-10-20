"""Regression coverage for behavior that differs between experiment runners."""

import importlib

import pytest


pytest.importorskip("optuna")
pytest.importorskip("tqdm")


def runner(dataset):
    return importlib.import_module(f"scripts.run_{dataset}_experiment")


def test_breast_cancer_retains_sigmoid_grid_and_em_limit():
    script = runner("breast_cancer")
    candidates = script.kernel_parameter_grid("bernoulli", "sigmoid")

    assert script.MODELS["bn"]().max_em_iter == 10000
    assert "sigmoid" in script.KERNELS
    assert [(item["gamma"], item["lambda_reg"]) for item in candidates] == [
        (0.25, 0.30),
        (0.25, 3.00),
        (1.00, 0.30),
        (1.00, 3.00),
    ]
    assert all(item["K_bin"] == 1 and item["coef0"] == 0.0 for item in candidates)
    assert all(
        item["A"] != 10
        for item in script.kernel_parameter_grid("poisson", "wavelet")
    )


def test_melanoma_retains_extra_wavelet_scale_and_rejects_sigmoid():
    script = runner("melanoma")
    candidates = script.kernel_parameter_grid("poisson", "wavelet")

    assert [item["A"] for item in candidates] == [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 10, 10]
    assert "max_em_iter" not in script.MODELS["bn"].keywords
    assert "sigmoid" not in script.KERNELS
    with pytest.raises(ValueError, match="^Unknown kernel: sigmoid$"):
        script.kernel_parameter_grid("poisson", "sigmoid")


@pytest.mark.parametrize("model_name", ["spline_poisson", "spline_bn"])
def test_pcm_retains_distinct_spline_penalties(model_name):
    candidates = runner("pcm").parameter_grid(model_name, None)

    assert [(item["n_knots"], item["lambda_reg"]) for item in candidates] == [
        (3, 0.30),
        (3, 3.00),
        (5, 0.30),
        (5, 3.00),
    ]
    for dataset in ("breast_cancer", "melanoma"):
        assert [item["lambda_reg"] for item in runner(dataset).SPLINE_PARAMETER_GRID] == [
            0.10, 1.00, 0.10, 1.00,
        ]


@pytest.mark.parametrize("dataset", ["breast_cancer", "melanoma"])
def test_clinical_grids_use_current_runner_settings(monkeypatch, dataset):
    script = runner(dataset)
    for name, value in (
        ("BINOMIAL_K", 7),
        ("KERNEL_PENALTIES", [11.0, 13.0]),
        ("KERNEL_GAMMAS", [17.0]),
        ("POLYNOMIAL_DEGREES", [4]),
        ("POLYNOMIAL_COEF0", 19.0),
        ("WAVELET_A_VALUES", [23.0]),
    ):
        monkeypatch.setattr(script, name, value)

    candidates = script.kernel_parameter_grid("binomial", "polynomial")
    assert len(candidates) == 2
    assert candidates[0] == {
        "K_bin": 7, "gamma": 17.0, "degree": 4, "coef0": 19.0, "lambda_reg": 11.0,
    }
    assert candidates[1]["lambda_reg"] == 13.0
    assert [item["A"] for item in script.kernel_parameter_grid("poisson", "wavelet")] == [
        23.0, 23.0,
    ]

    if dataset == "breast_cancer":
        monkeypatch.setattr(script, "SIGMOID_COEF0", -2.0)
        assert script.kernel_parameter_grid("poisson", "sigmoid")[0]["coef0"] == -2.0


def test_pcm_grid_uses_current_model_settings(monkeypatch):
    script = runner("pcm")
    monkeypatch.setattr(script, "BINOMIAL_K", 7)
    assert script.parameter_grid("binomial", "linear")[0]["K_bin"] == 7

    monkeypatch.setattr(script, "KERNEL_MODELS", set())
    assert script.parameter_grid("binomial", "linear") == [{}]


@pytest.mark.parametrize("dataset", ["breast_cancer", "melanoma"])
def test_spline_tuning_skips_failures_and_keeps_first_best(monkeypatch, dataset):
    script = runner(dataset)
    candidates = [{"lambda_reg": value} for value in (0.1, 1.0, 3.0, 10.0)]
    prepared = object()
    estimator = object()
    seen = []
    monkeypatch.setattr(script, "SPLINE_PARAMETER_GRID", candidates)
    monkeypatch.setattr(script, "prepare_inner_splits", lambda *args, **kwargs: prepared)

    def score(inner_splits, *, estimator, kernel, params):
        assert inner_splits is prepared
        assert kernel is None
        seen.append(params["lambda_reg"])
        if params["lambda_reg"] == 0.1:
            raise ValueError("candidate failed")
        if params["lambda_reg"] == 1.0:
            return float("inf")
        return 0.4

    monkeypatch.setattr(script, "score_params_on_inner_splits", score)
    result = script.tune_spline_grid(None, estimator=estimator, inner_seed=5)

    assert seen == [0.1, 1.0, 3.0, 10.0]
    assert result == ({"lambda_reg": 3.0}, 0.4, 4, 2)
    result[0]["lambda_reg"] = -1
    assert candidates[2]["lambda_reg"] == 3.0


@pytest.mark.parametrize("dataset", ["breast_cancer", "melanoma"])
@pytest.mark.parametrize("failure_kind", ["exception", "nonfinite"])
def test_spline_tuning_retains_dataset_failure_messages(monkeypatch, dataset, failure_kind):
    script = runner(dataset)
    monkeypatch.setattr(script, "SPLINE_PARAMETER_GRID", [{"n_knots": 3}, {"n_knots": 5}])
    monkeypatch.setattr(script, "prepare_inner_splits", lambda *args, **kwargs: [])
    seen = []

    def score(*args, **kwargs):
        seen.append(kwargs["params"]["n_knots"])
        if failure_kind == "exception":
            raise ValueError("candidate failed")
        return float("nan")

    monkeypatch.setattr(script, "score_params_on_inner_splits", score)

    with pytest.raises(RuntimeError) as error:
        script.tune_spline_grid(None, estimator=object(), inner_seed=5)

    if dataset == "breast_cancer":
        expected = "All spline-grid candidates failed during inner tuning."
        if failure_kind == "exception":
            expected += " First failure: ValueError: candidate failed"
    else:
        expected = "All spline parameter candidates failed during inner tuning."
    assert str(error.value) == expected
    assert seen == [3, 5]
