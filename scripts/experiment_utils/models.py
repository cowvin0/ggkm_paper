"""Model factories and ordered parameter grids shared by experiment runners."""

from __future__ import annotations

from functools import partial

from skcure.models import MixtureCureModel, PromotionTimeModel
from skcure.models.ggkm import GGKM

from models import SplineNegativeBinomialModel, SplinePoissonModel


KERNEL_MODEL_NAMES = frozenset({"bernoulli", "binomial", "poisson", "bn"})
PARAMETRIC_MODEL_NAMES = frozenset({"mixture_cure", "promotion_time"})
SPLINE_MODEL_NAMES = frozenset({"spline_poisson", "spline_bn"})


def make_model_registry(*, negative_binomial_max_em_iter=None):
    """Create independent model registrations in the experiment task order."""
    negative_binomial_params = {"latent_distribution": "negative_binomial"}
    if negative_binomial_max_em_iter is not None:
        negative_binomial_params["max_em_iter"] = negative_binomial_max_em_iter

    return {
        "bernoulli": partial(GGKM, latent_distribution="binomial"),
        "binomial": partial(GGKM, latent_distribution="binomial"),
        "poisson": partial(GGKM, latent_distribution="poisson"),
        "bn": partial(GGKM, **negative_binomial_params),
        "mixture_cure": MixtureCureModel,
        "promotion_time": PromotionTimeModel,
        "spline_poisson": SplinePoissonModel,
        "spline_bn": SplineNegativeBinomialModel,
    }


def build_model(estimator, kernel, params):
    if kernel is None:
        return estimator(**params)
    return estimator(kernel=kernel, **params)


def fixed_kernel_model_params(model_name, *, binomial_k):
    if model_name == "bernoulli":
        return {"K_bin": 1}
    if model_name == "binomial":
        return {"K_bin": binomial_k}
    return {}


def spline_parameter_grid(*, penalties, n_knots=(3, 5), degree=3):
    """Return fresh candidate dictionaries, ordered by knots then penalty."""
    return [
        {"n_knots": knots, "degree": degree, "lambda_reg": penalty}
        for knots in n_knots
        for penalty in penalties
    ]


def kernel_parameter_grid(
    model_name,
    kernel,
    *,
    binomial_k,
    penalties,
    gammas,
    polynomial_degrees,
    polynomial_coef0,
    wavelet_a_values,
    sigmoid_coef0=0.0,
    allow_sigmoid=True,
    require_nonempty=True,
    base_params=None,
):
    """Build candidates in the runners' original nested-loop order.

    Runner configuration is supplied at call time so adjustments to a runner's
    constants continue to apply to subsequent grids.
    """
    if base_params is None:
        base_params = fixed_kernel_model_params(model_name, binomial_k=binomial_k)

    grid = []

    if kernel == "linear":
        for lambda_reg in penalties:
            grid.append({**base_params, "lambda_reg": lambda_reg})

    elif kernel in {"rbf", "laplacian", "exponential", "cauchy"}:
        for gamma in gammas:
            for lambda_reg in penalties:
                grid.append(
                    {**base_params, "gamma": gamma, "lambda_reg": lambda_reg}
                )

    elif kernel == "sigmoid" and allow_sigmoid:
        for gamma in gammas:
            for lambda_reg in penalties:
                grid.append(
                    {
                        **base_params,
                        "gamma": gamma,
                        "coef0": sigmoid_coef0,
                        "lambda_reg": lambda_reg,
                    }
                )

    elif kernel == "polynomial":
        for gamma in gammas:
            for degree in polynomial_degrees:
                for lambda_reg in penalties:
                    grid.append(
                        {
                            **base_params,
                            "gamma": gamma,
                            "degree": degree,
                            "coef0": polynomial_coef0,
                            "lambda_reg": lambda_reg,
                        }
                    )

    elif kernel == "wavelet":
        for A in wavelet_a_values:
            for lambda_reg in penalties:
                grid.append({**base_params, "A": A, "lambda_reg": lambda_reg})

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if require_nonempty and not grid:
        raise RuntimeError(f"Empty parameter grid for kernel={kernel}.")

    return grid


def pcm_parameter_grid(model_name, kernel, *, kernel_models, binomial_k):
    if model_name in {"spline_poisson", "spline_bn"}:
        return spline_parameter_grid(penalties=(0.30, 3.00))

    if model_name not in kernel_models:
        return [{}]

    return kernel_parameter_grid(
        model_name,
        kernel,
        binomial_k=binomial_k,
        penalties=(0.30, 3.00),
        gammas=(0.25, 1.00),
        polynomial_degrees=(2, 3),
        polynomial_coef0=1.0,
        wavelet_a_values=(0.50, 1.00, 2.00),
        sigmoid_coef0=0.0,
        require_nonempty=False,
    )
