import os
import time

import numpy as np

from pathlib import Path
from functools import partial
from skcure.models.ggkm import GGKM
from skcure.models import MixtureCureModel, PromotionTimeModel
from ggkm_paper.models import (
    SplinePoissonModel,
    SplineNegativeBinomialModel,
)


from ggkm_paper.evaluate.cross_validation import (
    monte_carlo_pcm,
    summarize_monte_carlo,
)

KERNELS = [
    "linear",
    "rbf",
    "laplacian",
    "exponential",
    "cauchy",
    "polynomial",
    "wavelet",
]


SAMPLE_SIZES = [
    300,
    600,
]


METHODS = [
    1,
    2,
    3,
    4,
    5,
]


GENERATING_SCENARIOS = [
    {
        "name": "poisson",
        "distribution": "poisson",
        "phi": None,
    },
    {
        "name": "nb025",
        "distribution": "negative_binomial",
        "phi": 0.25,
    },
    {
        "name": "nb100",
        "distribution": "negative_binomial",
        "phi": 1.00,
    },
]

MODELS = {
    "bernoulli": partial(
        GGKM,
        latent_distribution="binomial",
    ),
    "binomial": partial(
        GGKM,
        latent_distribution="binomial",
    ),
    "poisson": partial(
        GGKM,
        latent_distribution="poisson",
    ),
    "bn": partial(
        GGKM,
        latent_distribution="negative_binomial",
    ),
    "mixture_cure": MixtureCureModel,
    "promotion_time": PromotionTimeModel,
    "spline_poisson": SplinePoissonModel,
    "spline_bn": SplineNegativeBinomialModel,
}

KERNEL_MODELS = {
    "bernoulli",
    "binomial",
    "poisson",
    "bn",
}


N_REPLICATIONS = 250
TEST_SIZE = 0.30
N_INNER_SPLITS = 2
INNER_GRID_POINTS = 24
OUTER_GRID_POINTS = 36
MASTER_SEED = 20260817

BINOMIAL_K = 2
OUTPUT_DIR = Path("data/results")


def parameter_grid(
    model_name,
    kernel,
):

    if model_name in {
        "spline_poisson",
        "spline_bn",
    }:

        grid = []

        for n_knots in [
            3,
            5,
        ]:
            for lambda_reg in [
                0.30,
                3.00,
            ]:
                grid.append(
                    {
                        "n_knots": n_knots,
                        "degree": 3,
                        "lambda_reg": lambda_reg,
                    }
                )

        return grid

    if model_name not in KERNEL_MODELS:
        return [{}]

    base_params = {}

    if model_name == "bernoulli":
        base_params["K_bin"] = 1

    elif model_name == "binomial":
        base_params["K_bin"] = BINOMIAL_K

    penalties = [
        0.30,
        3.00,
    ]

    grid = []

    if kernel == "linear":

        for lambda_reg in penalties:
            grid.append(
                {
                    **base_params,
                    "lambda_reg": lambda_reg,
                }
            )

    elif kernel in {
        "rbf",
        "laplacian",
        "exponential",
        "cauchy",
    }:

        for gamma in [
            0.25,
            1.00,
        ]:
            for lambda_reg in penalties:
                grid.append(
                    {
                        **base_params,
                        "gamma": gamma,
                        "lambda_reg": lambda_reg,
                    }
                )

    elif kernel == "sigmoid":

        for gamma in [
            0.25,
            1.00,
        ]:
            for lambda_reg in penalties:
                grid.append(
                    {
                        **base_params,
                        "gamma": gamma,
                        "coef0": 0.0,
                        "lambda_reg": lambda_reg,
                    }
                )

    elif kernel == "polynomial":

        for gamma in [
            0.25,
            1.00,
        ]:
            for degree in [
                2,
                3,
            ]:
                for lambda_reg in penalties:
                    grid.append(
                        {
                            **base_params,
                            "gamma": gamma,
                            "degree": degree,
                            "coef0": 1.0,
                            "lambda_reg": lambda_reg,
                        }
                    )

    elif kernel == "wavelet":

        for A in [0.50, 1.00, 2.00]:
            for lambda_reg in penalties:
                grid.append(
                    {
                        **base_params,
                        "A": A,
                        "lambda_reg": lambda_reg,
                    }
                )

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return grid


def build_experiments():
    experiments = []

    for generating_scenario in GENERATING_SCENARIOS:
        for model_name in MODELS:
            for n in SAMPLE_SIZES:
                for method in METHODS:

                    if model_name in KERNEL_MODELS:
                        for kernel in KERNELS:
                            experiments.append(
                                {
                                    "model_name": model_name,
                                    "n": n,
                                    "method": method,
                                    "generating_name": generating_scenario["name"],
                                    "generating_distribution": generating_scenario[
                                        "distribution"
                                    ],
                                    "generating_phi": generating_scenario["phi"],
                                    "kernel": kernel,
                                }
                            )

                    else:
                        experiments.append(
                            {
                                "model_name": model_name,
                                "n": n,
                                "method": method,
                                "generating_name": generating_scenario["name"],
                                "generating_distribution": generating_scenario[
                                    "distribution"
                                ],
                                "generating_phi": generating_scenario["phi"],
                                "kernel": None,
                            }
                        )

    return experiments


def get_experiment(
    experiments,
    task_id,
):
    if task_id >= len(experiments):
        raise ValueError(
            f"Invalid PBS_ARRAY_INDEX={task_id}. " f"Expected 0-{len(experiments)-1}."
        )

    return experiments[task_id]


def run_experiment(
    model_name,
    n,
    method,
    generating_distribution,
    generating_phi,
    kernel,
):
    estimator = MODELS[model_name]

    grid = parameter_grid(
        model_name,
        kernel,
    )

    return monte_carlo_pcm(
        n=n,
        method=method,
        generating_distribution=generating_distribution,
        generating_phi=generating_phi,
        generating_K_bin=BINOMIAL_K,
        estimator=estimator,
        estimator_name=model_name,
        kernel=kernel,
        parameter_grid=grid,
        replications=N_REPLICATIONS,
        test_size=TEST_SIZE,
        n_inner_splits=N_INNER_SPLITS,
        inner_grid_points=INNER_GRID_POINTS,
        outer_grid_points=OUTER_GRID_POINTS,
        random_state=MASTER_SEED,
        return_diagnostics=True,
        surface_grid_points=41,
    )


def save_results(
    results,
    diagnostics,
    task_id,
    model_name,
    n,
    method,
    generating_name,
    kernel,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_directory = OUTPUT_DIR / "diagnostics"

    diagnostics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    kernel_label = kernel if kernel is not None else "parametric"

    prefix = (
        f"task{task_id}_"
        f"{model_name}_"
        f"n{n}_"
        f"method{method}_"
        f"gen{generating_name}_"
        f"kernel{kernel_label}"
    )

    replication_file = OUTPUT_DIR / f"replications_{prefix}.csv"

    summary_file = OUTPUT_DIR / f"summary_{prefix}.csv"

    diagnostics_file = diagnostics_directory / f"diagnostics_{prefix}.npz"

    results.to_csv(
        replication_file,
        index=False,
    )

    summary = summarize_monte_carlo(results)

    summary.to_csv(
        summary_file,
        index=False,
    )

    np.savez_compressed(
        diagnostics_file,
        **diagnostics,
    )

    return (
        replication_file,
        summary_file,
        diagnostics_file,
    )


def main():
    start_time = time.time()

    task_id = int(os.environ["PBS_ARRAY_INDEX"])
    experiments = build_experiments()
    exp = get_experiment(
        experiments,
        task_id,
    )
    model_name = exp["model_name"]
    n = exp["n"]
    method = exp["method"]
    generating_name = exp["generating_name"]
    generating_distribution = exp["generating_distribution"]
    generating_phi = exp["generating_phi"]

    kernel = exp["kernel"]

    print(
        f"Task {task_id} started",
        flush=True,
    )

    print(
        f"Model={model_name} "
        f"n={n} "
        f"method={method} "
        f"generator={generating_name} "
        f"kernel={kernel}",
        flush=True,
    )

    if kernel is not None:
        print(
            f"Parameter grid: " f"{parameter_grid(model_name, kernel)}",
            flush=True,
        )

    (
        results,
        diagnostics,
    ) = run_experiment(
        model_name=model_name,
        n=n,
        method=method,
        generating_distribution=(generating_distribution),
        generating_phi=(generating_phi),
        kernel=kernel,
    )

    (
        replication_file,
        summary_file,
        diagnostics_file,
    ) = save_results(
        results=results,
        diagnostics=diagnostics,
        task_id=task_id,
        model_name=model_name,
        n=n,
        method=method,
        generating_name=(generating_name),
        kernel=kernel,
    )

    elapsed = time.time() - start_time

    successful = int((results["status"] == "ok").sum())

    failed = len(results) - successful

    print(
        f"Task {task_id} finished",
        flush=True,
    )

    print(
        f"Replications: " f"{len(results)}",
        flush=True,
    )

    print(
        f"Successful: " f"{successful}",
        flush=True,
    )

    print(
        f"Failed: " f"{failed}",
        flush=True,
    )

    print(
        f"Elapsed seconds: " f"{elapsed:.2f}",
        flush=True,
    )

    print(
        f"Elapsed minutes: " f"{elapsed / 60:.2f}",
        flush=True,
    )

    print(
        f"Saved replications: " f"{replication_file}",
        flush=True,
    )

    print(
        f"Saved summary: " f"{summary_file}",
        flush=True,
    )

    print(
        f"Saved diagnostics: " f"{diagnostics_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
