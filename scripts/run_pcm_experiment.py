"""Simulation settings and PCM experiment entry point."""

import os
import time
from pathlib import Path

from evaluate.cross_validation import (
    monte_carlo_pcm,
    summarize_monte_carlo,
)

from scripts.experiment_utils import (
    models as experiment_models,
    reporting,
    results as experiment_results,
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

MODELS = experiment_models.make_model_registry(negative_binomial_max_em_iter=10000)

KERNEL_MODELS = set(experiment_models.KERNEL_MODEL_NAMES)


N_REPLICATIONS = 250
TEST_SIZE = 0.30
N_INNER_SPLITS = 2
INNER_GRID_POINTS = 24
OUTER_GRID_POINTS = 36
MASTER_SEED = 20260817

BINOMIAL_K = 2
OUTPUT_DIR = Path("data/results")


def parameter_grid(model_name, kernel):
    return experiment_models.pcm_parameter_grid(
        model_name,
        kernel,
        kernel_models=KERNEL_MODELS,
        binomial_k=BINOMIAL_K,
    )


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
    results, diagnostics, task_id, model_name, n, method, generating_name, kernel
):
    return experiment_results.save_pcm_results(
        results,
        diagnostics,
        task_id,
        model_name,
        n,
        method,
        generating_name,
        kernel,
        output_dir=OUTPUT_DIR,
        summarize_monte_carlo=summarize_monte_carlo,
    )


def main():
    start_time = time.time()
    task_id = int(os.environ["PBS_ARRAY_INDEX"])
    experiments = build_experiments()
    exp = get_experiment(experiments, task_id)
    model_name = exp["model_name"]
    n = exp["n"]
    method = exp["method"]
    generating_name = exp["generating_name"]
    generating_distribution = exp["generating_distribution"]
    generating_phi = exp["generating_phi"]
    kernel = exp["kernel"]

    reporting.print_pcm_start(
        task_id=task_id,
        model_name=model_name,
        n=n,
        method=method,
        generating_name=generating_name,
        kernel=kernel,
        parameter_grid=parameter_grid,
    )
    results, diagnostics = run_experiment(
        model_name=model_name,
        n=n,
        method=method,
        generating_distribution=generating_distribution,
        generating_phi=generating_phi,
        kernel=kernel,
    )
    replication_file, summary_file, diagnostics_file = save_results(
        results=results,
        diagnostics=diagnostics,
        task_id=task_id,
        model_name=model_name,
        n=n,
        method=method,
        generating_name=generating_name,
        kernel=kernel,
    )
    elapsed = time.time() - start_time
    reporting.print_pcm_completion(
        results=results,
        task_id=task_id,
        elapsed=elapsed,
        replication_file=replication_file,
        summary_file=summary_file,
        diagnostics_file=diagnostics_file,
    )


if __name__ == "__main__":
    main()
