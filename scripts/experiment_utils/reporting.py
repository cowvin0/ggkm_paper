"""Console reports shared by the experiment command-line entry points."""

from __future__ import annotations


def print_clinical_header(task_id, model_name, kernel):
    print("==================================", flush=True)
    print(f"Task {task_id}", flush=True)
    if kernel is None:
        print(f"Model={model_name}", flush=True)
    else:
        print(f"Model={model_name}, kernel={kernel}", flush=True)
    print("", flush=True)


def print_clinical_design(
    *,
    dataset_name,
    outer_splits,
    outer_test_size,
    inner_splits,
    worker_count,
    inner_grid_points,
    outer_grid_points,
    observed_time_quantile,
    minimum_censoring_survival,
    master_seed,
    outer_training_size=None,
):
    """Print design settings, querying workers at the usual reporting step."""
    print(f"{dataset_name} evaluation design:", flush=True)
    print(f"  Outer repeated holdouts: {outer_splits}", flush=True)
    if outer_training_size is None:
        print(f"  Outer test fraction: {outer_test_size:.2f}", flush=True)
    else:
        print(f"  Outer training size: {outer_training_size}", flush=True)
        print(f"  Outer test size: {outer_test_size}", flush=True)
    print(f"  Inner folds: {inner_splits}", flush=True)
    print(f"  Outer workers: {worker_count()}", flush=True)
    print(f"  Inner grid points: {inner_grid_points}", flush=True)
    print(f"  Outer grid points: {outer_grid_points}", flush=True)
    print(f"  Time quantile: {observed_time_quantile}", flush=True)
    print(f"  Minimum censoring survival: {minimum_censoring_survival}", flush=True)
    print(f"  Master seed: {master_seed}", flush=True)
    print("", flush=True)


def print_clinical_tuning(
    *,
    model_name,
    kernel,
    kernel_models,
    spline_models,
    kernel_parameter_grid,
    spline_parameter_grid,
    show_spline_candidate_count,
):
    if model_name in kernel_models:
        print("Hyperparameter tuning: exhaustive grid search", flush=True)
        candidates = kernel_parameter_grid(model_name, kernel)
        print(f"  Candidates per outer split: {len(candidates)}", flush=True)
        for candidate in candidates:
            print(f"  {candidate}", flush=True)
    elif model_name in spline_models:
        print("Hyperparameter tuning: exhaustive spline grid", flush=True)
        if show_spline_candidate_count:
            print(
                f"  Candidates per outer split: {len(spline_parameter_grid)}",
                flush=True,
            )
        for candidate in spline_parameter_grid:
            print(f"  {candidate}", flush=True)
    else:
        print("Hyperparameter tuning: none", flush=True)
    print("==================================", flush=True)


def print_dataset_counts(df):
    print(f"Number of observations: {len(df)}", flush=True)
    print(f"Number of events: {int(df['delta'].sum())}", flush=True)
    print(f"Number censored: {int((1 - df['delta']).sum())}", flush=True)


def print_clinical_completion(summary, elapsed, split_file, summary_file):
    print("", flush=True)
    print("==================================", flush=True)
    print("Finished", flush=True)
    print(f"Splits: {summary['splits']}", flush=True)
    print(f"Successful: {summary['successful']}", flush=True)
    print(f"Failed: {summary['failed']}", flush=True)
    print(f"Converged: {summary['converged']}/{summary['splits']}", flush=True)
    print(
        f"IPCW IBS: {summary['mean_integrated_brier']:.4f} "
        f"({summary['split_sd_integrated_brier']:.4f})",
        flush=True,
    )
    print(
        f"Mean predicted cure: {summary['mean_predicted_cure']:.4f} "
        f"({summary['split_sd_mean_predicted_cure']:.4f})",
        flush=True,
    )
    print(
        f"Runtime per split: {summary['mean_runtime_seconds']:.2f} "
        f"({summary['maximum_runtime_seconds']:.2f}) seconds",
        flush=True,
    )
    print(f"Total elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Total elapsed minutes: {elapsed / 60:.2f}", flush=True)
    print(f"Total elapsed hours: {elapsed / 3600:.2f}", flush=True)
    print(f"Saved split results: {split_file}", flush=True)
    print(f"Saved summary: {summary_file}", flush=True)
    print("==================================", flush=True)


def print_pcm_start(
    *,
    task_id,
    model_name,
    n,
    method,
    generating_name,
    kernel,
    parameter_grid,
):
    print(f"Task {task_id} started", flush=True)
    print(
        f"Model={model_name} n={n} method={method} "
        f"generator={generating_name} kernel={kernel}",
        flush=True,
    )
    if kernel is not None:
        print(f"Parameter grid: {parameter_grid(model_name, kernel)}", flush=True)


def print_pcm_completion(
    *,
    results,
    task_id,
    elapsed,
    replication_file,
    summary_file,
    diagnostics_file,
):
    successful = int((results["status"] == "ok").sum())
    failed = len(results) - successful
    print(f"Task {task_id} finished", flush=True)
    print(f"Replications: {len(results)}", flush=True)
    print(f"Successful: {successful}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Elapsed minutes: {elapsed / 60:.2f}", flush=True)
    print(f"Saved replications: {replication_file}", flush=True)
    print(f"Saved summary: {summary_file}", flush=True)
    print(f"Saved diagnostics: {diagnostics_file}", flush=True)
