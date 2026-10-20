"""Result summaries and experiment output files."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_results(
    results,
):
    valid = results.loc[results["status"] == "ok"].copy()

    def metric_summary(
        column,
    ):
        values = pd.to_numeric(
            valid[column],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            return (
                np.nan,
                np.nan,
            )

        mean = float(values.mean())

        sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan

        return (
            mean,
            sd,
        )

    (
        mean_ibs,
        sd_ibs,
    ) = metric_summary("integrated_brier")

    (
        mean_cindex,
        sd_cindex,
    ) = metric_summary("uno_c_index")

    (
        mean_auc,
        sd_auc,
    ) = metric_summary("cure_auc")

    (
        mean_cure,
        sd_cure,
    ) = metric_summary("mean_predicted_cure")

    runtime = pd.to_numeric(
        valid["runtime_seconds"],
        errors="coerce",
    ).dropna()

    mean_runtime = float(runtime.mean()) if len(runtime) else np.nan

    maximum_runtime = float(runtime.max()) if len(runtime) else np.nan

    return {
        "model_name": (results["model_name"].iloc[0]),
        "kernel": (results["kernel"].iloc[0]),
        "splits": int(len(results)),
        "successful": int((results["status"] == "ok").sum()),
        "failed": int((results["status"] == "failed").sum()),
        "converged": int(results["converged"].fillna(False).astype(bool).sum()),
        "mean_integrated_brier": (mean_ibs),
        "split_sd_integrated_brier": (sd_ibs),
        "mean_uno_c_index": (mean_cindex),
        "split_sd_uno_c_index": (sd_cindex),
        "mean_cure_auc": (mean_auc),
        "split_sd_cure_auc": (sd_auc),
        "mean_predicted_cure": (mean_cure),
        "split_sd_mean_predicted_cure": (sd_cure),
        "mean_runtime_seconds": (mean_runtime),
        "maximum_runtime_seconds": (maximum_runtime),
    }


def save_clinical_results(
    results,
    task_id,
    model_name,
    kernel,
    *,
    output_dir,
    experiment_prefix,
    summarize_results,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = experiment_prefix(
        task_id,
        model_name,
        kernel,
    )

    split_file = output_dir / f"splits_{prefix}.csv"

    summary_file = output_dir / f"summary_{prefix}.csv"

    results.to_csv(
        split_file,
        index=False,
    )

    summary = summarize_results(results)

    pd.DataFrame([summary]).to_csv(
        summary_file,
        index=False,
    )

    return (
        split_file,
        summary_file,
        summary,
    )


def save_pcm_results(
    results,
    diagnostics,
    task_id,
    model_name,
    n,
    method,
    generating_name,
    kernel,
    *,
    output_dir,
    summarize_monte_carlo,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_directory = output_dir / "diagnostics"

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

    replication_file = output_dir / f"replications_{prefix}.csv"

    summary_file = output_dir / f"summary_{prefix}.csv"

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
