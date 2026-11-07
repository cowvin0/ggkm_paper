"""Export separate IBS and Uno's C-index figures from saved experiment CSVs.

The original IBS/cure-AUC figures and their method selection remain available.
No patient data, prediction archives, fitting, or tuning are needed here.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import melanoma_auc_plot as clinical_plot
from scripts.table_utils.clinical import ClinicalMethod, ClinicalResults, KERNELS


OUTPUT_DIR = clinical_plot.OUTPUT_DIR
DATASETS = ("melanoma", "breast-cancer", "fosp-melanoma", "simulation")
METRICS = {
    "ibs": {
        "column": "integrated_brier",
        "ylabel": "IPCW integrated Brier score",
        "title": "Paired repeated-holdout prediction error",
    },
    "uno_c_index": {
        "column": "uno_c_index",
        "ylabel": "Uno's C-index",
        "title": "Paired repeated-holdout survival discrimination",
    },
}


def _score_axis_options(metric, arrays):
    if metric == "ibs":
        return {}
    lower = max(0.0, min(0.48, float(np.min(np.concatenate(arrays))) - 0.03))
    return {"reference": 0.5, "ylim": (lower, 1.0)}


def collect_clinical_scores(results, specs):
    """Validate complete paired holdouts and retain finite successful scores."""
    results.validate()
    scores = {}
    for metric, settings in METRICS.items():
        column = settings["column"]
        scores[metric] = {}
        for method in results.methods:
            frame = results.frame(method)
            finite = frame.loc[frame["status"].eq("ok") & np.isfinite(frame[column])]
            if finite.empty:
                raise ValueError(f"No finite successful {column} values for {method.model}/{method.kernel}.")
            scores[metric][method] = finite.set_index("split")[column]
    records = []
    for metric, method_scores in scores.items():
        for view, methods in (
            ("models", [ClinicalMethod(spec["model_name"], spec["kernel"] or "parametric") for spec in specs]),
            ("kernels", [ClinicalMethod(model, kernel) for kernel in KERNELS for model in ("poisson", "bn")]),
        ):
            for method in methods:
                for split, value in method_scores[method].items():
                    records.append({
                        "dataset": results.dataset, "comparison": view,
                        "model_name": method.model, "kernel": method.kernel,
                        "metric": metric, "split": split, "value": value,
                    })
    return scores, pd.DataFrame(records)


def generate_clinical_score_plots(dataset, *, results_dir=None, output_dir=OUTPUT_DIR):
    """Export model and seven-kernel comparisons, one metric per file."""
    if dataset not in DATASETS[:3]:
        raise ValueError(f"Unknown clinical dataset: {dataset!r}.")
    prefix = dataset.replace("-", "_")
    module = importlib.import_module(f"scripts.{prefix}_auc_plot")
    directory = Path(results_dir) if results_dir is not None else module.RESULTS_DIR
    results = ClinicalResults(directory, dataset)
    # Keep the same six families and the same GG-KM kernels as the existing figure.
    specs = module.create_model_specs(directory)
    scores, values = collect_clinical_scores(results, specs)
    output_dir = Path(output_dir)
    paths = []
    for metric, settings in METRICS.items():
        method_scores = scores[metric]
        arrays = [
            method_scores[ClinicalMethod(spec["model_name"], spec["kernel"] or "parametric")].to_numpy()
            for spec in specs
        ]
        fig, axis = clinical_plot.plt.subplots(figsize=(8.0, 5.4))
        clinical_plot.draw_score_panel(
            axis, arrays, specs, title=settings["title"], ylabel=settings["ylabel"],
            **_score_axis_options(metric, arrays),
        )
        fig.subplots_adjust(bottom=0.20, top=0.92, left=0.12, right=0.98)
        paths.extend(clinical_plot.save_figure(
            fig, output_dir / f"{prefix}_predictive_assessment_{metric}",
        ))

        poisson = [method_scores[ClinicalMethod("poisson", kernel)].to_numpy() for kernel in KERNELS]
        nb = [method_scores[ClinicalMethod("bn", kernel)].to_numpy() for kernel in KERNELS]
        fig, axis = clinical_plot.plt.subplots(figsize=(9.0, 5.4))
        colors = clinical_plot.draw_grouped_kernel_panel(
            axis, poisson, nb, title="GG-KM kernel comparison", ylabel=settings["ylabel"],
            kernels=KERNELS, kernel_labels=clinical_plot.KERNEL_LABELS,
            **_score_axis_options(metric, poisson + nb),
        )
        fig.legend(handles=[
            clinical_plot.Patch(facecolor=colors[0], edgecolor="0.25", label="Poisson", alpha=0.78),
            clinical_plot.Patch(facecolor=colors[1], edgecolor="0.25", hatch="//", label="Negative binomial", alpha=0.78),
        ], loc="lower center", ncol=2, frameon=False, fontsize=9.5)
        fig.subplots_adjust(bottom=0.26, top=0.92, left=0.12, right=0.98)
        paths.extend(clinical_plot.save_figure(
            fig, output_dir / f"{prefix}_ggkm_kernel_comparison_{metric}",
        ))
    values_path = output_dir / f"{prefix}_standalone_metric_values.csv"
    values.to_csv(values_path, index=False)
    return (*paths, values_path)


def generate_simulation_score_plots(*, results_dir=None, output_dir=OUTPUT_DIR):
    """Export primary and higher-dimensional paired contrasts for each metric."""
    from scripts import create_paired_plot_1_2_3_methods as primary
    from scripts import create_paired_plot_interaction_quadratic_surfaces as highdim

    directory = Path(results_dir) if results_dir is not None else primary.RESULTS_DIR
    primary_metrics = (
        ("integrated_brier", "Mean IPCW IBS difference"),
        ("uno_c_index", "Mean Uno's C-index difference"),
    )
    summary, audit = primary.build_paired_summary(
        primary.index_replication_files(directory), metrics=primary_metrics,
    )
    highdim_data = highdim.calculate_plot_data(
        highdim.discover_files(directory), metrics=("ibs", "cindex"),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric_spec, suffix, highdim_metric in zip(
        primary_metrics, ("ibs", "uno_c_index"), ("ibs", "cindex"), strict=True,
    ):
        metric, _ = metric_spec
        metric_summary = summary.loc[summary["metric"].eq(metric)]
        stem = output_dir / f"simulation_rbf_paired_differences_{suffix}"
        figure = primary.create_figure(metric_summary, metrics=(metric_spec,))
        try:
            paths.extend(primary.save_figure(figure, stem))
        finally:
            primary.plt.close(figure)
        csv = stem.with_suffix(".csv")
        metric_summary.to_csv(csv, index=False)
        paths.append(csv)

        metric_data = highdim_data.loc[highdim_data["metric"].eq(highdim_metric)]
        stem = output_dir / f"highdim_rbf_vs_linear_spline_polynomial_paired_differences_{suffix}"
        paths.extend(highdim.create_figure(metric_data, stem, metrics=(highdim_metric,)))
        csv = stem.with_suffix(".csv")
        metric_data.to_csv(csv, index=False)
        paths.append(csv)
    audit_path = output_dir / "simulation_standalone_metric_file_audit.csv"
    audit.to_csv(audit_path, index=False)
    return (*paths, audit_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        if args.dataset == "simulation":
            paths = generate_simulation_score_plots(results_dir=args.results_dir, output_dir=args.output_dir)
        else:
            paths = generate_clinical_score_plots(
                args.dataset, results_dir=args.results_dir, output_dir=args.output_dir,
            )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
