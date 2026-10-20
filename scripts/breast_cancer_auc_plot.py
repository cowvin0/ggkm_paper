"""Recreate breast-cancer survival, IBS, cure-AUC, and GG-KM kernel figures.

Use the repeated-holdout CSVs and original prediction NPZs written by
``run_breast_cancer_experiment.py``. The survival panel preserves recorded
``Tempo`` units and uses a common supported horizon. Legacy CSV-only results
can be reconstructed explicitly with ``--refit-missing`` using saved parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import melanoma_auc_plot as shared
from scripts.breast_cancer_plot_utils import (
    DATA_PATH, KERNEL_LABELS, KERNELS, OUTPUT_DIR, PROJECT_ROOT, RESULTS_DIR,
    choose_best_kernel, construct_outer_splits, full_cohort_times,
    load_breast_cancer, read_split_file, unique_result_file, verify_split_hashes,
)
from scripts.breast_cancer_refit_utils import obtain_split_predictions

OUTPUT_BASE = OUTPUT_DIR / "breast_cancer_predictive_assessment_ibs_auc"
KERNEL_OUTPUT_BASE = OUTPUT_DIR / "breast_cancer_ggkm_kernel_comparison_ibs_auc"


def create_model_specs(results_dir=RESULTS_DIR, *, kernels=None):
    """Keep the same six families as melanoma; choose GG-KM kernels by cure AUC."""
    kernels = KERNELS if kernels is None else kernels
    selected = {
        model: choose_best_kernel(model, results_dir, kernels=kernels)
        for model in ("poisson", "bn")
    }
    colors = shared.plt.rcParams["axes.prop_cycle"].by_key()["color"]
    specs = []
    baselines = (
        ("mixture_cure", "Mixture cure", "Mixture\ncure", "-", None),
        ("promotion_time", "Promotion time", "Promotion\ntime", "--", None),
        ("spline_poisson", "Spline, Poisson", "Spline\nP", "-", None),
        ("spline_bn", "Spline, negative binomial", "Spline\nNB", "--", "//"),
    )
    for index, (model, name, label, linestyle, hatch) in enumerate(baselines):
        specs.append({
            "model_name": model, "name": name, "box_label": label, "kernel": None,
            "path": unique_result_file(f"splits_breast_cancer_task*_{model}.csv", results_dir),
            "linestyle": linestyle, "color": colors[index], "hatch": hatch,
        })
    for index, (model, distribution, short, linestyle, hatch) in enumerate((
        ("poisson", "Poisson", "P", "-", None),
        ("bn", "negative binomial", "NB", "--", "//"),
    ), start=4):
        kernel = selected[model]["kernel"]
        specs.append({
            "model_name": model, "name": f"GG-KM, {distribution} ({KERNEL_LABELS[kernel]})",
            "box_label": f"GG-KM {short}\n{KERNEL_LABELS[kernel]}", "kernel": kernel,
            "path": selected[model]["path"], "linestyle": linestyle,
            "color": colors[index], "hatch": hatch,
        })
    return specs


def reconstruct_results(
    df, specs, *, results_dir=RESULTS_DIR, project_root=PROJECT_ROOT,
    outer_splits=None, refit_missing=False,
):
    if outer_splits is None:
        outer_splits = construct_outer_splits(df)
    return shared.reconstruct_results(
        df, specs, results_dir=results_dir, project_root=project_root,
        outer_splits=outer_splits, refit_missing=refit_missing,
        prediction_loader=obtain_split_predictions, split_verifier=verify_split_hashes,
    )


def collect_kernel_score_data(model_name, results_dir=RESULTS_DIR, *, kernels=None):
    ibs_values, auc_values = [], []
    for kernel in KERNELS if kernels is None else kernels:
        path = unique_result_file(
            f"splits_breast_cancer_task*_{model_name}_kernel_{kernel}.csv", results_dir,
        )
        frame = read_split_file(path)
        ibs = frame.loc[frame["valid_ibs"], "integrated_brier"].to_numpy(dtype=float)
        auc = frame.loc[frame["valid_auc"], "cure_auc"].to_numpy(dtype=float)
        if not len(ibs) or not len(auc):
            raise RuntimeError(f"No finite successful IBS/cure-AUC values for {model_name}-{kernel}.")
        ibs_values.append(ibs)
        auc_values.append(auc)
    return ibs_values, auc_values


def make_kernel_comparison_figure(
    results_dir=RESULTS_DIR, output_base=KERNEL_OUTPUT_BASE, *, kernels=None,
):
    kernels = KERNELS if kernels is None else kernels

    def collect(model_name, selected_dir):
        return collect_kernel_score_data(model_name, selected_dir, kernels=kernels)

    return shared.make_kernel_comparison_figure(
        results_dir, output_base, score_collector=collect,
        kernels=kernels, kernel_labels=KERNEL_LABELS,
    )


def make_figure(df, specs, reconstructed, output_base=OUTPUT_BASE):
    return shared.make_figure(
        df, specs, reconstructed, output_base,
        cohort_times=full_cohort_times, time_label="Follow-up time (recorded units)",
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--kernels", nargs="+", choices=KERNELS, default=None,
        help="Explicit subset for older or partial runs; default requires all eight current kernels.",
    )
    parser.add_argument(
        "--refit-missing", action="store_true",
        help="Refit saved best parameters only when a prediction NPZ is missing; never retune.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    kernels = KERNELS if args.kernels is None else list(dict.fromkeys(args.kernels))
    df = load_breast_cancer(args.data_path)
    specs = create_model_specs(args.results_dir, kernels=kernels)
    reconstructed = reconstruct_results(
        df, specs, results_dir=args.results_dir, refit_missing=args.refit_missing,
    )
    # Validate the full kernel comparison before producing any output files.
    for model in ("poisson", "bn"):
        collect_kernel_score_data(model, args.results_dir, kernels=kernels)
    print(f"Common plotting horizon: {reconstructed['common_horizon']:.6f} recorded time units")
    for model, summary in reconstructed["summary"].items():
        print(
            f"{model}: {summary['reconstructed']}/{summary['saved_successful']} splits, "
            f"mean IBS={summary['mean_ibs']:.4f}, mean cure AUC={summary['mean_auc']:.4f}"
        )
    paths = make_figure(df, specs, reconstructed, args.output_dir / OUTPUT_BASE.name)
    paths += make_kernel_comparison_figure(
        args.results_dir, args.output_dir / KERNEL_OUTPUT_BASE.name, kernels=kernels,
    )
    summary_path = args.output_dir / f"{OUTPUT_BASE.name}_summary.csv"
    pd.DataFrame.from_dict(reconstructed["summary"], orient="index").rename_axis(
        "model"
    ).reset_index().to_csv(summary_path, index=False)
    for path in (*paths, summary_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
