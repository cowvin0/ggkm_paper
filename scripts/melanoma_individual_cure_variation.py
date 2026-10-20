"""Recreate the melanoma per-patient cure-prediction violin figure.

The summaries use only a patient's appearances in outer test sets. Panel A
contains each patient's mean prediction; panel B contains the sample standard
deviation of those predictions (at least two finite appearances are required).
Saved split predictions are used by default. ``--refit-missing`` explicitly
allows refitting legacy results that contain CSV files without prediction NPZs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.melanoma_plot_utils import (
    DATA_PATH,
    KERNELS,
    RESULTS_DIR,
    construct_outer_splits,
    load_melanoma,
    read_split_file,
    unique_result_file,
)
from scripts.melanoma_refit_utils import obtain_split_predictions


OUTPUT_DIR = PROJECT_ROOT / "figures" / "review"
OUTPUT_STEM = "melanoma_individual_cure_variation"

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#26343f",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 9.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "svg.hashsalt": "melanoma-individual-cure-variation",
}


def create_model_specs(
    results_dir=RESULTS_DIR, ggkm_kernel="rbf", *, dataset_name="melanoma",
    kernels=KERNELS,
):
    """Resolve the reference figure's linear, spline, polynomial, and RBF pairs.

    RBF is the GG-KM predictor in the supplementary individual-prediction
    comparison. ``ggkm_kernel`` allows an explicit alternative when desired.
    """
    if ggkm_kernel not in kernels:
        raise ValueError(f"Unknown GG-KM kernel: {ggkm_kernel}")
    results_dir = Path(results_dir)
    specifications = []
    families = (
        ("Log-lin", "linear", "#FF5500"),
        ("Spline", None, "#00AF83"),
        ("Poly.", "polynomial", "#E7298A"),
        ("GG-KM", ggkm_kernel, "#0072B2"),
    )
    for family, kernel, color in families:
        for model, law in (("poisson", "P"), ("bn", "NB")):
            model_name = f"spline_{model}" if family == "Spline" else model
            selected_kernel = kernel
            suffix = "" if kernel is None else f"_kernel_{kernel}"
            path = unique_result_file(
                f"splits_{dataset_name}_task*_{model_name}{suffix}.csv",
                results_dir=results_dir,
            )
            specifications.append(
                {
                    "name": f"{family} {law}",
                    "label": f"{family}\n{law}",
                    "model_name": model_name,
                    "kernel": selected_kernel,
                    "path": path,
                    "color": color,
                    "hatch": "//" if law == "NB" else None,
                }
            )
    return specifications


def collect_individual_predictions(
    specifications,
    *,
    df,
    results_dir=RESULTS_DIR,
    project_root=PROJECT_ROOT,
    outer_splits,
    refit_missing=False,
    prediction_loader=None,
    cohort_name="melanoma",
):
    """Collect held-out predictions, preserving original cohort row indices."""
    if prediction_loader is None:
        prediction_loader = obtain_split_predictions
    frames = []
    for specification in specifications:
        result = read_split_file(specification["path"])
        successful = result.loc[result["successful"]]
        if successful.empty:
            raise ValueError(
                f"No successful outer splits for {specification['name']}: "
                f"{specification['path']}"
            )
        for _, row in successful.iterrows():
            archive = prediction_loader(
                row,
                df=df,
                model_name=specification["model_name"],
                kernel=specification["kernel"],
                outer_splits=outer_splits,
                results_dir=Path(results_dir),
                project_root=Path(project_root),
                refit_missing=refit_missing,
            )
            patient_indices = np.asarray(archive["test_indices"], dtype=np.int64)
            cure = np.asarray(archive["cure_probability_test"], dtype=float)
            if patient_indices.ndim != 1 or cure.ndim != 1:
                raise ValueError("Test indices and cure predictions must be one-dimensional.")
            if len(patient_indices) != len(cure):
                raise ValueError("Test indices and cure predictions have different lengths.")
            if len(np.unique(patient_indices)) != len(patient_indices):
                raise ValueError("A patient appears more than once in one outer test set.")
            if np.any((patient_indices < 0) | (patient_indices >= len(df))):
                raise ValueError(f"Saved patient indices are outside the {cohort_name} cohort.")
            finite = np.isfinite(cure)
            if np.any((cure[finite] < 0.0) | (cure[finite] > 1.0)):
                raise ValueError("Saved cure probabilities must lie between zero and one.")
            frames.append(
                pd.DataFrame(
                    {
                        "method": specification["name"],
                        "model_name": specification["model_name"],
                        "kernel": specification["kernel"] or "parametric",
                        "split": int(row["split"]),
                        "patient_index": patient_indices,
                        "cure_probability": cure,
                    }
                )
            )
    if not frames:
        raise ValueError("No held-out cure predictions were found.")
    return pd.concat(frames, ignore_index=True)


def summarize_individual_predictions(predictions, *, n_patients=None):
    """Summarize finite predictions by method and original patient index.

    ``sd_predicted_cure`` uses ``ddof=1`` and remains missing for a patient with
    fewer than two finite test-set predictions. Patients never evaluated remain
    missing rather than being assigned a cure probability or deviation of zero.
    """
    required = {"method", "patient_index", "split", "cure_probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Prediction records are missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("No prediction records were supplied.")
    duplicate = predictions.duplicated(["method", "patient_index", "split"])
    if duplicate.any():
        raise ValueError("Duplicate patient predictions within the same method and split.")

    records = predictions.copy()
    records["cure_probability"] = pd.to_numeric(
        records["cure_probability"], errors="raise"
    )
    records.loc[
        ~np.isfinite(records["cure_probability"]), "cure_probability"
    ] = np.nan

    frames = []
    for method, group in records.groupby("method", sort=False):
        summary = group.groupby("patient_index")["cure_probability"].agg(
            n_test_appearances="count",
            mean_predicted_cure="mean",
            sd_predicted_cure="std",
        )
        if n_patients is not None:
            if int(n_patients) != n_patients or n_patients < 1:
                raise ValueError("n_patients must be a positive integer.")
            indices = np.asarray(group["patient_index"])
            if np.any((indices < 0) | (indices >= n_patients)):
                raise ValueError("Patient indices are outside the requested cohort.")
            summary = summary.reindex(pd.RangeIndex(int(n_patients), name="patient_index"))
        summary["n_test_appearances"] = summary["n_test_appearances"].fillna(0).astype(int)
        summary.insert(0, "method", method)
        for column in ("model_name", "kernel"):
            if column in group.columns:
                values = group[column].drop_duplicates()
                if len(values) != 1:
                    raise ValueError(f"Multiple {column} values found for {method}.")
                summary.insert(len(summary.columns), column, values.iloc[0])
        frames.append(summary.reset_index())
    return pd.concat(frames, ignore_index=True)


def plot_individual_cure_variation(
    statistics, specifications, *, output_dir=OUTPUT_DIR,
    title="Melanoma", output_stem=OUTPUT_STEM,
):
    """Write the two-panel violin plot as PDF, PNG, and SVG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
        for axis, column in zip(
            axes, ("mean_predicted_cure", "sd_predicted_cure"), strict=True
        ):
            for position, specification in enumerate(specifications, start=1):
                values = statistics.loc[
                    statistics["method"].eq(specification["name"]), column
                ].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue
                if np.all(values == values[0]):
                    # A singleton or constant distribution has no density to
                    # estimate. Show its value without adding artificial spread.
                    axis.hlines(
                        values[0], position - 0.2, position + 0.2,
                        color="#26343f", linewidth=1.3,
                    )
                    continue
                violin = axis.violinplot(
                    [values],
                    positions=[position],
                    widths=0.8,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                    points=100,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(specification["color"])
                    body.set_edgecolor("#26343f")
                    body.set_linewidth(0.9)
                    body.set_alpha(1.0)
                    if specification["hatch"]:
                        body.set_hatch(specification["hatch"])
                violin["cmedians"].set_color("#26343f")
                violin["cmedians"].set_linewidth(1.3)
            axis.set_xticks(
                range(1, len(specifications) + 1),
                [specification["label"] for specification in specifications],
            )
            axis.set_xlim(0.2, len(specifications) + 0.8)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#d7e2e8", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", length=3, pad=5)
            axis.tick_params(axis="y", length=3)

        axes[0].set_title("A  Individual mean predictions", loc="left", weight="bold")
        axes[0].set_ylabel("Mean predicted cure fraction")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_yticks(np.linspace(0.0, 1.0, 6))
        axes[1].set_title("B  Repeated-holdout variation", loc="left", weight="bold")
        axes[1].set_ylabel("Across-split standard deviation")
        axes[1].set_ylim(bottom=0.0)
        fig.suptitle(title, fontsize=14, weight="bold", y=0.985)
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.16, top=0.91, wspace=0.145)
        try:
            for extension in ("pdf", "png", "svg"):
                path = output_dir / f"{output_stem}.{extension}"
                fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
                figure_paths.append(path)
        finally:
            plt.close(fig)
    return figure_paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--ggkm-kernel", choices=KERNELS, default="rbf",
        help="GG-KM predictor in the individual comparison (reference figure: rbf).",
    )
    parser.add_argument(
        "--refit-missing",
        action="store_true",
        help="Refit saved best parameters when legacy CSV results have no prediction NPZ.",
    )
    args = parser.parse_args(argv)

    df = load_melanoma(args.data_path)
    outer_splits = construct_outer_splits(df)
    specifications = create_model_specs(args.results_dir, ggkm_kernel=args.ggkm_kernel)
    predictions = collect_individual_predictions(
        specifications,
        df=df,
        results_dir=args.results_dir,
        outer_splits=outer_splits,
        refit_missing=args.refit_missing,
    )
    statistics = summarize_individual_predictions(predictions, n_patients=len(df))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = args.output_dir / f"{OUTPUT_STEM}_individual_statistics.csv"
    statistics.to_csv(statistics_path, index=False)
    paths = plot_individual_cure_variation(
        statistics, specifications, output_dir=args.output_dir
    )
    for specification in specifications:
        group = statistics.loc[statistics["method"].eq(specification["name"])]
        n_means = int(group["mean_predicted_cure"].notna().sum())
        n_sd = int(group["sd_predicted_cure"].notna().sum())
        print(
            f"{specification['name']} ({specification['kernel'] or 'parametric'}): "
            f"{n_means}/{len(df)} patient means; {n_sd}/{len(df)} sample deviations"
        )
    for path in [*paths, statistics_path]:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
