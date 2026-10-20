"""Plot paired PCM comparisons for linear, quadratic, and periodic cure surfaces.

Read replication CSVs written by ``run_pcm_experiment.py`` from ``data/results``
and save the figure, paired summary, and input-file audit to ``figures/review``.
Paths are resolved from this repository, independently of the working directory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "review"
OUTPUT_STEM = OUTPUT_DIR / "simulation_rbf_paired_differences_with_auc"
PAIRED_OUTPUT = OUTPUT_DIR / "simulation_rbf_paired_summary_with_auc.csv"
AUDIT_OUTPUT = OUTPUT_DIR / "simulation_rbf_paired_file_audit_with_auc.csv"

FIGURE_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "axes.edgecolor": "#26343f",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3,
    "legend.fontsize": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "svg.hashsalt": "ggkm-20260817",
}
INK = "#26343f"
GRID = "#d7dde2"

FILE_PATTERN = re.compile(
    r"^replications_task(?P<task>\d+)_"
    r"(?P<model>.+?)_"
    r"n(?P<n>\d+)_"
    r"method(?P<method>\d+)_"
    r"gen(?P<generating>poisson|nb025|nb100)_"
    r"kernel(?P<kernel>.+)\.csv$"
)

SURFACES = {1: "linear", 2: "quadratic", 3: "periodic"}
SCENARIOS = (
    ("poisson", 300, "P\n300"),
    ("nb025", 300, "NB\n.25\n300"),
    ("nb100", 300, "NB\n1.00\n300"),
    ("poisson", 600, "P\n600"),
    ("nb025", 600, "NB\n.25\n600"),
    ("nb100", 600, "NB\n1.00\n600"),
)

# Every contrast is GG-KM RBF minus comparator. Negative IBS/RMSE differences
# and positive AUC differences favor RBF.
COMPARISONS = (
    {
        "count": "P",
        "focal_model": "poisson",
        "focal_kernel": "rbf",
        "comparator_model": "poisson",
        "comparator_kernel": "linear",
        "method_a": "ggkm-poisson-rbf",
        "method_b": "ggkm-poisson-linear",
        "label": "GG-KM linear, P",
        "color": "#D55E00",
        "marker": "o",
        "filled": True,
    },
    {
        "count": "NB",
        "focal_model": "bn",
        "focal_kernel": "rbf",
        "comparator_model": "bn",
        "comparator_kernel": "linear",
        "method_a": "ggkm-nb-rbf",
        "method_b": "ggkm-nb-linear",
        "label": "GG-KM linear, NB",
        "color": "#D55E00",
        "marker": "o",
        "filled": False,
    },
    {
        "count": "P",
        "focal_model": "poisson",
        "focal_kernel": "rbf",
        "comparator_model": "spline_poisson",
        "comparator_kernel": "parametric",
        "method_a": "ggkm-poisson-rbf",
        "method_b": "spline-poisson",
        "label": "Spline, P",
        "color": "#009E73",
        "marker": "s",
        "filled": True,
    },
    {
        "count": "NB",
        "focal_model": "bn",
        "focal_kernel": "rbf",
        "comparator_model": "spline_bn",
        "comparator_kernel": "parametric",
        "method_a": "ggkm-nb-rbf",
        "method_b": "spline-nb",
        "label": "Spline, NB",
        "color": "#009E73",
        "marker": "s",
        "filled": False,
    },
    {
        "count": "P",
        "focal_model": "poisson",
        "focal_kernel": "rbf",
        "comparator_model": "poisson",
        "comparator_kernel": "polynomial",
        "method_a": "ggkm-poisson-rbf",
        "method_b": "ggkm-poisson-polynomial",
        "label": "GG-KM polynomial, P",
        "color": "#CC79A7",
        "marker": "^",
        "filled": True,
    },
    {
        "count": "NB",
        "focal_model": "bn",
        "focal_kernel": "rbf",
        "comparator_model": "bn",
        "comparator_kernel": "polynomial",
        "method_a": "ggkm-nb-rbf",
        "method_b": "ggkm-nb-polynomial",
        "label": "GG-KM polynomial, NB",
        "color": "#CC79A7",
        "marker": "^",
        "filled": False,
    },
)

REPLICATION_ALIASES = (
    "replication", "replication_id", "replication_index", "replicate", "rep", "split",
)
IBS_ALIASES = ("test_ibs", "integrated_brier", "integrated_brier_score", "ibs")
RMSE_ALIASES = ("cure_rmse", "test_cure_rmse", "rmse_cure")
AUC_ALIASES = ("test_auc", "cure_auc", "auc_cure", "auc")
METRICS = (
    ("integrated_brier", "Mean IPCW IBS difference"),
    ("cure_rmse", "Mean cure RMSE difference"),
    ("cure_auc", "Mean cure AUC difference"),
)


def parse_filename(path: Path):
    match = FILE_PATTERN.match(path.name)
    if match is None:
        return None
    values = match.groupdict()
    return {
        "task": int(values["task"]),
        "model": values["model"],
        "n": int(values["n"]),
        "method": int(values["method"]),
        "generating": values["generating"],
        "kernel": values["kernel"],
        "path": path,
    }


def index_replication_files(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    records = []
    for path in sorted(results_dir.glob("replications_*.csv")):
        parsed = parse_filename(path)
        if parsed is not None:
            records.append(parsed)
    files = pd.DataFrame(records)
    if files.empty:
        raise RuntimeError(
            "No replications_*.csv files found in:\n"
            f"{results_dir.resolve()}\n"
            "Generate or copy the PCM experiment results into this directory first."
        )
    return files


def find_column(frame: pd.DataFrame, aliases):
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def locate_file(files: pd.DataFrame, *, model, kernel, n, method, generating):
    selected = files.loc[
        (files["model"] == model)
        & (files["kernel"] == kernel)
        & (files["n"] == n)
        & (files["method"] == method)
        & (files["generating"] == generating)
    ]
    if len(selected) != 1:
        details = (
            f"model={model}, kernel={kernel}, n={n}, method={method}, "
            f"generating={generating}, matches={len(selected)}"
        )
        if len(selected):
            details += "\n" + selected.to_string(index=False)
        raise RuntimeError("Expected exactly one matching file:\n" + details)
    return Path(selected.iloc[0]["path"])


def read_metric(path, metric):
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        status = frame["status"].astype(str).str.strip().str.lower()
        frame = frame.loc[status == "ok"].copy()

    if metric == "integrated_brier":
        aliases = IBS_ALIASES
    elif metric == "cure_rmse":
        aliases = RMSE_ALIASES
    elif metric == "cure_auc":
        aliases = AUC_ALIASES
    else:
        raise ValueError(f"Unknown metric: {metric}")

    metric_column = find_column(frame, aliases)
    if metric_column is None:
        raise RuntimeError(
            f"\nMetric '{metric}' not found in:\n{path}\n\n"
            f"Available columns:\n{list(frame.columns)}"
        )

    replication_column = find_column(frame, REPLICATION_ALIASES)
    output = pd.DataFrame()
    if replication_column is not None:
        output["replication"] = frame[replication_column].to_numpy()
        explicit_replication = True
    else:
        output["replication"] = frame.index.to_numpy()
        explicit_replication = False
    output["value"] = pd.to_numeric(frame[metric_column], errors="coerce").to_numpy()
    return output, explicit_replication, metric_column


def calculate_paired_difference(focal_path, comparator_path, *, metric):
    focal, focal_has_replication, focal_metric_column = read_metric(focal_path, metric)
    comparator, comparator_has_replication, comparator_metric_column = read_metric(
        comparator_path, metric,
    )
    if focal_has_replication != comparator_has_replication:
        raise RuntimeError(
            "\nReplication IDs are present in only one member of a pair:\n"
            f"  {focal_path.name}\n  {comparator_path.name}"
        )
    paired = focal.merge(
        comparator, on="replication", how="inner",
        suffixes=("_focal", "_comparator"), validate="one_to_one",
    )
    finite = np.isfinite(paired["value_focal"]) & np.isfinite(paired["value_comparator"])
    paired = paired.loc[finite].copy()
    if len(paired) < 2:
        raise RuntimeError(
            "\nFewer than two finite paired replications:\n"
            f"  focal      = {focal_path.name}\n"
            f"  comparator = {comparator_path.name}\n"
            f"  metric     = {metric}"
        )
    difference = paired["value_focal"] - paired["value_comparator"]
    n_pairs = int(len(difference))
    mean_difference = float(difference.mean())
    sd_difference = float(difference.std(ddof=1))
    mcse = sd_difference / np.sqrt(n_pairs)
    return {
        "pairs": n_pairs,
        "mean_difference_a_minus_b": mean_difference,
        "standard_deviation": sd_difference,
        "mcse": float(mcse),
        "ci95_lower": float(mean_difference - 1.96 * mcse),
        "ci95_upper": float(mean_difference + 1.96 * mcse),
        "focal_metric_column": focal_metric_column,
        "comparator_metric_column": comparator_metric_column,
    }


def build_paired_summary(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    audit_rows = []
    for method, surface in SURFACES.items():
        for generating, n, _ in SCENARIOS:
            scenario = f"{surface}-{generating}-n{n}"
            for comparison in COMPARISONS:
                focal_path = locate_file(
                    files, model=comparison["focal_model"],
                    kernel=comparison["focal_kernel"], n=n,
                    method=method, generating=generating,
                )
                comparator_path = locate_file(
                    files, model=comparison["comparator_model"],
                    kernel=comparison["comparator_kernel"], n=n,
                    method=method, generating=generating,
                )
                audit_rows.append({
                    "scenario": scenario,
                    "label": comparison["label"],
                    "focal_file": focal_path.name,
                    "comparator_file": comparator_path.name,
                })
                for metric, _ in METRICS:
                    result = calculate_paired_difference(
                        focal_path, comparator_path, metric=metric,
                    )
                    paired_rows.append({
                        "scenario": scenario,
                        "surface": surface,
                        "generating_law": generating,
                        "sample_size": n,
                        "metric": metric,
                        "method_a": comparison["method_a"],
                        "method_b": comparison["method_b"],
                        "label": comparison["label"],
                        "count_model": comparison["count"],
                        **{key: result[key] for key in (
                            "pairs", "mean_difference_a_minus_b", "standard_deviation",
                            "mcse", "ci95_lower", "ci95_upper",
                        )},
                    })
    return pd.DataFrame(paired_rows), pd.DataFrame(audit_rows).drop_duplicates()


def get_pair(paired_summary: pd.DataFrame, *, scenario, metric, method_a, method_b):
    selected = paired_summary.loc[
        (paired_summary["scenario"] == scenario)
        & (paired_summary["metric"] == metric)
        & (paired_summary["method_a"] == method_a)
        & (paired_summary["method_b"] == method_b)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one paired row for "
            f"{scenario}, {metric}, {method_a}, {method_b}."
        )
    row = selected.iloc[0]
    return float(row["mean_difference_a_minus_b"]), 1.96 * float(row["mcse"])


def style_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    axis.tick_params(color=INK, labelcolor=INK)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)


def create_figure(paired_summary: pd.DataFrame):
    """Create the supplied three-by-three paired-difference figure."""
    mpl.rcParams.update(FIGURE_STYLE)
    figure, axes = plt.subplots(
        3, 3, figsize=(7.15, 7.0), sharex=True, constrained_layout=False,
    )
    x_positions = np.arange(len(SCENARIOS), dtype=float)
    offsets = np.linspace(-0.18, 0.18, len(COMPARISONS))
    for row_index, (metric, ylabel) in enumerate(METRICS):
        for column_index, (method, surface) in enumerate(SURFACES.items()):
            axis = axes[row_index, column_index]
            for offset, comparison in zip(offsets, COMPARISONS, strict=True):
                values, errors = [], []
                for generating, n, _ in SCENARIOS:
                    value, error = get_pair(
                        paired_summary, scenario=f"{surface}-{generating}-n{n}",
                        metric=metric, method_a=comparison["method_a"],
                        method_b=comparison["method_b"],
                    )
                    values.append(value)
                    errors.append(error)
                axis.errorbar(
                    x_positions + offset, values, yerr=errors,
                    fmt=comparison["marker"],
                    linestyle="-" if comparison["filled"] else "--",
                    color=comparison["color"], linewidth=0.75, markersize=4.3,
                    markerfacecolor=comparison["color"] if comparison["filled"] else "white",
                    markeredgecolor=comparison["color"], markeredgewidth=0.9,
                    ecolor=comparison["color"], elinewidth=0.8, capsize=1.8, zorder=3,
                )
            axis.axhline(0.0, color=INK, linewidth=0.8, zorder=1)
            axis.axvline(2.5, color=GRID, linewidth=0.8, zorder=1)
            panel_letter = chr(ord("A") + row_index * 3 + column_index)
            axis.set_title(
                f"{panel_letter}  {surface.capitalize()} cure surface",
                loc="left", fontweight="bold",
            )
            if column_index == 0:
                axis.set_ylabel(ylabel)
            axis.set_xticks(x_positions, [label for _, _, label in SCENARIOS])
            style_axis(axis)

    handles = [
        Line2D(
            [0], [0], color=comparison["color"],
            linestyle="-" if comparison["filled"] else "--",
            marker=comparison["marker"], markersize=5.0,
            markerfacecolor=comparison["color"] if comparison["filled"] else "white",
            markeredgecolor=comparison["color"], markeredgewidth=0.9,
            label=comparison["label"],
        )
        for comparison in COMPARISONS
    ]
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.050),
        frameon=False, ncol=3, columnspacing=1.1, handletextpad=0.45,
    )
    figure.text(
        0.5, 0.077, "Generating count law and sample size",
        ha="center", va="center", color=INK,
    )
    figure.tight_layout(rect=(0.0, 0.105, 1.0, 1.0), h_pad=1.0, w_pad=1.0)
    return figure


def save_figure(figure, output_stem: Path = OUTPUT_STEM) -> tuple[Path, Path, Path]:
    pdf_file = Path(f"{output_stem}.pdf")
    png_file = Path(f"{output_stem}.png")
    svg_file = Path(f"{output_stem}.svg")
    figure.savefig(
        pdf_file, bbox_inches="tight", pad_inches=0.03, facecolor="white",
        metadata={"Creator": "GG-KM paired differences", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        png_file, dpi=400, bbox_inches="tight", pad_inches=0.03, facecolor="white",
    )
    figure.savefig(
        svg_file, bbox_inches="tight", pad_inches=0.03, facecolor="white",
        metadata={"Creator": "GG-KM paired differences", "Date": None},
    )
    return pdf_file, png_file, svg_file


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory containing PCM replication CSVs (default: repository data/results).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Figure and CSV destination (default: repository figures/review).",
    )
    args = parser.parse_args(argv)
    output_stem = args.output_dir / OUTPUT_STEM.name
    paired_output = args.output_dir / PAIRED_OUTPUT.name
    audit_output = args.output_dir / AUDIT_OUTPUT.name

    files = index_replication_files(args.results_dir)
    print(f"Indexed {len(files)} replication files.")
    paired_summary, audit = build_paired_summary(files)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_summary.to_csv(paired_output, index=False)
    audit.to_csv(audit_output, index=False)
    figure = create_figure(paired_summary)
    try:
        pdf_file, png_file, svg_file = save_figure(figure, output_stem)
    finally:
        plt.close(figure)

    print(f"\nPaired summary ({len(paired_summary)} rows):\n  {paired_output.resolve()}")
    print(f"\nFile audit:\n  {audit_output.resolve()}")
    print(f"\nPDF:\n  {pdf_file.resolve()}")
    print(f"\nPNG:\n  {png_file.resolve()}")
    print(f"\nSVG:\n  {svg_file.resolve()}")
    print("\nContrast definition: GG-KM RBF - comparator")
    print("Interpretation: IBS < 0, RMSE < 0, and AUC > 0 favor RBF.")
    print("Error bars: +/- 1.96 paired Monte Carlo SE")
    print("\nPair counts:")
    print(paired_summary.groupby(["metric", "label"])["pairs"].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
