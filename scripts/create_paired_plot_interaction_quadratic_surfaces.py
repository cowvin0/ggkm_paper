"""Plot paired RBF contrasts for the PCM interaction and radial scenarios.

Run this script after the method 4 and 5 PCM experiments have saved their
replication CSVs in ``data/results``. Figures and the plotted values are written
to ``figures/review`` relative to the repository, regardless of the working
directory. The scenario labels follow the supplied figure specification.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "review"
OUTPUT_STEM = (
    OUTPUT_DIR
    / "highdim_rbf_vs_linear_spline_polynomial_paired_differences_with_auc"
)
DATA_OUTPUT = OUTPUT_STEM.with_suffix(".csv")

SURFACES = {4: "Interaction", 5: "Radial"}
GENERATING_LAWS = [
    ("poisson", "P", 300),
    ("nb025", "NB\n.25", 300),
    ("nb100", "NB\n1.00", 300),
    ("poisson", "P", 600),
    ("nb025", "NB\n.25", 600),
    ("nb100", "NB\n1.00", 600),
]
COMPARATOR_FAMILIES = ["GG-KM linear", "Spline", "GG-KM polynomial"]
COUNT_LAWS = ["P", "NB"]
COLORS = {
    "GG-KM linear": "#E65100",
    "Spline": "#00A976",
    "GG-KM polynomial": "#DD62A4",
}
MARKERS = {"GG-KM linear": "o", "Spline": "s", "GG-KM polynomial": "^"}

FILE_PATTERN = re.compile(
    r"^replications_task(?P<task>\d+)_"
    r"(?P<model>.+?)_"
    r"n(?P<n>\d+)_"
    r"method(?P<method>\d+)_"
    r"gen(?P<generating>poisson|nb025|nb100)_"
    r"kernel(?P<kernel>.+)\.csv$"
)
REPLICATION_ALIASES = [
    "replication", "replication_id", "replication_index", "replicate", "rep",
    "replication_seed", "seed",
]
IBS_ALIASES = [
    "test_ibs", "integrated_brier", "integrated_brier_score", "ibs",
    "IPCW integrated Brier score",
]
RMSE_ALIASES = ["cure_rmse", "test_cure_rmse", "rmse_cure", "Cure RMSE"]
AUC_ALIASES = ["test_auc", "cure_auc", "auc_cure", "auc", "Cure AUC"]

GRID_COLOR = "#D7E0E5"
ZERO_COLOR = "#90999E"
SPINE_COLOR = "#394B55"
TEXT_COLOR = "#25343D"


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
    }


def discover_files(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    records = []
    for path in sorted(results_dir.glob("replications_*.csv")):
        metadata = parse_filename(path)
        if metadata is not None and metadata["method"] in SURFACES:
            records.append({**metadata, "path": path})
    files = pd.DataFrame(records)
    if files.empty:
        raise RuntimeError(
            "No high-dimensional replications_*.csv files found in:\n"
            f"{results_dir.resolve()}\n"
            "Run scripts/run_pcm_experiment.py for methods 4 and 5 first."
        )
    return files


def find_column(df: pd.DataFrame, aliases: list[str]):
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        key = alias.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def locate_file(
    files: pd.DataFrame,
    *,
    model: str,
    kernel: str | None,
    n: int,
    surface_id: int,
    generating: str,
):
    mask = (
        (files["model"] == model)
        & (files["n"] == n)
        & (files["method"] == surface_id)
        & (files["generating"] == generating)
    )
    if kernel is not None:
        mask &= files["kernel"] == kernel
    matches = files.loc[mask].copy()
    if len(matches) != 1:
        details = (
            f"model={model}, kernel={kernel}, n={n}, surface={surface_id}, "
            f"generating={generating}"
        )
        raise RuntimeError(
            f"Expected exactly one matching replication file for {details}; "
            f"found {len(matches)}.\n{matches.to_string(index=False)}"
        )
    return Path(matches.iloc[0]["path"])


def locate_spline_file(
    files: pd.DataFrame,
    *,
    count_law: str,
    n: int,
    surface_id: int,
    generating: str,
):
    candidate_models = (
        ["spline_poisson"] if count_law == "P"
        else ["spline_bn", "spline_negative_binomial"]
    )
    found = []
    for model in candidate_models:
        mask = (
            (files["model"] == model)
            & (files["n"] == n)
            & (files["method"] == surface_id)
            & (files["generating"] == generating)
        )
        matches = files.loc[mask]
        if len(matches) == 1:
            found.append(Path(matches.iloc[0]["path"]))
    if len(found) != 1:
        raise RuntimeError(
            f"Could not uniquely locate spline {count_law} file for "
            f"method={surface_id}, generating={generating}, n={n}. "
            f"Found {len(found)}."
        )
    return found[0]


def read_metric(path: Path, metric: str):
    df = pd.read_csv(path)
    aliases = {"ibs": IBS_ALIASES, "rmse": RMSE_ALIASES, "auc": AUC_ALIASES}
    if metric not in aliases:
        raise ValueError(f"Unknown metric: {metric}")
    metric_column = find_column(df, aliases[metric])
    if metric_column is None:
        raise RuntimeError(
            f"Could not find metric '{metric}' in:\n{path}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )
    replication_column = find_column(df, REPLICATION_ALIASES)
    out = pd.DataFrame()
    out["row_index"] = np.arange(len(df), dtype=int)
    if replication_column is not None:
        out["replication_key"] = df[replication_column]
        explicit_key = True
    else:
        out["replication_key"] = out["row_index"]
        explicit_key = False
    out["value"] = pd.to_numeric(df[metric_column], errors="coerce")
    return out, explicit_key


def paired_difference(focal_path: Path, comparator_path: Path, *, metric: str):
    """Return focal RBF minus comparator and its paired Monte Carlo interval."""
    focal, focal_has_id = read_metric(focal_path, metric)
    comparator, comparator_has_id = read_metric(comparator_path, metric)
    key = "replication_key" if focal_has_id and comparator_has_id else "row_index"
    focal = focal[[key, "value"]].rename(columns={"value": "focal_value"})
    comparator = comparator[[key, "value"]].rename(
        columns={"value": "comparator_value"}
    )
    paired = focal.merge(comparator, on=key, how="inner", validate="one_to_one")
    finite = (
        np.isfinite(paired["focal_value"])
        & np.isfinite(paired["comparator_value"])
    )
    paired = paired.loc[finite].copy()
    if len(paired) < 2:
        raise RuntimeError(
            "Not enough finite paired replications for:\n"
            f"  focal:      {focal_path.name}\n"
            f"  comparator: {comparator_path.name}\n"
            f"  metric:     {metric}"
        )
    difference = paired["focal_value"] - paired["comparator_value"]
    mean_difference = float(difference.mean())
    sd_difference = float(difference.std(ddof=1))
    mcse = sd_difference / np.sqrt(len(difference))
    return {
        "mean_difference": mean_difference,
        "mcse": float(mcse),
        "halfwidth_95": float(1.96 * mcse),
        "paired_n": int(len(difference)),
    }


def focal_model(count_law: str):
    return ("poisson", "rbf") if count_law == "P" else ("bn", "rbf")


def comparator_model(family: str, count_law: str):
    model = "poisson" if count_law == "P" else "bn"
    if family == "GG-KM linear":
        return model, "linear"
    if family == "GG-KM polynomial":
        return model, "polynomial"
    raise ValueError(f"Unknown GG-KM comparator family: {family}")


def calculate_plot_data(files: pd.DataFrame) -> pd.DataFrame:
    records = []
    for surface_id, surface_name in SURFACES.items():
        for x_position, (generating, generating_display, n) in enumerate(
            GENERATING_LAWS
        ):
            for count_law in COUNT_LAWS:
                focal_model_name, focal_kernel = focal_model(count_law)
                focal_path = locate_file(
                    files, model=focal_model_name, kernel=focal_kernel, n=n,
                    surface_id=surface_id, generating=generating,
                )
                if "kernelrbf" not in focal_path.name:
                    raise RuntimeError(f"Focal experiment is not RBF:\n{focal_path}")
                for family in COMPARATOR_FAMILIES:
                    if family == "Spline":
                        comparator_path = locate_spline_file(
                            files, count_law=count_law, n=n,
                            surface_id=surface_id, generating=generating,
                        )
                    else:
                        model, kernel = comparator_model(family, count_law)
                        comparator_path = locate_file(
                            files, model=model, kernel=kernel, n=n,
                            surface_id=surface_id, generating=generating,
                        )
                    for metric in ("ibs", "rmse", "auc"):
                        result = paired_difference(
                            focal_path, comparator_path, metric=metric,
                        )
                        records.append({
                            "surface_id": surface_id,
                            "surface": surface_name,
                            "x_position": x_position,
                            "generating": generating,
                            "generating_display": generating_display,
                            "n": n,
                            "count_law": count_law,
                            "comparator": family,
                            "metric": metric,
                            **result,
                            "focal_model": focal_model_name,
                            "focal_kernel": focal_kernel,
                            "focal_file": focal_path.name,
                            "comparator_file": comparator_path.name,
                        })
    plot_data = pd.DataFrame(records)
    if not (plot_data["focal_kernel"] == "rbf").all():
        raise RuntimeError("At least one plotted focal model is not RBF.")
    return plot_data


def print_audit(plot_data: pd.DataFrame) -> None:
    print("\nAUDIT OF MODELS USED IN THE FIGURE")
    audit = plot_data[[
        "surface", "generating", "n", "count_law", "comparator",
        "focal_file", "comparator_file",
    ]].drop_duplicates()
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 300,
    ):
        print(audit.to_string(index=False))
    print("\nAll focal models use kernel='rbf':",
          bool((plot_data["focal_kernel"] == "rbf").all()))
    print("\nPAIR COUNTS BY METRIC")
    print(plot_data.groupby(["metric", "comparator", "count_law"])["paired_n"]
          .agg(["min", "max"]).to_string())


def style_axis(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=TEXT_COLOR, width=0.8, length=3.5)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.85, alpha=1.0, zorder=0)
    ax.axvline(2.5, color=GRID_COLOR, linewidth=1.0, zorder=0)
    ax.axhline(0.0, color=ZERO_COLOR, linewidth=1.0, zorder=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def series_style(family: str, count_law: str):
    color, marker = COLORS[family], MARKERS[family]
    return {
        "color": color,
        "linestyle": "-" if count_law == "P" else "--",
        "marker": marker,
        "markerfacecolor": color if count_law == "P" else "white",
        "markeredgecolor": color,
        "markeredgewidth": 0.9 if count_law == "P" else 1.0,
        "markersize": 5.2,
    }


def draw_panel(ax, plot_data: pd.DataFrame, *, surface_id: int, metric: str):
    panel_data = plot_data.loc[
        (plot_data["surface_id"] == surface_id) & (plot_data["metric"] == metric)
    ].copy()
    if panel_data.empty:
        raise RuntimeError(
            f"No plotting data for surface_id={surface_id}, metric={metric}."
        )
    for family in COMPARATOR_FAMILIES:
        for count_law in COUNT_LAWS:
            series = panel_data.loc[
                (panel_data["comparator"] == family)
                & (panel_data["count_law"] == count_law)
            ].sort_values("x_position")
            if len(series) != 6:
                raise RuntimeError(
                    f"Expected 6 points for {surface_id=}, {metric=}, "
                    f"{family=}, {count_law=}; found {len(series)}."
                )
            ax.errorbar(
                series["x_position"].to_numpy(dtype=float),
                series["mean_difference"].to_numpy(dtype=float),
                yerr=series["halfwidth_95"].to_numpy(dtype=float),
                **series_style(family, count_law),
                linewidth=1.10, elinewidth=1.0, capsize=3.0, capthick=1.0,
                zorder=3,
            )
    lower = panel_data["mean_difference"] - panel_data["halfwidth_95"]
    upper = panel_data["mean_difference"] + panel_data["halfwidth_95"]
    ymin, ymax = min(float(lower.min()), 0.0), max(float(upper.max()), 0.0)
    span = ymax - ymin
    if span <= 0.0:
        span = 1.0
    margin = 0.08 * span
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlim(-0.45, 5.45)


def create_figure(plot_data: pd.DataFrame, output_stem: Path = OUTPUT_STEM):
    """Render all six panels with the supplied colors, layout, and intervals."""
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9.0,
        "axes.titlesize": 11.0, "axes.titleweight": "bold",
        "axes.labelsize": 10.0, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.3, "axes.linewidth": 1.0, "lines.linewidth": 1.15,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    }):
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8.73, 8.10), sharex="col")
        try:
            fig.patch.set_facecolor("white")
            for ax in axes.flat:
                style_axis(ax)
            metrics = ["ibs", "rmse", "auc"]
            ylabels = [
                "Mean IPCW IBS difference", "Mean cure RMSE difference",
                "Mean cure AUC difference",
            ]
            for row, metric in enumerate(metrics):
                for col, (surface_id, surface_name) in enumerate(SURFACES.items()):
                    ax = axes[row, col]
                    draw_panel(ax, plot_data, surface_id=surface_id, metric=metric)
                    letter = chr(ord("A") + row * 2 + col)
                    ax.set_title(f"{letter}  {surface_name} cure surface", loc="left", pad=8)
                axes[row, 0].set_ylabel(ylabels[row])
            x_labels = [
                "P\n300", "NB\n.25\n300", "NB\n1.00\n300",
                "P\n600", "NB\n.25\n600", "NB\n1.00\n600",
            ]
            for ax in axes[2, :]:
                ax.set_xticks(np.arange(6))
                ax.set_xticklabels(x_labels)
            for ax in axes[:2, :].flat:
                ax.tick_params(axis="x", labelbottom=False)
            fig.text(
                0.5, 0.105, "Generating count law and sample size",
                ha="center", va="center", fontsize=9.5, color=TEXT_COLOR,
            )
            legend_handles = []
            # Matplotlib fills column-wise: append P then NB for each family.
            for family in COMPARATOR_FAMILIES:
                for count_law in COUNT_LAWS:
                    style = series_style(family, count_law)
                    legend_handles.append(Line2D(
                        [0], [0], color=style["color"], linestyle=style["linestyle"],
                        linewidth=1.3, marker=style["marker"], markersize=5.5,
                        markerfacecolor=style["markerfacecolor"],
                        markeredgecolor=style["markeredgecolor"],
                        markeredgewidth=1.0, label=f"{family}, {count_law}",
                    ))
            fig.legend(
                handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
                ncol=3, frameon=False, handlelength=2.0, handletextpad=0.5,
                columnspacing=1.7, borderaxespad=0.0,
            )
            fig.subplots_adjust(
                left=0.115, right=0.965, top=0.955, bottom=0.205,
                wspace=0.19, hspace=0.28,
            )
            output_stem.parent.mkdir(parents=True, exist_ok=True)
            paths = [output_stem.with_suffix(f".{suffix}") for suffix in ("png", "pdf", "svg")]
            for path in paths:
                kwargs = {"dpi": 300} if path.suffix == ".png" else {}
                fig.savefig(path, facecolor="white", **kwargs)
        finally:
            plt.close(fig)
    return paths


def main(results_dir: Path = RESULTS_DIR, output_dir: Path = OUTPUT_DIR) -> None:
    files = discover_files(results_dir)
    print(f"Found {len(files)} eligible high-dimensional replication files.")
    plot_data = calculate_plot_data(files)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / OUTPUT_STEM.name
    data_output = output_stem.with_suffix(".csv")
    plot_data.to_csv(data_output, index=False)
    print_audit(plot_data)
    paths = create_figure(plot_data, output_stem)
    print("\nFIGURE CREATED")
    for path in paths:
        print(f"{path.suffix[1:].upper()}: {path.resolve()}")
    print(f"Data: {data_output.resolve()}")
    print("\nDefinition of every plotted contrast: RBF GG-KM - comparator")
    print("Comparators: orange = GG-KM linear; green = spline; magenta = GG-KM polynomial")
    print("IBS and RMSE differences < 0, AUC differences > 0: RBF GG-KM better")
    print("IBS and RMSE differences > 0, AUC differences < 0: comparator better")
    print("Error bars: +/- 1.96 paired Monte Carlo SE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory containing PCM replication CSVs (default: data/results).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for figures and plotted values (default: figures/review).",
    )
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir)
