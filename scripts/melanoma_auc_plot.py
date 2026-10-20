"""Recreate melanoma survival, IBS, cure-AUC, and GG-KM kernel figures.

Read the repeated-holdout CSVs and prediction NPZs saved by
``run_melanoma_experiment.py``. Figures use the supplied reference's model
selection and styling and are saved in ``figures/review``. Saved survival
curves are linearly interpolated onto the reference's 250-point common grid;
the saved IBS and cure-AUC scores are used unchanged. For legacy CSV-only
results, ``--refit-missing`` explicitly enables reconstruction without tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.melanoma_plot_utils import (
    DATA_PATH,
    KERNEL_LABELS,
    KERNELS,
    MASTER_SEED,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    choose_best_kernel,
    construct_outer_splits,
    load_melanoma,
    read_split_file,
    unique_result_file,
    verify_split_hashes,
)
from scripts.melanoma_refit_utils import obtain_split_predictions

N_GRID = 250
OUTPUT_BASE = OUTPUT_DIR / "melanoma_predictive_assessment_ibs_auc"
KERNEL_OUTPUT_BASE = OUTPUT_DIR / "melanoma_ggkm_kernel_comparison_ibs_auc"


def create_model_specs(results_dir=RESULTS_DIR):
    """Keep the reference's six models, selecting GG-KM kernels by cure AUC."""
    best_poisson = choose_best_kernel("poisson", results_dir)
    best_nb = choose_best_kernel("bn", results_dir)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    specs = []
    baselines = (
        ("mixture_cure", "Mixture cure", "Mixture\ncure", "-", None),
        ("promotion_time", "Promotion time", "Promotion\ntime", "--", None),
        ("spline_poisson", "Spline, Poisson", "Spline\nP", "-", None),
        ("spline_bn", "Spline, negative binomial", "Spline\nNB", "--", "//"),
    )
    for index, (model, name, label, linestyle, hatch) in enumerate(baselines):
        specs.append({
            "model_name": model,
            "name": name,
            "box_label": label,
            "kernel": None,
            "path": unique_result_file(f"splits_melanoma_task*_{model}.csv", results_dir),
            "linestyle": linestyle,
            "color": colors[index],
            "hatch": hatch,
        })
    for index, (model, distribution, short, selected, style, hatch) in enumerate((
        ("poisson", "Poisson", "P", best_poisson, "-", None),
        ("bn", "negative binomial", "NB", best_nb, "--", "//"),
    ), start=4):
        kernel = selected["kernel"]
        specs.append({
            "model_name": model,
            "name": f"GG-KM, {distribution} ({KERNEL_LABELS[kernel]})",
            "box_label": f"GG-KM {short}\n{KERNEL_LABELS[kernel]}",
            "kernel": kernel,
            "path": selected["path"],
            "linestyle": style,
            "color": colors[index],
            "hatch": hatch,
        })
    return specs


def full_cohort_times(df):
    """Use the runner's follow-up conversion from days to years."""
    time = df["time"].to_numpy(dtype=float) / 365.25
    event = df["delta"].to_numpy(dtype=int)
    if np.any(~np.isfinite(time)) or np.any(time < 0):
        raise ValueError("Melanoma follow-up times must be finite and nonnegative.")
    return time, event


def reconstruct_results(
    df, specs, *, results_dir=RESULTS_DIR, project_root=PROJECT_ROOT,
    outer_splits=None, refit_missing=False,
    prediction_loader=None, split_verifier=None,
):
    """Average held-out subject predictions within each split, then across splits."""
    if outer_splits is None:
        outer_splits = construct_outer_splits(df)
    if prediction_loader is None:
        prediction_loader = obtain_split_predictions
    if split_verifier is None:
        split_verifier = verify_split_hashes
    frames = {}
    horizons = []
    for spec in specs:
        frame = read_split_file(spec["path"])
        split_verifier(frame, outer_splits, spec["name"])
        successful = frame.loc[frame["successful"]].sort_values("split")
        frames[spec["name"]] = successful
        values = successful["horizon"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite):
            horizons.append(float(finite.min()))
    if not horizons or min(horizons) <= 0:
        raise RuntimeError("No positive finite evaluation horizons were found.")
    common_horizon = min(horizons)
    t_grid = np.linspace(0.0, common_horizon, N_GRID)
    result = {
        "common_horizon": common_horizon,
        "t_grid": t_grid,
        "survival_curves": {},
        "ibs_values": {},
        "auc_values": {},
        "summary": {},
    }
    for spec in specs:
        name = spec["name"]
        frame = frames[name]
        if frame.empty:
            raise RuntimeError(f"No successful outer splits were found for {name}.")
        for key, column in (("ibs_values", "integrated_brier"), ("auc_values", "cure_auc")):
            values = frame[column].to_numpy(dtype=float)
            result[key][name] = values[np.isfinite(values)]
            if not len(result[key][name]):
                raise RuntimeError(f"No finite {column} values were found for {name}.")
        curves = []
        for _, row in frame.iterrows():
            predictions = prediction_loader(
                row, df=df, model_name=spec["model_name"], kernel=spec["kernel"],
                outer_splits=outer_splits, results_dir=results_dir,
                project_root=project_root, refit_missing=refit_missing,
                prediction_grid=t_grid,
            )
            times = np.asarray(predictions["evaluation_time_grid"], dtype=float)
            survival = np.asarray(predictions["survival_test"], dtype=float)
            if times.ndim != 1 or len(times) < 2 or np.any(~np.isfinite(times)):
                raise ValueError(f"{name}, split {row.split}: invalid saved time grid.")
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"{name}, split {row.split}: time grid must increase.")
            if survival.shape != (len(outer_splits[int(row.split)][1]), len(times)):
                raise ValueError(f"{name}, split {row.split}: invalid survival matrix shape.")
            if np.any(~np.isfinite(survival)):
                raise ValueError(f"{name}, split {row.split}: non-finite survival predictions.")
            tolerance = 1e-10 * max(1.0, common_horizon)
            if times[0] > tolerance or times[-1] < common_horizon - tolerance:
                raise ValueError(
                    f"{name}, split {row.split}: saved survival grid does not cover "
                    f"the common plotting interval [0, {common_horizon}]."
                )
            split_curve = np.clip(np.mean(survival, axis=0), 0.0, 1.0)
            curves.append(np.interp(t_grid, times, split_curve))
        result["survival_curves"][name] = np.mean(np.vstack(curves), axis=0)
        result["summary"][name] = {
            "model_name": spec["model_name"],
            "kernel": spec["kernel"] or "parametric",
            "saved_successful": len(frame),
            "reconstructed": len(curves),
            "mean_ibs": float(np.mean(result["ibs_values"][name])),
            "mean_auc": float(np.mean(result["auc_values"][name])),
        }
    return result


def collect_kernel_score_data(model_name, results_dir=RESULTS_DIR):
    ibs_values, auc_values = [], []
    for kernel in KERNELS:
        path = unique_result_file(
            f"splits_melanoma_task*_{model_name}_kernel_{kernel}.csv", results_dir,
        )
        frame = read_split_file(path)
        ibs = frame.loc[frame["valid_ibs"], "integrated_brier"].to_numpy(dtype=float)
        auc = frame.loc[frame["valid_auc"], "cure_auc"].to_numpy(dtype=float)
        if not len(ibs) or not len(auc):
            raise RuntimeError(f"No finite successful IBS/cure-AUC values for {model_name}-{kernel}.")
        ibs_values.append(ibs)
        auc_values.append(auc)
    return ibs_values, auc_values


def save_figure(fig, output_base):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_base.with_suffix(extension) for extension in (".png", ".pdf", ".svg"))
    try:
        for path in paths:
            fig.savefig(path, dpi=400, bbox_inches="tight")
    finally:
        plt.close(fig)
    return paths


def kaplan_meier(time, event, horizon):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    event_times = np.sort(np.unique(time[event == 1]))
    event_times = event_times[event_times <= horizon]
    x = [0.0]
    y = [1.0]
    survival = 1.0
    for current_time in event_times:
        at_risk = int(np.sum(time >= current_time))
        n_events = int(np.sum((time == current_time) & (event == 1)))
        if at_risk > 0:
            survival *= 1.0 - n_events / at_risk
        x.append(float(current_time))
        y.append(float(survival))
    if x[-1] < horizon:
        x.append(float(horizon))
        y.append(float(survival))
    return (np.asarray(x), np.asarray(y))


def draw_score_panel(ax, values, specs, *, title, ylabel, reference=None, ylim=None):
    positions = np.arange(1, len(specs) + 1)
    boxplot = ax.boxplot(
        values, positions=positions, widths=0.58, patch_artist=True, showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.5},
        whiskerprops={"color": "0.30", "linewidth": 1.1},
        capprops={"color": "0.30", "linewidth": 1.1},
    )
    for (patch, spec) in zip(boxplot['boxes'], specs, strict=True):
        patch.set_facecolor(spec['color'])
        patch.set_edgecolor('0.25')
        patch.set_linewidth(0.9)
        patch.set_alpha(0.82)
        if spec.get('hatch'):
            patch.set_hatch(spec['hatch'])
    rng = np.random.default_rng(MASTER_SEED)
    for (position, split_values, spec) in zip(positions, values, specs, strict=True):
        jitter = rng.normal(loc=0.0, scale=0.055, size=len(split_values))
        ax.scatter(position + jitter, split_values, s=27, color=spec['color'], alpha=0.48, linewidths=0.0, zorder=3)
    if reference is not None:
        ax.axhline(reference, linestyle=':', linewidth=1.1, color='0.35', alpha=0.8)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=11.5)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels([spec['box_label'] for spec in specs], fontsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def draw_grouped_kernel_panel(
    ax, poisson_values, nb_values, *, title, ylabel, reference=None, ylim=None,
    kernels=None, kernel_labels=None,
):
    kernels = KERNELS if kernels is None else kernels
    kernel_labels = KERNEL_LABELS if kernel_labels is None else kernel_labels
    centers = np.arange(1, len(kernels) + 1, dtype=float)
    offset = 0.18
    box_width = 0.3
    poisson_positions = centers - offset
    nb_positions = centers + offset
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    poisson_color = default_colors[0]
    nb_color = default_colors[1]
    common_boxplot_kwargs = {
        "widths": box_width, "patch_artist": True, "showfliers": False,
        "medianprops": {"color": "0.15", "linewidth": 1.35},
        "whiskerprops": {"color": "0.30", "linewidth": 1.0},
        "capprops": {"color": "0.30", "linewidth": 1.0},
    }
    poisson_boxplot = ax.boxplot(poisson_values, positions=poisson_positions, **common_boxplot_kwargs)
    nb_boxplot = ax.boxplot(nb_values, positions=nb_positions, **common_boxplot_kwargs)
    for patch in poisson_boxplot['boxes']:
        patch.set_facecolor(poisson_color)
        patch.set_edgecolor('0.25')
        patch.set_linewidth(0.85)
        patch.set_alpha(0.78)
    for patch in nb_boxplot['boxes']:
        patch.set_facecolor(nb_color)
        patch.set_edgecolor('0.25')
        patch.set_linewidth(0.85)
        patch.set_alpha(0.78)
        patch.set_hatch('//')
    rng = np.random.default_rng(MASTER_SEED)
    for (position, split_values) in zip(poisson_positions, poisson_values, strict=True):
        jitter = rng.normal(loc=0.0, scale=0.025, size=len(split_values))
        ax.scatter(position + jitter, split_values, s=18, color=poisson_color, alpha=0.45, linewidths=0.0, zorder=3)
    for (position, split_values) in zip(nb_positions, nb_values, strict=True):
        jitter = rng.normal(loc=0.0, scale=0.025, size=len(split_values))
        ax.scatter(position + jitter, split_values, s=18, color=nb_color, alpha=0.45, linewidths=0.0, zorder=3)
    if reference is not None:
        ax.axhline(reference, linestyle=':', linewidth=1.1, color='0.35', alpha=0.8)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=11.5)
    ax.set_ylabel(ylabel)
    ax.set_xticks(centers)
    ax.set_xticklabels([kernel_labels[kernel] for kernel in kernels], rotation=30, ha='right', fontsize=8.5)
    ax.set_xlim(0.45, len(kernels) + 0.55)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return (poisson_color, nb_color)


def make_kernel_comparison_figure(
    results_dir=RESULTS_DIR, output_base=KERNEL_OUTPUT_BASE, *,
    score_collector=None, kernels=None, kernel_labels=None,
):
    score_collector = collect_kernel_score_data if score_collector is None else score_collector
    (poisson_ibs, poisson_auc) = score_collector('poisson', results_dir)
    (nb_ibs, nb_auc) = score_collector('bn', results_dir)
    all_auc = np.concatenate(poisson_auc + nb_auc)
    auc_min = float(np.nanmin(all_auc))
    auc_lower = max(0.0, min(0.48, auc_min - 0.03))
    (fig, (ax_a, ax_b)) = plt.subplots(nrows=1, ncols=2, figsize=(14.5, 5.4), gridspec_kw={'wspace': 0.24})
    poisson_color, nb_color = draw_grouped_kernel_panel(
        ax_a, poisson_ibs, nb_ibs,
        title="A  GG-KM kernel comparison: prediction error",
        ylabel="IPCW integrated Brier score",
        kernels=kernels, kernel_labels=kernel_labels,
    )
    draw_grouped_kernel_panel(
        ax_b, poisson_auc, nb_auc,
        title="B  GG-KM kernel comparison: cure discrimination",
        ylabel=r"$\mathrm{AUC}_{\mathrm{cure}}$", reference=0.5, ylim=(auc_lower, 1.0),
        kernels=kernels, kernel_labels=kernel_labels,
    )
    legend_handles = [
        Patch(facecolor=poisson_color, edgecolor="0.25", label="Poisson", alpha=0.78),
        Patch(facecolor=nb_color, edgecolor="0.25", hatch="//", label="Negative binomial", alpha=0.78),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(bottom=0.24, top=0.92, left=0.07, right=0.985, wspace=0.24)
    return save_figure(fig, output_base)


def make_figure(
    df, specs, reconstructed, output_base=OUTPUT_BASE, *,
    cohort_times=None, time_label='Follow-up time (years)',
):
    horizon = reconstructed['common_horizon']
    t_grid = reconstructed['t_grid']
    cohort_times = full_cohort_times if cohort_times is None else cohort_times
    (full_time, full_event) = cohort_times(df)
    (km_time, km_survival) = kaplan_meier(full_time, full_event, horizon)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        nrows=1, ncols=3, figsize=(17.2, 5.6),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0], "wspace": 0.28},
    )
    ax_a.step(km_time, km_survival, where='post', linewidth=2.0, color='black', label='Kaplan--Meier')
    for spec in specs:
        ax_a.plot(
            t_grid, reconstructed["survival_curves"][spec["name"]],
            linewidth=1.9, color=spec["color"], linestyle=spec["linestyle"], label=spec["name"],
        )
    ax_a.set_title('A  Out-of-sample survival calibration', loc='left', fontweight='bold', fontsize=11.5)
    ax_a.set_xlabel(time_label)
    ax_a.set_ylabel('Event-free survival')
    ax_a.set_xlim(0.0, horizon)
    ax_a.set_ylim(0.0, 1.02)
    ax_a.grid(axis='y', alpha=0.3)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ibs_values = [reconstructed['ibs_values'][spec['name']] for spec in specs]
    draw_score_panel(
        ax_b, ibs_values, specs, title="B  Paired repeated-holdout prediction error",
        ylabel="IPCW integrated Brier score",
    )
    auc_values = [reconstructed['auc_values'][spec['name']] for spec in specs]
    all_auc = np.concatenate(auc_values)
    auc_min = float(np.nanmin(all_auc))
    auc_lower = min(0.48, auc_min - 0.03)
    auc_lower = max(0.0, auc_lower)
    draw_score_panel(
        ax_c, auc_values, specs, title="C  Paired repeated-holdout cure discrimination",
        ylabel=r"$\mathrm{AUC}_{\mathrm{cure}}$", reference=0.5, ylim=(auc_lower, 1.0),
    )
    (handles, labels) = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.075), ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(bottom=0.25, top=0.92, left=0.06, right=0.985)
    return save_figure(fig, output_base)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--refit-missing", action="store_true",
        help="Refit saved best parameters only when a prediction NPZ is missing; never retune.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    df = load_melanoma(args.data_path)
    specs = create_model_specs(args.results_dir)
    reconstructed = reconstruct_results(
        df, specs, results_dir=args.results_dir, refit_missing=args.refit_missing,
    )
    print(f"Common plotting horizon: {reconstructed['common_horizon']:.6f} years")
    for model, summary in reconstructed["summary"].items():
        print(
            f"{model}: {summary['reconstructed']}/{summary['saved_successful']} splits, "
            f"mean IBS={summary['mean_ibs']:.4f}, mean cure AUC={summary['mean_auc']:.4f}"
        )
    output_base = args.output_dir / OUTPUT_BASE.name
    kernel_base = args.output_dir / KERNEL_OUTPUT_BASE.name
    paths = make_figure(df, specs, reconstructed, output_base)
    paths += make_kernel_comparison_figure(args.results_dir, kernel_base)
    summary_path = args.output_dir / f"{OUTPUT_BASE.name}_summary.csv"
    pd.DataFrame.from_dict(reconstructed["summary"], orient="index").rename_axis(
        "model"
    ).reset_index().to_csv(summary_path, index=False)
    for path in (*paths, summary_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
