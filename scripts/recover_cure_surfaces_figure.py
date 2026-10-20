"""Plot pointwise cure-surface recovery from saved PCM diagnostics.

Run from any directory with this script's absolute path, or from the project
root with ``python -m scripts.recover_cure_surfaces_figure``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.contour import ContourSet
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
FIGURES_DIR = PROJECT_ROOT / "figures" / "review"
PDF_FILE = FIGURES_DIR / "simulation_cure_surface_recovery.pdf"
PNG_FILE = FIGURES_DIR / "simulation_cure_surface_recovery.png"

STYLE = {
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
GRID_POINTS = 41
POINTS = GRID_POINTS**2
SURFACES = (1, 2, 3)
SURFACE_LABELS = {1: "Linear", 2: "Quadratic", 3: "Periodic"}

# All four methods use the Poisson latent-count law, n=600, genpoisson.
METHODS = (
    {"key": "linear", "title": "GG-KM linear", "model": "poisson", "kernel": "linear"},
    {
        "key": "spline", "title": "Additive spline",
        "model": "spline_poisson", "kernel": "parametric",
    },
    {
        "key": "polynomial", "title": "GG-KM polynomial",
        "model": "poisson", "kernel": "polynomial",
    },
    {"key": "rbf", "title": "GG-KM RBF", "model": "poisson", "kernel": "rbf"},
)

PREDICTION_ALIASES = (
    "predicted_cure_surface",  # Current monte_carlo_pcm diagnostics schema.
    "surface_predicted_cure",
    "surface_cure_predictions",
    "surface_predictions",
    "cure_surface_predictions",
    "grid_predicted_cure",
    "surface_prediction",
    "surface_cure_hat",
)


def expit(z):
    """Numerically stable logistic function for the generating surfaces."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def true_cure_surface(surface, x1, x2):
    """The primary mechanisms in evaluate.simulated_data.cure_probability."""
    if surface == 1:
        eta = 0.15 - 1.00 * x1 + 0.75 * x2
    elif surface == 2:
        eta = 0.60 - 1.45 * x1**2 - 1.05 * x2**2 + 0.55 * x1 * x2
    elif surface == 3:
        eta = 0.10 + 1.15 * np.sin(np.pi * x1) + 0.80 * np.cos(np.pi * x2)
    else:
        raise ValueError(f"Unknown surface={surface}")
    return expit(eta)


def find_diagnostics_file(*, model, surface, kernel, diagnostics_dir=DIAGNOSTICS_DIR):
    pattern = (
        f"diagnostics_task*_{model}_n600_"
        f"method{surface}_genpoisson_kernel{kernel}.npz"
    )
    diagnostics_dir = Path(diagnostics_dir)
    matches = sorted(diagnostics_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one diagnostics file for {model}, "
            f"surface={surface}, kernel={kernel}; found {len(matches)}.\n"
            f"Pattern: {diagnostics_dir / pattern}\n"
            f"Matches: {[str(path) for path in matches]}\n"
            "Generate the required diagnostics with run_pcm_experiment.py first."
        )
    return matches[0]


def flatten_payload(value, prefix=""):
    """Support both flat NPZ files and legacy nested object dictionaries."""
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_name = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_payload(child, child_name))
        return output
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        try:
            return flatten_payload(value.item(), prefix)
        except Exception:
            pass
    output[prefix] = value
    return output


def load_npz_payload(path):
    with np.load(path, allow_pickle=True) as archive:
        flat = {}
        for key in archive.files:
            flat.update(flatten_payload(archive[key], key))
    return flat


def print_inventory(path, payload):
    print(f"\n{'=' * 78}\nDIAGNOSTIC FILE\n{'=' * 78}\n{path}")
    for key in sorted(payload):
        try:
            array = np.asarray(payload[key])
            print(f"  {key:45s} shape={str(array.shape):16s} dtype={array.dtype}")
        except Exception:
            print(f"  {key:45s} type={type(payload[key]).__name__}")


def normalize_name(name):
    return name.lower().replace("-", "_").replace(".", "_")


def prediction_score(name):
    name = normalize_name(name)
    score = 0
    if "surface" in name:
        score += 10
    if "grid" in name:
        score += 3
    if "cure" in name:
        score += 8
    if "pred" in name or "estimate" in name or "hat" in name:
        score += 10
    if "true" in name:
        score -= 20
    if "truth" in name:
        score -= 20
    if "test" in name:
        score -= 8
    if "survival" in name:
        score -= 15
    if "time" in name:
        score -= 10
    return score


def numeric_array(value):
    array = np.asarray(value)
    if array.dtype == object:
        try:
            array = np.asarray(array.tolist(), dtype=np.float64)
        except (TypeError, ValueError):
            return None
    try:
        return np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def normalize_prediction_array(array):
    """Normalize supported layouts to replications x 1681 surface points."""
    array = numeric_array(array)
    if array is None:
        return None
    if array.ndim == 1 and array.size == POINTS:
        return array.reshape(1, POINTS)
    if array.ndim == 2:
        if array.shape == (GRID_POINTS, GRID_POINTS):
            return array.reshape(1, POINTS)
        if array.shape[1] == POINTS:
            return array
        if array.shape[0] == POINTS:
            return array.T
    if array.ndim == 3:
        if array.shape[-2:] == (GRID_POINTS, GRID_POINTS):
            return array.reshape(array.shape[0], POINTS)
        if array.shape[:2] == (GRID_POINTS, GRID_POINTS):
            return np.moveaxis(array, -1, 0).reshape(array.shape[-1], POINTS)
    return None


def extract_surface_predictions(payload, path):
    normalized = {normalize_name(key): key for key in payload}
    for alias in PREDICTION_ALIASES:
        for name, original_key in normalized.items():
            if name == alias or name.endswith("_" + alias):
                candidate = normalize_prediction_array(payload[original_key])
                if candidate is not None:
                    print("Prediction array:", original_key, candidate.shape)
                    return candidate

    candidates = []
    for key, value in payload.items():
        candidate = normalize_prediction_array(value)
        if candidate is not None:
            candidates.append((prediction_score(key), key, candidate))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        print_inventory(path, payload)
        raise RuntimeError(
            "Could not find any diagnostics array compatible with a "
            "41 x 41 recovery surface."
        )
    best_score = candidates[0][0]
    if best_score <= 0 or sum(item[0] == best_score for item in candidates) != 1:
        print_inventory(path, payload)
        for score, key, candidate in candidates:
            print(f"  score={score:3d} {key}: {candidate.shape}")
        raise RuntimeError("Surface prediction array could not be identified unambiguously.")
    _, key, candidate = candidates[0]
    print("Prediction array:", key, candidate.shape)
    return candidate


def extract_coordinate(payload, coordinate):
    candidates = []
    for key, value in payload.items():
        name = normalize_name(key)
        if coordinate not in name:
            continue
        score = 12 + 10 * ("surface" in name) + 6 * ("grid" in name)
        score -= 8 * ("test" in name) + 8 * ("pred" in name)
        array = numeric_array(value)
        if array is None:
            continue
        if array.size == POINTS:
            candidates.append((score, key, array.reshape(POINTS)))
        elif array.ndim >= 2 and array.shape[-1] == POINTS and array.size:
            candidates.append((score, key, array.reshape(-1, POINTS)[0]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, key, values = candidates[0]
    print(f"{coordinate} coordinates:", key)
    return values


def extract_grid_matrix(payload):
    candidates = []
    for key, value in payload.items():
        name = normalize_name(key)
        if "surface" not in name and "grid" not in name:
            continue
        array = numeric_array(value)
        if array is None:
            continue
        score = 10 * ("surface" in name) + 8 * ("grid" in name)
        if array.ndim == 2 and array.shape == (POINTS, 2):
            candidates.append((score, key, array))
        elif array.ndim == 3 and array.shape[-2:] == (POINTS, 2) and array.size:
            candidates.append((score, key, array.reshape(-1, POINTS, 2)[0]))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, key, grid = candidates[0]
    print("Surface grid:", key, grid.shape)
    return grid[:, 0], grid[:, 1]


def build_grid_order(payload):
    grid_pair = extract_grid_matrix(payload)
    if grid_pair is not None:
        x1_flat, x2_flat = grid_pair
    else:
        x1_flat = extract_coordinate(payload, "x1")
        x2_flat = extract_coordinate(payload, "x2")
    if x1_flat is None or x2_flat is None:
        axis = np.linspace(-1.0, 1.0, GRID_POINTS, dtype=np.float64)
        x1_grid, x2_grid = np.meshgrid(axis, axis, indexing="xy")
        x1_flat, x2_flat = x1_grid.ravel(), x2_grid.ravel()
        print("Surface coordinates not stored; using the configured 41x41 [-1,1]^2 grid.")
    order = np.lexsort((x1_flat, x2_flat))
    x_values = np.unique(np.round(np.asarray(x1_flat)[order], 12))
    y_values = np.unique(np.round(np.asarray(x2_flat)[order], 12))
    if x_values.size != GRID_POINTS or y_values.size != GRID_POINTS:
        raise RuntimeError("Recovered coordinates do not form a 41 x 41 surface grid.")
    return x_values, y_values, order


def corresponding_replication_file(diagnostic_file, results_dir=RESULTS_DIR):
    name = diagnostic_file.name
    if not name.startswith("diagnostics_"):
        return None
    name = "replications_" + name[len("diagnostics_"):].removesuffix(".npz") + ".csv"
    candidate = Path(results_dir) / name
    return candidate if candidate.exists() else None


def mask_failed_replications(predictions, diagnostic_file, results_dir=RESULTS_DIR):
    """Mask legacy full-length diagnostics; current NPZs contain only successes."""
    replication_file = corresponding_replication_file(diagnostic_file, results_dir)
    if replication_file is None:
        return predictions
    import pandas as pd

    frame = pd.read_csv(replication_file)
    if "status" not in frame.columns or len(frame) != predictions.shape[0]:
        return predictions
    ok = frame["status"].astype(str).str.lower().eq("ok").to_numpy()
    result = predictions.copy()
    result[~ok, :] = np.nan
    print("Successful rows used:", int(ok.sum()), "/", int(ok.size))
    return result


def recovery_field(*, model, kernel, surface, results_dir=RESULTS_DIR):
    path = find_diagnostics_file(
        model=model, surface=surface, kernel=kernel,
        diagnostics_dir=Path(results_dir) / "diagnostics",
    )
    print(f"\nLoading: {path.name}")
    payload = load_npz_payload(path)
    predictions = extract_surface_predictions(payload, path)
    x_values, y_values, order = build_grid_order(payload)
    if predictions.shape[1] != POINTS:
        raise RuntimeError(f"{path.name}: expected {POINTS} surface points.")
    predictions = mask_failed_replications(predictions[:, order], path, results_dir)
    x1_grid, x2_grid = np.meshgrid(x_values, y_values, indexing="xy")
    truth = true_cure_surface(surface, x1_grid, x2_grid)
    finite = np.isfinite(predictions)
    n_valid = finite.sum(axis=0)
    predictions = np.where(finite, predictions, np.nan)
    mean_prediction = np.full(POINTS, np.nan, dtype=np.float64)
    np.divide(
        np.nansum(predictions, axis=0), n_valid,
        out=mean_prediction, where=n_valid > 0,
    )
    squared_error = (predictions - truth.ravel()[None, :])**2
    mean_squared_error = np.full(POINTS, np.nan, dtype=np.float64)
    np.divide(
        np.nansum(squared_error, axis=0), n_valid,
        out=mean_squared_error, where=n_valid > 0,
    )
    return {
        "x": x_values,
        "y": y_values,
        "true_cure": truth,
        "mean_cure": mean_prediction.reshape(GRID_POINTS, GRID_POINTS),
        "rmse": np.sqrt(mean_squared_error).reshape(GRID_POINTS, GRID_POINTS),
        "n_valid": n_valid.reshape(GRID_POINTS, GRID_POINTS),
        "n_total": int(predictions.shape[0]),
        "path": path,
    }


def contour_label_anchors(
    handle: ContourSet,
    blocked: list[tuple[float, float]],
    *,
    minimum_separation: float = 0.22,
    edge_margin: float = 0.10,
):
    anchors = []
    for path in handle.get_paths():
        branches = sorted(
            (branch for branch in path.to_polygons(closed_only=False) if len(branch) >= 3),
            key=len, reverse=True,
        )
        chosen = fallback = None
        for branch in branches:
            for fraction in (0.5, 0.35, 0.65, 0.2, 0.8):
                index = int(round(fraction * (len(branch) - 1)))
                point = float(branch[index][0]), float(branch[index][1])
                position = 0.5 * (point[0] + 1.0), 0.5 * (point[1] + 1.0)
                if not all(edge_margin <= value <= 1.0 - edge_margin for value in position):
                    continue
                if fallback is None:
                    fallback = point
                if all(
                    (position[0] - taken[0])**2 + (position[1] - taken[1])**2
                    >= minimum_separation**2 for taken in blocked
                ):
                    chosen = point
                    break
            if chosen is not None:
                break
        if chosen is None:
            chosen = fallback
        if chosen is not None:
            blocked.append((0.5 * (chosen[0] + 1.0), 0.5 * (chosen[1] + 1.0)))
            anchors.append(chosen)
    return anchors


def mean_rmse(field):
    values = field["rmse"]
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def plot_recovery(recovery):
    figure, axes = plt.subplots(
        len(SURFACES), len(METHODS), figsize=(7.15, 6.35), sharex=True, sharey=True,
    )
    fill_levels = np.linspace(0.0, 0.30, 13)
    contour_levels = (0.25, 0.50, 0.60)
    estimated_level = 0.50
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.30)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap="magma_r")
    true_halo = [path_effects.withStroke(linewidth=1.9, foreground="#ffffff")]
    estimated_halo = [path_effects.withStroke(linewidth=2.4, foreground="#111111")]
    annotation_box = {"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.0}

    for row_index, surface in enumerate(SURFACES):
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            field = recovery[surface, method["key"]]
            x_values, y_values = field["x"], field["y"]
            true_cure, mean_cure = field["true_cure"], field["mean_cure"]
            error_grid = field["rmse"]
            if np.isfinite(error_grid).any():
                axis.contourf(
                    x_values, y_values, np.ma.masked_invalid(error_grid),
                    levels=fill_levels, cmap="magma_r", norm=norm, extend="max",
                )
            else:
                axis.set_facecolor("#eceff1")
                axis.text(
                    0.5, 0.5, "No valid fits", transform=axis.transAxes,
                    ha="center", va="center", color=INK, fontsize=7.0,
                )
            finite_mean = mean_cure[np.isfinite(mean_cure)]
            estimated_present = bool(
                finite_mean.size and finite_mean.min() <= estimated_level <= finite_mean.max()
            )
            blocked = []
            if not estimated_present:
                axis.text(
                    0.02, 0.02, f"Estimated {estimated_level:.2f} level absent",
                    transform=axis.transAxes, ha="left", va="bottom",
                    fontsize=5.5, color="#111111", bbox=annotation_box,
                )
                blocked.extend((fraction, 0.06) for fraction in np.linspace(0.05, 0.70, 6))
            drawn_levels = [
                level for level in contour_levels
                if np.nanmin(true_cure) <= level <= np.nanmax(true_cure)
            ]
            if drawn_levels:
                true_handle = axis.contour(
                    x_values, y_values, true_cure, levels=drawn_levels,
                    colors="#111111", linestyles="-",
                    linewidths=[1.35 if level == estimated_level else 0.75 for level in drawn_levels],
                )
                true_handle.set_path_effects(true_halo)
                anchors = contour_label_anchors(true_handle, blocked)
                for label in axis.clabel(
                    true_handle, fmt="%.2f", fontsize=5.5, inline=True,
                    inline_spacing=3, manual=anchors,
                ):
                    label.set_path_effects(true_halo)
            if estimated_present:
                estimated_handle = axis.contour(
                    x_values, y_values, np.ma.masked_invalid(mean_cure),
                    levels=[estimated_level], colors="#ffffff",
                    linestyles="--", linewidths=1.4,
                )
                estimated_handle.set_path_effects(estimated_halo)
            axis.text(
                0.5, 1.02, f"RMSE {mean_rmse(field):.4f}", transform=axis.transAxes,
                ha="center", va="bottom", fontsize=5.5, color="#111111", bbox=annotation_box,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(-1.0, 1.0)
            axis.set_ylim(-1.0, 1.0)
            axis.set_xticks((-1.0, 0.0, 1.0))
            axis.set_yticks((-1.0, 0.0, 1.0))
            axis.tick_params(labelsize=6.5, colors=INK)
            for spine in axis.spines.values():
                spine.set_color(INK)
                spine.set_linewidth(0.6)
            if row_index == 0:
                axis.set_title(method["title"], fontsize=8.0, fontweight="bold", color=INK, pad=13.0)
            if row_index == len(SURFACES) - 1:
                axis.set_xlabel("$x_1$", color=INK)
            if column_index == 0:
                axis.set_ylabel("$x_2$", color=INK)
                axis.text(
                    -0.34, 0.5, SURFACE_LABELS[surface], transform=axis.transAxes,
                    rotation=90, ha="center", va="center", fontsize=8.0,
                    fontweight="bold", color=INK,
                )

    colorbar_axis = figure.add_axes((0.91, 0.18, 0.018, 0.68))
    colorbar = figure.colorbar(mappable, cax=colorbar_axis, extend="max")
    colorbar.set_label("Pointwise RMSE of predicted cure fraction", color=INK)
    colorbar.ax.tick_params(labelsize=6.5, colors=INK)
    figure.legend(
        handles=(
            Line2D(
                [0], [0], color="#111111", linestyle="-", linewidth=1.35,
                path_effects=true_halo, label="True cure contour",
            ),
            Line2D(
                [0], [0], color="#ffffff", linestyle="--", linewidth=1.4,
                path_effects=estimated_halo,
                label=f"Mean estimated {estimated_level:.2f} contour",
            ),
        ),
        loc="lower center", bbox_to_anchor=(0.5, 0.015), frameon=False,
        ncol=2, handlelength=2.9, columnspacing=2.0,
    )
    figure.subplots_adjust(
        left=0.10, right=0.89, bottom=0.09, top=0.94, wspace=0.10, hspace=0.18,
    )
    return figure


def main(results_dir=RESULTS_DIR, output_dir=FIGURES_DIR):
    mpl.use("Agg")
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    pdf_file = output_dir / PDF_FILE.name
    png_file = output_dir / PNG_FILE.name
    print(f"\n{'=' * 78}\nCURE-SURFACE RECOVERY")
    print(f"n = 600 | generating law = Poisson\n{'=' * 78}")
    recovery = {
        (surface, method["key"]): recovery_field(
            model=method["model"], kernel=method["kernel"], surface=surface,
            results_dir=results_dir,
        )
        for surface in SURFACES for method in METHODS
    }
    with mpl.rc_context(STYLE):
        figure = plot_recovery(recovery)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                pdf_file, bbox_inches="tight", pad_inches=0.04,
                metadata={
                    "Creator": "GG-KM simulation analysis",
                    "CreationDate": None, "ModDate": None,
                },
            )
            figure.savefig(png_file, dpi=300, bbox_inches="tight", pad_inches=0.04)
        finally:
            plt.close(figure)
    print(f"\n{'=' * 78}\nPANEL SUMMARY\n{'=' * 78}")
    for surface in SURFACES:
        print(f"\n{SURFACE_LABELS[surface]}")
        for method in METHODS:
            field = recovery[surface, method["key"]]
            minimum_valid = int(field["n_valid"].min())
            maximum_valid = int(field["n_valid"].max())
            print(
                f"  {method['title']:18s} RMSE={mean_rmse(field):.4f} "
                f"n_valid={minimum_valid}-{maximum_valid} n_total={field['n_total']}"
            )
    print(f"\n{'=' * 78}\nSAVED\n{'=' * 78}")
    print(pdf_file.resolve())
    print(png_file.resolve())
    return pdf_file, png_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="PCM results directory containing diagnostics/ and replication CSVs.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FIGURES_DIR,
        help="Directory for the PDF and PNG figures (default: figures/review).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir)
