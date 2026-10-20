"""Read and validate saved melanoma holdout results without fitting models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "melanoma.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "melanoma_results" / "grid_search"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "review"

N_OUTER_SPLITS = 50
OUTER_TEST_SIZE = 62
MASTER_SEED = 20260817

KERNELS = [
    "linear", "rbf", "laplacian", "exponential", "cauchy", "polynomial", "wavelet",
]
KERNEL_LABELS = {
    "linear": "Linear", "rbf": "RBF", "laplacian": "Laplacian",
    "exponential": "Exponential", "cauchy": "Cauchy",
    "polynomial": "Polynomial", "wavelet": "Wavelet",
}


def load_melanoma(data_path=DATA_PATH):
    """Apply the experiment's event coding while preserving original row order."""
    return (
        pd.read_csv(data_path)
        .assign(status=lambda frame: (frame.status == 1).astype(int))
        .fillna(0)
        .rename(columns={"status": "delta"})
        .reset_index(drop=True)
    )


def index_hash(indices):
    indices = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(indices.tobytes()).hexdigest()[:16]


def parse_best_params(value):
    if isinstance(value, dict):
        return dict(value)
    if value is None or pd.isna(value):
        return {}
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return {}
    try:
        params = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not decode best_params: {text}") from error
    if not isinstance(params, dict):
        raise TypeError("best_params must contain a JSON dictionary.")
    return params


def unique_result_file(pattern, results_dir=RESULTS_DIR):
    matches = sorted(Path(results_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched: {Path(results_dir) / pattern}")
    if len(matches) != 1:
        raise RuntimeError(
            f"More than one file matched {pattern}:\n"
            + "\n".join(str(path) for path in matches)
        )
    return matches[0]


def read_split_file(path):
    """Keep the original score-validity rules and reject ambiguous split IDs."""
    path = Path(path)
    frame = pd.read_csv(
        path, dtype={"train_index_hash": "string", "test_index_hash": "string"},
    )
    required = {
        "split", "status", "converged", "horizon", "integrated_brier", "cure_auc",
        "best_params", "train_index_hash", "test_index_hash",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path.name} is missing columns: {sorted(missing)}")
    splits = pd.to_numeric(frame["split"], errors="raise")
    if not np.isfinite(splits).all() or (splits < 0).any() or (splits % 1 != 0).any():
        raise ValueError(f"{path.name} contains invalid split identifiers.")
    frame["split"] = splits.astype(int)
    if frame["split"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate split identifiers.")
    for column in ("integrated_brier", "cure_auc", "horizon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["successful"] = frame["status"].astype(str).str.strip().str.lower().eq("ok")
    frame["valid_ibs"] = frame["successful"] & np.isfinite(frame["integrated_brier"])
    frame["valid_auc"] = frame["successful"] & np.isfinite(frame["cure_auc"])
    # Carry the CSV identity into iterrows/itertuples for portable NPZ lookup.
    frame["_result_file"] = str(path.resolve())
    return frame


def choose_best_kernel(model_name, results_dir=RESULTS_DIR):
    """Select the highest mean cure AUC; break ties by IBS and kernel name."""
    rows = []
    for kernel in KERNELS:
        path = unique_result_file(
            f"splits_melanoma_task*_{model_name}_kernel_{kernel}.csv", results_dir,
        )
        frame = read_split_file(path)
        auc = frame.loc[frame["valid_auc"], "cure_auc"]
        ibs = frame.loc[frame["valid_ibs"], "integrated_brier"]
        if len(auc):
            rows.append({
                "kernel": kernel, "mean_auc": float(auc.mean()),
                "mean_ibs": float(ibs.mean()) if len(ibs) else np.nan, "path": path,
            })
    if not rows:
        raise RuntimeError(f"No successful GG-KM cure-AUC results for {model_name}.")
    comparison = pd.DataFrame(rows).sort_values(
        ["mean_auc", "mean_ibs", "kernel"],
        ascending=[False, True, True], kind="mergesort",
    )
    return comparison.iloc[0].to_dict()


def construct_outer_splits(
    frame, n_splits=N_OUTER_SPLITS, test_size=OUTER_TEST_SIZE, random_state=MASTER_SEED,
):
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state,
    )
    return {
        split: (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for split, (train, test) in enumerate(splitter.split(frame, frame["delta"]))
    }


def verify_split_hashes(frame, outer_splits, model_name):
    for _, row in frame.iterrows():
        split = int(row["split"])
        if split not in outer_splits:
            raise RuntimeError(f"Unknown outer split {split} for {model_name}.")
        train, test = outer_splits[split]
        for label, indices in (("train", train), ("test", test)):
            stored = row[f"{label}_index_hash"]
            reconstructed = index_hash(indices)
            if pd.isna(stored) or str(stored) != reconstructed:
                raise RuntimeError(
                    f"Outer {label} split mismatch. Model: {model_name}; split: {split}; "
                    f"stored: {stored}; reconstructed: {reconstructed}. "
                    "Use the same melanoma CSV and split settings as the experiment."
                )


def _row_value(row, key, default=None):
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def _present(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _prediction_path(row, results_dir, project_root):
    """Find archives after moving a complete results directory off the cluster."""
    results_dir, project_root = Path(results_dir), Path(project_root)
    split = int(_row_value(row, "split"))
    filename = f"split_{split:03d}.npz"
    candidates = []
    saved_candidates = []
    source = _row_value(row, "_result_file")
    if _present(source):
        prefix = Path(str(source)).stem.removeprefix("splits_")
        candidates.append(results_dir / "plot_data" / prefix / filename)
    saved = _row_value(row, "plot_data_file")
    if _present(saved):
        saved = Path(str(saved))
        if saved.is_absolute():
            saved_candidates.append(saved)
        else:
            candidates.append(results_dir / saved)
            saved_candidates.append(project_root / saved)
        if "plot_data" in saved.parts:
            position = saved.parts.index("plot_data")
            candidates.append(results_dir.joinpath(*saved.parts[position:]))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    # Raw dictionaries may not carry the source filename, so use model metadata.
    model = _row_value(row, "model_name")
    kernel = _row_value(row, "kernel")
    if _present(model):
        suffix = str(model)
        if _present(kernel) and str(kernel) not in {"parametric", "none", "None"}:
            suffix += f"_kernel_{kernel}"
        matches = sorted((results_dir / "plot_data").glob(f"melanoma_task*_{suffix}/{filename}"))
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous prediction archives for split {split}: "
                + ", ".join(str(path) for path in matches)
            )
        if matches:
            return matches[0]
    for candidate in dict.fromkeys(saved_candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No saved prediction archive found for split {split} in {results_dir}. "
        "Copy the experiment's plot_data/ directory together with its split CSVs, "
        "or regenerate predictions on the cluster. Older CSV-only runs require "
        "the explicit --refit-missing option."
    )


def _scalar_text(arrays, key, path):
    value = arrays[key]
    if value.ndim != 0:
        raise ValueError(f"{path}: {key} must be scalar metadata.")
    return str(value.item())


def _indices(values, key, path):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{path}: {key} must be a one-dimensional integer array.")
    if (values < 0).any() or len(np.unique(values)) != len(values) or not len(values):
        raise ValueError(f"{path}: {key} must contain unique, nonnegative patient IDs.")
    return values.astype(np.int64, copy=False)


def load_split_predictions(
    row, *, results_dir=RESULTS_DIR, project_root=PROJECT_ROOT,
    expected_train_indices=None, expected_test_indices=None,
) -> dict[str, np.ndarray]:
    """Load one saved holdout, preserving the test-row to patient-ID mapping."""
    path = _prediction_path(row, results_dir, project_root)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read melanoma prediction archive {path}: {error}") from error
    required = {
        "split", "model_name", "kernel", "train_indices", "test_indices",
        "evaluation_time_grid", "survival_test", "cure_probability_test",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"{path}: missing saved prediction fields {sorted(missing)}.")
    split_value = np.asarray(arrays["split"])
    if split_value.ndim != 0 or split_value.item() != int(_row_value(row, "split")):
        raise ValueError(f"{path}: split identifier does not match its CSV row.")
    for key in ("model_name", "kernel"):
        stored = _scalar_text(arrays, key, path)
        expected = _row_value(row, key)
        if _present(expected) and stored != str(expected):
            raise ValueError(f"{path}: {key} metadata does not match its CSV row.")
    for label, expected in (
        ("train", expected_train_indices), ("test", expected_test_indices),
    ):
        values = _indices(arrays[f"{label}_indices"], f"{label}_indices", path)
        arrays[f"{label}_indices"] = values
        if expected is not None and not np.array_equal(values, expected):
            raise ValueError(f"{path}: saved {label} patient indices do not match the outer split.")
        saved_hash = _row_value(row, f"{label}_index_hash")
        if _present(saved_hash) and index_hash(values) != str(saved_hash):
            raise ValueError(f"{path}: saved {label} patient indices do not match the CSV hash.")
    if np.intersect1d(arrays["train_indices"], arrays["test_indices"]).size:
        raise ValueError(f"{path}: training and test patient indices overlap.")
    grid = np.asarray(arrays["evaluation_time_grid"], dtype=float)
    if (
        grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
        or (grid < 0).any() or (np.diff(grid) <= 0).any()
    ):
        raise ValueError(f"{path}: evaluation_time_grid must be finite and strictly increasing.")
    n_test = len(arrays["test_indices"])
    for key, expected_shape in (
        ("survival_test", (n_test, len(grid))), ("cure_probability_test", (n_test,)),
    ):
        values = np.asarray(arrays[key], dtype=float)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"{path}: {key} must contain finite values with shape {expected_shape}.")
        if ((values < -1e-10) | (values > 1 + 1e-10)).any():
            raise ValueError(f"{path}: {key} must contain probabilities between zero and one.")
        arrays[key] = values
    arrays["evaluation_time_grid"] = grid
    if "evaluation_horizon" in arrays:
        horizon = np.asarray(arrays["evaluation_horizon"], dtype=float)
        if horizon.ndim != 0 or not np.isfinite(horizon) or horizon < grid[-1] - 1e-10:
            raise ValueError(f"{path}: evaluation_horizon is inconsistent with its time grid.")
        csv_horizon = _row_value(row, "horizon")
        if _present(csv_horizon) and not np.isclose(float(horizon), float(csv_horizon)):
            raise ValueError(f"{path}: evaluation horizon does not match its CSV row.")
    for label in ("train", "test"):
        for suffix in ("time", "event"):
            key = f"{label}_{suffix}"
            if key in arrays:
                values = np.asarray(arrays[key], dtype=float)
                if values.shape != arrays[f"{label}_indices"].shape or not np.isfinite(values).all():
                    raise ValueError(f"{path}: {key} does not match its patient indices.")
    return arrays
