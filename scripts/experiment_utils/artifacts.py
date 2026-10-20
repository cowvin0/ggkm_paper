"""Atomic checkpoints and portable diagnostic arrays for clinical experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def index_hash(
    indices,
):
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    return hashlib.sha256(indices.tobytes()).hexdigest()[:16]


def finite_or_none(
    value,
):
    if value is None:
        return None

    value = float(value)

    if np.isfinite(value):
        return value

    return None


def atomic_write_json(
    path,
    payload,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(f".{path.name}." f"{os.getpid()}.tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            allow_nan=False,
        )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        temporary,
        path,
    )


def atomic_write_npz(
    path,
    **arrays,
):
    """Atomically write a compressed NumPy archive."""
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
        )
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        temporary,
        path,
    )


def array_or_empty(
    value,
    *,
    dtype=float,
):
    if value is None:
        return np.empty(0, dtype=dtype)

    return np.asarray(
        value,
        dtype=dtype,
    )


def preprocessed_feature_names(
    preprocessor,
    n_features,
):
    """Return feature names when the preprocessor exposes them."""
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = np.asarray(
                preprocessor.get_feature_names_out(),
                dtype=str,
            ).reshape(-1)
            if names.size == n_features:
                return names
        except Exception:
            pass

    for attribute in (
        "feature_names_out_",
        "feature_names_",
    ):
        if hasattr(preprocessor, attribute):
            names = np.asarray(
                getattr(preprocessor, attribute),
                dtype=str,
            ).reshape(-1)
            if names.size == n_features:
                return names

    return np.asarray(
        [f"x{index}" for index in range(n_features)],
        dtype=str,
    )


def fitted_model_arrays(model):
    """Collect portable fitted numeric state useful for diagnostics."""
    arrays = {}

    for attribute in (
        "a_",
        "d_",
        "p_",
        "phi_",
        "beta0_",
        "intercept_",
        "coef_",
        "alpha_",
    ):
        if not hasattr(model, attribute):
            continue

        value = getattr(model, attribute)

        if value is None:
            continue

        try:
            array = np.asarray(
                value,
                dtype=float,
            )
        except (TypeError, ValueError):
            continue

        arrays[f"model_{attribute}"] = array

    return arrays


def portable_path(path, *, project_root):
    path = Path(path)

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def resolve_saved_path(path, *, project_root):
    path = Path(path)

    if path.is_absolute():
        return path

    return project_root / path


def load_valid_checkpoint(
    path,
    *,
    split_index,
    model_name,
    kernel,
    train_hash,
    test_hash,
    resolve_path,
):
    path = Path(path)

    if not path.exists():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            row = json.load(handle)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    expected_kernel = kernel if kernel is not None else "parametric"

    if (
        row.get("split") != split_index
        or row.get("model_name") != model_name
        or row.get("kernel") != expected_kernel
        or row.get("train_index_hash") != train_hash
        or row.get("test_index_hash") != test_hash
    ):
        return None

    if row.get("status") == "ok":
        plot_file = row.get("plot_data_file")

        if plot_file is None or not resolve_path(plot_file).exists():
            return None

    return row
