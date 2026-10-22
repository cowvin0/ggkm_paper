"""Validate an independent PCM pilot and project Monte Carlo precision.

Allocation uses the largest sample standard deviation *within* a scenario/model
cell.  Variation between cell means is not Monte Carlo variation.  Cure MSE is
read directly from ``cure_mse``; the paper's cure-RMSE summaries are not used.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


PILOT_METRICS = ("test_ibs", "cure_mse", "test_auc")
_SEEDS = ("generation_seed", "split_seed", "inner_seed")
_IDENTITY = ("n", "method", "generating_distribution", "generating_phi", "model", "kernel")
_REQUIRED_COLUMNS = frozenset(("replication", "status", *_SEEDS, *_IDENTITY, *PILOT_METRICS))
_CELL_COLUMNS = (
    "task_id", "n", "method", "generating_name", "generating_distribution",
    "generating_phi", "model_name", "kernel",
)


def _positive_integer(value, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if isinstance(value, (bool, np.bool_)) or result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(result)


def _positive_finite(value, name: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a positive finite number.") from error
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


def _numeric(frame: pd.DataFrame, column: str, task_id: int) -> pd.Series:
    try:
        return pd.to_numeric(frame[column], errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(f"Pilot task {task_id}: {column!r} contains nonnumeric values.") from error


def _integer_column(frame: pd.DataFrame, column: str, task_id: int) -> pd.Series:
    values = _numeric(frame, column, task_id)
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError(f"Pilot task {task_id}: {column!r} must contain nonnegative integer values.")
    return values


def _validate_frame(frame: pd.DataFrame, cell: Mapping, planned: int) -> pd.DataFrame:
    task_id = cell["task_id"]
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Pilot task {task_id}: expected a pandas DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError(f"Pilot task {task_id}: duplicate column names.")
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Pilot task {task_id}: missing required columns: {', '.join(sorted(missing))}.")
    frame = frame.copy()
    replication = _integer_column(frame, "replication", task_id)
    if replication.duplicated().any():
        raise ValueError(f"Pilot task {task_id}: duplicate replication IDs.")
    if (replication >= planned).any():
        raise ValueError(f"Pilot task {task_id}: replication IDs must be in 0..{planned - 1}.")
    frame["replication"] = replication.astype(int)
    status = frame["status"].astype("string").str.strip().str.lower()
    if not status.isin(("ok", "failed")).all():
        raise ValueError(f"Pilot task {task_id}: status must be 'ok' or 'failed' on every row.")
    frame["status"] = status
    for column in _SEEDS:
        frame[column] = _integer_column(frame, column, task_id)
    for column in ("n", "method"):
        if not _numeric(frame, column, task_id).eq(cell[column]).all():
            raise ValueError(f"Pilot task {task_id}: {column!r} does not match the planned cell.")
    for column, expected in (
        ("model", cell["model_name"]),
        ("generating_distribution", cell["generating_distribution"]),
    ):
        if not frame[column].astype("string").eq(expected).fillna(False).all():
            raise ValueError(f"Pilot task {task_id}: {column!r} does not match the planned cell.")
    actual_kernel = frame["kernel"].astype("string").fillna("parametric")
    if not actual_kernel.eq(cell["kernel"] or "parametric").all():
        raise ValueError(f"Pilot task {task_id}: 'kernel' does not match the planned cell.")
    phi = _numeric(frame, "generating_phi", task_id)
    expected_phi = cell["generating_phi"]
    phi_matches = phi.isna() if expected_phi is None else np.isclose(phi, expected_phi, rtol=1e-10, atol=0)
    if not np.all(phi_matches):
        raise ValueError(f"Pilot task {task_id}: 'generating_phi' does not match the planned cell.")
    for column in PILOT_METRICS:
        frame[column] = _numeric(frame, column, task_id)
    return frame


def _required_replications(sd: float, target: float) -> int:
    ratio = sd / target
    if not math.isfinite(ratio) or not math.isfinite(ratio * ratio):
        raise ValueError("The requested target MCSE produces an unrepresentable replication count.")
    return max(1, math.ceil(ratio * ratio))


def summarize_pilot_cells(
    cells: Sequence[Mapping],
    frames: Mapping[int, pd.DataFrame],
    *,
    planned_replications: int = 5,
    minimum_successful: int = 4,
    final_replications: int = 250,
    ibs_target: float = 0.003,
    cure_mse_target: float = 0.005,
) -> tuple[pd.DataFrame, dict]:
    """Return one summary row per planned cell and a JSON-safe allocation report.

    Missing cells and unfinished replication IDs produce an incomplete report;
    invalid schemas, identities, or shared replication seeds raise ``ValueError``.
    Failed rows remain in the attempt counts and never contribute metric values.
    """
    planned = _positive_integer(planned_replications, "planned_replications")
    minimum = _positive_integer(minimum_successful, "minimum_successful")
    final = _positive_integer(final_replications, "final_replications")
    if minimum < 2 or minimum > planned:
        raise ValueError("minimum_successful must be at least 2 and no larger than planned_replications.")
    targets = {
        "test_ibs": _positive_finite(ibs_target, "ibs_target"),
        "cure_mse": _positive_finite(cure_mse_target, "cure_mse_target"),
        "test_auc": None,
    }
    cells = list(cells)
    if not cells:
        raise ValueError("The pilot design must contain at least one cell.")
    task_ids = []
    for cell in cells:
        missing = set(_CELL_COLUMNS).difference(cell)
        if missing:
            raise ValueError(f"Pilot cell is missing fields: {', '.join(sorted(missing))}.")
        task_id = cell["task_id"]
        if isinstance(task_id, (bool, np.bool_)) or not isinstance(task_id, (int, np.integer)) or task_id < 0:
            raise ValueError("Pilot task IDs must be nonnegative integers.")
        task_ids.append(int(task_id))
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("The pilot design contains duplicate task IDs.")
    unknown = set(frames).difference(task_ids)
    if unknown:
        raise ValueError(f"Pilot data contains unplanned task IDs: {sorted(unknown)}.")

    rows, problems = [], []
    shared_seeds = {}
    expected_replications = set(range(planned))
    for cell in cells:
        task_id = int(cell["task_id"])
        frame = _validate_frame(frames[task_id], cell, planned) if task_id in frames else None
        row = {key: cell[key] for key in _CELL_COLUMNS}
        row["task_id"] = task_id
        if "source_task_id" in cell:
            row["source_task_id"] = cell["source_task_id"]
        row.update(
            planned_replications=planned,
            attempted_replications=0 if frame is None else len(frame),
            successful_replications=0 if frame is None else int(frame["status"].eq("ok").sum()),
            failed_replications=0 if frame is None else int(frame["status"].eq("failed").sum()),
        )
        cell_problems = []
        if frame is None:
            cell_problems.append("missing replication file")
        else:
            missing_replications = expected_replications.difference(frame["replication"])
            if missing_replications:
                cell_problems.append(f"missing replication IDs {sorted(missing_replications)}")
            for values in frame[["replication", *_SEEDS]].itertuples(index=False, name=None):
                replication, *seeds = values
                key = (cell["n"], cell["method"], cell["generating_name"], int(replication))
                entry = (task_id, tuple(int(seed) for seed in seeds))
                previous = shared_seeds.setdefault(key, entry)
                if previous[1] != entry[1]:
                    raise ValueError(
                        f"Pilot tasks {previous[0]} and {task_id}: inconsistent shared replication seeds "
                        f"for scenario {key[:3]}, replication {replication}."
                    )
        if row["successful_replications"] < minimum:
            cell_problems.append(f"{row['successful_replications']} successful replications; at least {minimum} required")
        for metric in PILOT_METRICS:
            values = np.array([], dtype=float)
            if frame is not None:
                keep = frame["status"].eq("ok") & np.isfinite(frame[metric])
                values = frame.loc[keep, metric].to_numpy(dtype=float)
            count = len(values)
            sd = float(np.std(values, ddof=1)) if count > 1 else None
            mean = float(np.mean(values)) if count else None
            if sd is not None and (not math.isfinite(sd) or not math.isfinite(mean)):
                raise ValueError(f"Pilot task {task_id}: {metric!r} has nonfinite summary statistics.")
            target = targets[metric]
            row[f"{metric}_count"] = count
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_projected_mcse"] = None if sd is None else sd / math.sqrt(final)
            row[f"{metric}_required_replications"] = (
                None if sd is None or target is None else _required_replications(sd, target)
            )
            if count < minimum:
                cell_problems.append(f"{metric} has {count} finite successful values; at least {minimum} required")
        row["complete"] = not cell_problems
        row["problems"] = "; ".join(cell_problems)
        problems.extend(f"Pilot task {task_id}: {problem}." for problem in cell_problems)
        rows.append(row)

    summary = pd.DataFrame(rows)
    complete = bool(summary["complete"].all())
    metrics = {}
    for metric, target in targets.items():
        valid_sd = summary[f"{metric}_sd"].dropna()
        max_sd = float(valid_sd.max()) if not valid_sd.empty else None
        projected_mcse = None if max_sd is None else max_sd / math.sqrt(final)
        required = None if max_sd is None or target is None else _required_replications(max_sd, target)
        worst_cells = (
            [] if max_sd is None else summary.loc[summary[f"{metric}_sd"].eq(max_sd), "task_id"].astype(int).tolist()
        )
        metrics[metric] = {
            "max_sd": max_sd,
            "provisional": not complete,
            "target_mcse": target,
            "required_replications": required,
            "projected_mcse": projected_mcse,
            "meets_target": None if target is None or projected_mcse is None else bool(projected_mcse <= target),
            "worst_cells": worst_cells,
        }
    required = max(metrics[metric]["required_replications"] for metric in ("test_ibs", "cure_mse")) if complete else None
    ready = complete and all(metrics[metric]["meets_target"] for metric in ("test_ibs", "cure_mse"))
    report = {
        "schema_version": 1,
        "planned_pilot_replications": planned,
        "minimum_successful": minimum,
        "final_replications": final,
        "n_expected_cells": len(cells),
        "n_recorded_cells": len(frames),
        "n_complete_cells": int(summary["complete"].sum()),
        "complete": complete,
        "ready_for_final": ready,
        "decision": "ready" if ready else "insufficient_precision" if complete else "incomplete",
        "required_replications": required,
        "metrics": metrics,
        "problems": problems,
    }
    return summary, report
