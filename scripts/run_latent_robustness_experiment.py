"""Run paired GGKM latent-count misspecification experiments or their pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.metrics import integrated_brier_score, uno_c_index_rmst
from evaluate.simulated_data import cure_probability
from evaluate.util_cross_validation import _make_evaluation_grid, _select_grid_parameters, fit_cure_model
from skcure.ggkm import GGKM
from skcure._latency import _GeneralizedGammaLatencyOptimizer
from skcure.metrics import auc_cure
from scripts import latent_robustness_utils as study
from scripts.experiment_utils.artifacts import atomic_write_json, atomic_write_npz, finite_or_none, index_hash
from scripts.experiment_utils.runtime import get_worker_count


build_experiments = study.build_experiments


class _KnownLatencyOptimizer(_GeneralizedGammaLatencyOptimizer):
    def maximize(self, *args, **kwargs):
        """The diagnostic fixes GG parameters; count/predictor updates still run."""
        return None


class KnownLatencyGGKM(GGKM):
    """Research-only GGKM diagnostic using the shared EM with fixed latency."""

    def _reset_parameters_for_fit(self):
        super()._reset_parameters_for_fit()
        self._latency_optimizer = _KnownLatencyOptimizer()


def estimator_factory(model_name, latency_mode, design):
    parameters = dict(design["estimator_parameters"][model_name])
    if latency_mode == "fixed":
        parameters.update(design["gg_parameters"])
    elif latency_mode != "estimated":
        raise ValueError(f"Unknown latency mode: {latency_mode}")
    cls = KnownLatencyGGKM if latency_mode == "fixed" else GGKM
    return partial(cls, **parameters)


def atomic_write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replication_paths(directory, task_id, replication):
    stem = f"case_{task_id:03d}/replication_{replication:04d}"
    return Path(directory) / "checkpoints" / f"{stem}.json", Path(directory) / "samples" / f"{stem}.npz"


def expected_fits(case):
    return {(mode, model) for mode in study.latency_modes(case) for model in study.FIT_MODELS}


def load_checkpoint(directory, case, replication, design):
    checkpoint, archive = replication_paths(directory, case["task_id"], replication)
    if not checkpoint.exists():
        return None
    saved = json.loads(checkpoint.read_text())
    if saved.get("study_id") != design["study_id"] or saved.get("task_id") != case["task_id"] or saved.get("replication") != replication:
        raise ValueError(f"Mismatched replication checkpoint: {checkpoint}")
    rows = saved.get("rows", [])
    if len(rows) != len(expected_fits(case)) or {(row["latency_mode"], row["model_name"]) for row in rows} != expected_fits(case):
        raise ValueError(f"Incomplete fitted-model records in {checkpoint}")
    if saved.get("sample_sha256") is not None:
        if not archive.is_file() or study.checksum(archive) != saved["sample_sha256"]:
            raise ValueError(f"Missing or modified replication sample: {archive}")
    elif not saved.get("preparation_failed"):
        raise ValueError(f"Checkpoint has no verifiable generated sample: {checkpoint}")
    return rows


def _fit_one(model_name, mode, case, design, context, base):
    start = perf_counter()
    row = {**base, "model_name": model_name, "latency_mode": mode,
           "status": "failed", "converged": False, "failure_reason": None}
    for name in (*study.METRICS, "cure_bias", "cure_mse", "cure_auc", "grid_cure_rmse", "grid_cure_bias",
                 "tail_survival_rmse", "fitted_a", "fitted_d", "fitted_p", "fitted_phi", "best_inner_ibs"):
        row[name] = None
    recorded_warnings = []
    try:
        X, time, event = (context[name] for name in ("X_train_raw", "train_time", "train_event"))
        factory = estimator_factory(model_name, mode, design)
        candidates = design["parameter_grids"][model_name]
        # The factory has the public defaults; tuned keyword arguments replace
        # these defaults in functools.partial, exactly as in the existing grid.
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            if len(candidates) == 1:
                parameters, inner_score = dict(candidates[0]), None
            else:
                parameters, inner_score = _select_grid_parameters(
                    X, time, event, factory, "rbf", candidates,
                    n_inner_splits=design["inner_splits"], t_grid_points=design["inner_grid_points"],
                    random_state=base["inner_seed"],
                )
            model = factory(kernel="rbf", **parameters)
            fit_cure_model(model, context["X_train"], time, event)
        grid = context["time_grid"]
        survival = np.asarray(model.predict_survival(context["X_test"], grid))
        cure = np.asarray(model.predict_cure_probability(context["X_test"]))
        if not np.isfinite(survival).all() or not np.isfinite(cure).all():
            raise FloatingPointError("Nonfinite held-out predictions")
        ibs = integrated_brier_score(survival, context["test_time"], context["test_event"], time, event, grid)
        if not np.isfinite(ibs):
            raise FloatingPointError("Nonfinite held-out IBS")
        error = cure - context["true_cure_test"]
        row.update(
            integrated_brier=float(ibs),
            uno_c_index=finite_or_none(uno_c_index_rmst(
                survival, context["test_time"], context["test_event"], time, event, grid, tau=grid[-1],
            )),
            cure_bias=float(error.mean()), cure_mse=float(np.mean(error ** 2)), cure_rmse=float(np.sqrt(np.mean(error ** 2))),
            best_inner_ibs=finite_or_none(inner_score), best_params=json.dumps(parameters, sort_keys=True),
        )
        grid_cure = np.asarray(model.predict_cure_probability(context["X_truth_grid"]))
        grid_survival = np.asarray(model.predict_survival(context["X_truth_grid"], grid))
        tail_survival = np.asarray(model.predict_survival(context["X_truth_grid"], context["tail_time_grid"]))
        if not all(np.isfinite(values).all() for values in (grid_cure, grid_survival, tail_survival)):
            raise FloatingPointError("Nonfinite truth-grid predictions")
        cure_error = grid_cure - context["true_cure_grid"]
        row["grid_cure_rmse"] = float(np.sqrt(np.mean(cure_error ** 2)))
        row["grid_cure_bias"] = float(cure_error.mean())
        for name, predicted, truth, times in (
            ("survival_rmse", grid_survival, context["true_survival_grid"], grid),
            ("tail_survival_rmse", tail_survival, context["true_tail_survival"], context["tail_time_grid"]),
        ):
            mse = np.mean((predicted - truth) ** 2, axis=0)
            row[name] = float(np.sqrt(np.trapezoid(mse, times) / (times[-1] - times[0])))
        try:
            row["cure_auc"] = finite_or_none(auc_cure(cure)[0])
        except (ValueError, FloatingPointError):
            pass  # Cure AUC is a retained secondary diagnostic, not a fit criterion.
        for name in ("a", "d", "p"):
            row[f"fitted_{name}"] = finite_or_none(getattr(model, f"{name}_"))
        row["fitted_phi"] = finite_or_none(getattr(model, "phi_", None))
        if mode == "fixed" and any(row[f"fitted_{name}"] != value for name, value in design["gg_parameters"].items()):
            raise RuntimeError("The fixed-latency diagnostic changed a latency parameter")
        row.update(status="ok", converged=bool(model.converged_), n_iter=int(model.n_iter_))
    except Exception as error:
        row["failure_reason"] = f"{type(error).__name__}: {error}"
    row["warnings"] = json.dumps(list(dict.fromkeys(str(warning.message) for warning in recorded_warnings)))
    row["runtime_seconds"] = float(perf_counter() - start)
    return row


def run_replication(case, replication, design, directory, calibration):
    start = perf_counter()
    checkpoint, archive = replication_paths(directory, case["task_id"], replication)
    seeds = study.replication_seeds(case, replication, design["master_seed"], phase=design["phase"])
    base = {
        **case, **seeds, "replication": replication, "study_id": design["study_id"],
        "phase": design["phase"], "censor_mean": calibration["censor_mean"], "kernel": "rbf",
    }
    sample_checksum = None
    preparation_failed = False
    try:
        data, _ = study.generate_replication(case, replication, design, calibration["censor_mean"])
        train, test = train_test_split(
            np.arange(case["n"]), stratify=data["event"], test_size=design["test_size"], random_state=seeds["split_seed"],
        )
        time, event, X = (data[name] for name in ("time", "event", "X"))
        grid = _make_evaluation_grid(time[train], event[train], points=design["outer_grid_points"])
        scaler = StandardScaler().fit(X[train])
        axis = np.linspace(-1, 1, design["truth_grid_points"])
        x1, x2 = np.meshgrid(axis, axis, indexing="xy")
        truth_X = np.column_stack((x1.ravel(), x2.ravel()))
        tail_end = study.gengamma.ppf(0.99, a=study.GG_PARAMETERS["d"] / study.GG_PARAMETERS["p"], c=study.GG_PARAMETERS["p"], scale=study.GG_PARAMETERS["a"])
        tail_grid = np.linspace(0, tail_end, design["outer_grid_points"])
        context = {
            "X_train_raw": X[train], "X_train": scaler.transform(X[train]), "X_test": scaler.transform(X[test]),
            "train_time": time[train], "test_time": time[test], "train_event": event[train], "test_event": event[test],
            "true_cure_test": data["true_cure"][test], "time_grid": grid, "tail_time_grid": tail_grid,
            "X_truth_grid": scaler.transform(truth_X),
            "true_cure_grid": cure_probability(truth_X[:, 0], truth_X[:, 1], case["method"]),
            "true_survival_grid": study.true_survival(truth_X, grid, case),
            "true_tail_survival": study.true_survival(truth_X, tail_grid, case),
        }
        susceptible = data["true_count"] > 0
        base.update(
            data_hash=study.array_hash(data), train_index_hash=index_hash(train), test_index_hash=index_hash(test),
            horizon=float(grid[-1]), time_grid_hash=study.array_hash({"grid": grid}),
            training_size=len(train), test_size=len(test),
            realized_censoring=float(np.mean(event == 0)), true_cure_fraction=float(np.mean(~susceptible)),
            susceptible_censoring=finite_or_none(np.mean(event[susceptible] == 0) if susceptible.any() else None),
            generated_count_mean=float(np.mean(data["true_count"])), generated_count_variance=float(np.var(data["true_count"])),
        )
        atomic_write_npz(archive, **data, train_indices=train, test_indices=test, time_grid=grid,
                         tail_time_grid=tail_grid, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        sample_checksum = study.checksum(archive)
        rows = [_fit_one(model, mode, case, design, context, base)
                for mode in study.latency_modes(case) for model in study.FIT_MODELS]
    except Exception as error:
        preparation_failed = True
        rows = [{**base, "latency_mode": mode, "model_name": model, "status": "failed", "converged": False,
                 "failure_reason": f"Preparation: {type(error).__name__}: {error}",
                 "runtime_seconds": float(perf_counter() - start), **{metric: None for metric in study.METRICS}}
                for mode in study.latency_modes(case) for model in study.FIT_MODELS]
    atomic_write_json(checkpoint, {
        "study_id": design["study_id"], "task_id": case["task_id"], "replication": replication,
        "sample_sha256": sample_checksum, "preparation_failed": preparation_failed, "rows": rows,
    })
    return rows


def run_case(case, design, directory, *, workers=1):
    study.write_design(directory, design)
    calibration = study.calibrate_censoring(case, sample_size=design["calibration_size"])
    atomic_write_json(Path(directory) / "calibration" / f"case_{case['task_id']:03d}.json", {**case, **calibration})
    records, pending = [], []
    for replication in range(design["replications"]):
        cached = load_checkpoint(directory, case, replication, design)
        if cached is None:
            pending.append(replication)
        else:
            records.extend(cached)
    print(f"{design['phase']} case {case['task_id']}: {case}; cached={design['replications'] - len(pending)}, pending={len(pending)}", flush=True)
    if workers == 1:
        for replication in pending:
            records.extend(run_replication(case, replication, design, directory, calibration))
            print(f"Completed replication {replication}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_replication, case, replication, design, directory, calibration): replication for replication in pending}
            for future in as_completed(futures):
                records.extend(future.result())
                print(f"Completed replication {futures[future]}", flush=True)
    frame = pd.DataFrame(records).sort_values(["replication", "latency_mode", "model_name"])
    path = Path(directory) / f"replications_case_{case['task_id']:03d}.csv"
    atomic_write_csv(path, frame)
    atomic_write_json(path.with_suffix(".metadata.json"), {
        "study_id": design["study_id"], "task_id": case["task_id"], "rows": len(frame), "csv_sha256": study.checksum(path),
    })
    print(f"Saved {path}; successful={int(frame.status.eq('ok').sum())}/{len(frame)}", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "final"), default="final")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-id", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--replications", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    try:
        design = study.build_design(args.phase, replications=args.replications, seed=args.seed)
        directory = args.output_dir or study.phase_directory(args.phase)
        workers = args.workers if args.workers is not None else get_worker_count()
        if workers < 1:
            raise ValueError("workers must be positive")
        cases = design["cases"]
        if not args.all:
            index = args.task_id if args.task_id is not None else int(os.environ.get("PBS_ARRAY_INDEX", "-1"))
            if not 0 <= index < len(cases):
                raise ValueError(f"Select --task-id 0..{len(cases) - 1}, --all, or use PBS_ARRAY_INDEX")
            cases = [cases[index]]
        for case in cases:
            run_case(case, design, directory, workers=workers)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
