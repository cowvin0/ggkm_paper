"""Audit, independent precision pilot, and confirmatory latent-count study.

The existing exploratory workflow and its outputs are never overwritten.
Run with --help, or use scripts/run_confirmatory_workflow.py for PBS chains.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.metrics import integrated_brier_score, uno_c_index_rmst
from evaluate.simulated_data import cure_probability
from evaluate.util_cross_validation import _make_evaluation_grid, _select_grid_parameters, fit_cure_model
from scripts import confirmatory_robustness_utils as confirm
from scripts import latent_robustness_utils as original
from scripts import summarize_latent_robustness as exploratory
from scripts.confirmatory_robustness_fitting import MultiStartGGKM, profile_dispersion
from scripts.experiment_utils.artifacts import atomic_write_json, atomic_write_npz, finite_or_none, index_hash
from scripts.experiment_utils.runtime import get_worker_count
from scripts.run_latent_robustness_experiment import atomic_write_csv, replication_paths


def build_experiments(stage="final"):
    cases = original.build_experiments()
    return [case for case in cases if case["task_id"] in confirm.AUDIT_CASE_IDS] if stage == "audit" else cases


def _read_gate(path, identity_key, expected_id):
    if not path.is_file():
        raise ValueError(f"Missing prerequisite report: {path}")
    report = json.loads(path.read_text())
    if report.get(identity_key) != expected_id or not report.get("ready"):
        raise ValueError(f"Prerequisite is not ready or has a different protocol: {path}")
    if report.get("report_id") != original.identity({key: value for key, value in report.items() if key != "report_id"}):
        raise ValueError(f"Prerequisite report checksum differs: {path}")
    return report


def _write_gate(path, report):
    report["report_id"] = original.identity({key: value for key, value in report.items() if key != "report_id"})
    atomic_write_json(path, report)


def build_design(stage, root, config):
    if stage not in {"audit", "pilot", "final"}:
        raise ValueError("Expected audit, pilot, or final")
    prerequisite_ids = {}
    if stage != "audit":
        confirm.require_margins(config)
        gate = _read_gate(root / "audit/report.json", "protocol_id", confirm.protocol_snapshot(config, audit=True)["protocol_id"])
        prerequisite_ids["audit_report_id"] = gate["report_id"]
    snapshot = confirm.freeze_protocol(root, config, audit=stage == "audit")
    if stage == "final":
        plan = _read_gate(root / "pilot/allocation.json", "protocol_id", snapshot["protocol_id"])
        replications = plan["final_replications"]
        prerequisite_ids["allocation_report_id"] = plan["report_id"]
    else:
        replications = config[f"{stage}_replications"]
    phase = "pilot" if stage == "pilot" else "final"
    payload = {**snapshot["design"], **prerequisite_ids, "protocol_id": snapshot["protocol_id"], "stage": stage,
               "phase": phase, "replications": replications, "master_seed": config["seeds"][stage],
               "cases": build_experiments(stage),
               "confirmatory_protocol": snapshot["protocol"] if stage == "audit" else config}
    return {**payload, "study_id": original.identity(payload)}


def prepare_replication(case, replication, design, calibration):
    data, seeds = original.generate_replication(case, replication, design, calibration["censor_mean"])
    train, test = train_test_split(np.arange(case["n"]), stratify=data["event"], test_size=design["test_size"], random_state=seeds["split_seed"])
    time, event, X = (data[name] for name in ("time", "event", "X"))
    grid = _make_evaluation_grid(time[train], event[train], points=design["outer_grid_points"])
    scaler = StandardScaler().fit(X[train])
    axis = np.linspace(-1, 1, design["truth_grid_points"])
    x1, x2 = np.meshgrid(axis, axis, indexing="xy")
    truth_X = np.column_stack((x1.ravel(), x2.ravel()))
    tail_end = original.gengamma.ppf(0.99, a=original.GG_PARAMETERS["d"] / original.GG_PARAMETERS["p"], c=original.GG_PARAMETERS["p"], scale=original.GG_PARAMETERS["a"])
    tail_grid = np.linspace(0, tail_end, design["outer_grid_points"])
    context = {
        "X_train_raw": X[train], "X_train": scaler.transform(X[train]), "X_test": scaler.transform(X[test]),
        "train_time": time[train], "test_time": time[test], "train_event": event[train], "test_event": event[test],
        "true_cure_test": data["true_cure"][test], "time_grid": grid, "tail_time_grid": tail_grid,
        "X_truth_grid": scaler.transform(truth_X), "true_cure_grid": cure_probability(truth_X[:, 0], truth_X[:, 1], case["method"]),
        "true_survival_grid": original.true_survival(truth_X, grid, case),
        "true_tail_survival": original.true_survival(truth_X, tail_grid, case),
    }
    base = {**case, **seeds, "replication": replication, "study_id": design["study_id"], "phase": design["phase"],
            "kernel": "rbf", "censor_mean": calibration["censor_mean"], "data_hash": original.array_hash(data),
            "train_index_hash": index_hash(train), "test_index_hash": index_hash(test), "horizon": float(grid[-1]),
            "time_grid_hash": original.array_hash({"grid": grid}), "training_size": len(train), "test_size": len(test)}
    arrays = {**data, "train_indices": train, "test_indices": test, "time_grid": grid,
              "tail_time_grid": tail_grid, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}
    return context, base, arrays


def fit_one(model_name, mode, case, design, context, base, directory):
    start = perf_counter()
    protocol = design["confirmatory_protocol"]
    row = {**base, "model_name": model_name, "latency_mode": mode, "status": "failed", "converged": False,
           "failure_reason": None, **{metric: None for metric in original.METRICS}}
    parameters = dict(design["estimator_parameters"][model_name])
    fixed_latency = [design["gg_parameters"][key] for key in ("a", "d", "p")] if mode == "fixed" else None
    if mode == "phi_fixed":
        parameters["phi"] = original.truth_spec(case["truth"])["phi"]
    journal = []

    def factory(**tuned):
        fit = MultiStartGGKM(fitting=protocol["fitting"], fixed_latency=fixed_latency, **{**parameters, **tuned})
        journal.append(fit)
        return fit

    diagnostic = {"study_id": design["study_id"], "profile": []}
    try:
        time, event = context["train_time"], context["train_event"]
        tuned, inner_score = _select_grid_parameters(
            context["X_train_raw"], time, event, factory, "rbf", design["parameter_grids"][model_name],
            n_inner_splits=design["inner_splits"], t_grid_points=design["inner_grid_points"], random_state=base["inner_seed"],
        )
        fit = factory(kernel="rbf", **tuned)
        fit_cure_model(fit, context["X_train"], time, event)
        model = fit.model_
        grid = context["time_grid"]
        survival = np.asarray(fit.predict_survival(context["X_test"], grid))
        cure = np.asarray(fit.predict_cure_probability(context["X_test"]))
        if not np.isfinite(survival).all() or not np.isfinite(cure).all():
            raise FloatingPointError("Nonfinite held-out predictions")
        ibs = integrated_brier_score(survival, context["test_time"], context["test_event"], time, event, grid)
        if not np.isfinite(ibs):
            raise FloatingPointError("Nonfinite held-out IBS")
        cure_error = cure - context["true_cure_test"]
        row.update(integrated_brier=float(ibs), cure_rmse=float(np.sqrt(np.mean(cure_error ** 2))),
                   cure_mse=float(np.mean(cure_error ** 2)), cure_bias=float(cure_error.mean()),
                   uno_c_index=finite_or_none(uno_c_index_rmst(survival, context["test_time"], context["test_event"], time, event, grid, tau=grid[-1])))
        grid_cure = np.asarray(fit.predict_cure_probability(context["X_truth_grid"]))
        row["grid_cure_rmse"] = float(np.sqrt(np.mean((grid_cure - context["true_cure_grid"]) ** 2)))
        for name, true_key, grid_key in (("survival_rmse", "true_survival_grid", "time_grid"), ("tail_survival_rmse", "true_tail_survival", "tail_time_grid")):
            times = context[grid_key]
            prediction = np.asarray(fit.predict_survival(context["X_truth_grid"], times))
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite survival-grid predictions")
            mse = np.mean((prediction - context[true_key]) ** 2, axis=0)
            row[name] = float(np.sqrt(np.trapezoid(mse, times) / (times[-1] - times[0])))
        row.update(status="ok", converged=bool(model.converged_), n_iter=int(model.n_iter_),
                   fitted_a=float(model.a_), fitted_d=float(model.d_), fitted_p=float(model.p_),
                   fitted_phi=finite_or_none(getattr(model, "phi_", None)), final_loglik=float(model.final_loglik_),
                   best_inner_ibs=float(inner_score), best_params=json.dumps(tuned, sort_keys=True))
        if fixed_latency is not None and [model.a_, model.d_, model.p_] != fixed_latency:
            raise RuntimeError("Fixed latency changed")
        if mode == "phi_fixed" and model.phi_ != parameters["phi"]:
            raise RuntimeError("Fixed dispersion changed")
        diagnostic.update(selected_attempt=fit.selected_attempt_, selected_final_loglik=float(model.final_loglik_))
        if design["stage"] == "audit" and model_name == "bn" and mode != "phi_fixed":
            diagnostic["profile"] = fit.profile_ or profile_dispersion(
                {**parameters, **tuned}, protocol["fitting"], context["X_train"], time, event, fixed_latency=fixed_latency,
            )
    except Exception as error:
        row.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
    diagnostic["fits"] = [{"parameters": item.parameters, "fixed_latency": item.fixed_latency,
                            "attempts": item.attempts_, "profile": item.profile_,
                            "selected_attempt": getattr(item, "selected_attempt_", None)} for item in journal]
    path = Path(directory) / "diagnostics" / f"case_{case['task_id']:03d}" / f"replication_{base['replication']:04d}_{mode}_{model_name}.json"
    atomic_write_json(path, diagnostic)
    row.update(diagnostic_path=str(path.relative_to(directory)), diagnostic_sha256=original.checksum(path), runtime_seconds=perf_counter() - start)
    row["start_attempts"] = sum(len(fit.attempts_) for fit in journal)
    return row


def run_replication(case, replication, design, directory, calibration):
    checkpoint, archive = replication_paths(directory, case["task_id"], replication)
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("study_id") != design["study_id"] or saved.get("replication") != replication or saved.get("task_id") != case["task_id"]:
            raise ValueError(f"Mismatched checkpoint: {checkpoint}")
        if len(saved["rows"]) != len(confirm.variants(case)) or {(r["latency_mode"], r["model_name"]) for r in saved["rows"]} != set(confirm.variants(case)):
            raise ValueError(f"Incomplete checkpoint: {checkpoint}")
        if saved["sample_sha256"] is not None:
            if not archive.is_file() or original.checksum(archive) != saved["sample_sha256"]:
                raise ValueError(f"Modified sample: {archive}")
        elif not saved.get("preparation_failed"):
            raise ValueError("Unverifiable sample checkpoint")
        for row in saved["rows"]:
            if "diagnostic_path" in row and original.checksum(Path(directory) / row["diagnostic_path"]) != row["diagnostic_sha256"]:
                raise ValueError("Modified fit diagnostics")
        return saved["rows"]
    preparation_failed, sample_checksum = False, None
    try:
        context, base, arrays = prepare_replication(case, replication, design, calibration)
        atomic_write_npz(archive, **arrays)
        sample_checksum = original.checksum(archive)
    except Exception as error:
        preparation_failed = True
        seeds = original.replication_seeds(case, replication, design["master_seed"], phase=design["phase"])
        rows = [{**case, **seeds, "replication": replication, "study_id": design["study_id"], "phase": design["phase"],
                 "latency_mode": mode, "model_name": model, "status": "failed", "converged": False,
                 "failure_reason": f"Preparation: {type(error).__name__}: {error}", "runtime_seconds": 0.,
                 **{metric: None for metric in original.METRICS}} for mode, model in confirm.variants(case)]
    if not preparation_failed:
        rows = [fit_one(model, mode, case, design, context, base, directory) for mode, model in confirm.variants(case)]
    atomic_write_json(checkpoint, {"study_id": design["study_id"], "task_id": case["task_id"], "replication": replication,
                                   "rows": rows, "sample_sha256": sample_checksum, "preparation_failed": preparation_failed})
    return rows


def run_case(case, design, directory, workers):
    original.write_design(directory, design)
    calibration = original.calibrate_censoring(case, sample_size=design["calibration_size"])
    atomic_write_json(directory / "calibration" / f"case_{case['task_id']:03d}.json", calibration)
    records = []
    if workers == 1:
        for replication in range(design["replications"]):
            records.extend(run_replication(case, replication, design, directory, calibration))
            print(f"{design['stage']} case {case['task_id']}: replication {replication + 1}/{design['replications']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_replication, case, rep, design, directory, calibration) for rep in range(design["replications"])]
            for future in as_completed(futures):
                records.extend(future.result())
    frame = pd.DataFrame(records).sort_values(["replication", "latency_mode", "model_name"])
    path = directory / f"replications_case_{case['task_id']:03d}.csv"
    atomic_write_csv(path, frame)
    atomic_write_json(path.with_suffix(".metadata.json"), {"study_id": design["study_id"], "task_id": case["task_id"], "csv_sha256": original.checksum(path)})
    print(f"Saved {path}", flush=True)


def read_results(stage, root, config):
    design = build_design(stage, root, config)
    directory = root / stage
    if json.loads((directory / "design.json").read_text()) != json.loads(json.dumps(design)):
        raise ValueError("Saved design differs from the frozen current study")
    frames = []
    for case in design["cases"]:
        path = directory / f"replications_case_{case['task_id']:03d}.csv"
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        if metadata.get("study_id") != design["study_id"] or metadata.get("task_id") != case["task_id"] or metadata.get("csv_sha256") != original.checksum(path):
            raise ValueError(f"Mismatched case output: {path}")
        frame = pd.read_csv(path)
        for key, value in {**case, "study_id": design["study_id"], "phase": design["phase"]}.items():
            if not (frame[key].isna() if value is None else frame[key].eq(value)).all():
                raise ValueError(f"Case {case['task_id']}: mismatched {key}")
        convergence = frame.converged.astype("string").str.lower()
        if not convergence.isin(["true", "false"]).all() or not frame.status.isin(["ok", "failed"]).all():
            raise ValueError("Invalid convergence/status records")
        frame["converged"] = convergence.eq("true")
        if set(frame.replication) != set(range(design["replications"])):
            raise ValueError("Missing replication IDs")
        for replication, rows in frame.groupby("replication"):
            if len(rows) != len(confirm.variants(case)) or set(zip(rows.latency_mode, rows.model_name)) != set(confirm.variants(case)):
                raise ValueError("Missing or duplicate fitted variants")
            seeds = original.replication_seeds(case, int(replication), design["master_seed"], phase=design["phase"])
            if any(not rows[name].eq(value).all() for name, value in seeds.items()):
                raise ValueError("Unpaired or incorrect seeds")
            if rows.status.eq("ok").any():
                for column in exploratory.PAIR_CHECKS:
                    if column not in rows or rows[column].isna().any() or rows[column].nunique() != 1:
                        raise ValueError(f"Unpaired {column}")
            # Verify samples and full fit histories, not just summary CSVs.
            checkpoint_rows = run_replication(case, int(replication), design, directory, {})
            expected = pd.DataFrame(checkpoint_rows).sort_values(["latency_mode", "model_name"])
            observed = rows.sort_values(["latency_mode", "model_name"])
            for metric in (*original.METRICS, "runtime_seconds"):
                np.testing.assert_allclose(pd.to_numeric(observed[metric]), pd.to_numeric(expected[metric]), rtol=1e-12, atol=1e-12, equal_nan=True)
            for column in ("status", "converged"):
                if list(observed[column]) != list(expected[column]):
                    raise ValueError(f"CSV and checkpoint differ in {column}")
        if not np.isfinite(frame.runtime_seconds).all() or (frame.runtime_seconds < 0).any():
            raise ValueError("Invalid runtimes")
        for metric in exploratory.SECONDARY_METRICS:
            if metric not in frame:
                frame[metric] = np.nan
        frames.append(frame)
    return design, pd.concat(frames, ignore_index=True)


def audit_summary(root, config, *, figures=True):
    design, frame = read_results("audit", root, config)
    problems, profiles = [], []
    if not frame.status.eq("ok").all():
        problems.append("Some audit fits failed; inspect diagnostics")
    if not frame.converged.all():
        problems.append("Some selected audit fits did not converge after prescribed retries")
    for row in frame.to_dict("records"):
        if "diagnostic_path" not in row or pd.isna(row["diagnostic_path"]):
            continue
        diagnostic = json.loads((root / "audit" / row["diagnostic_path"]).read_text())
        profile = diagnostic.get("profile", [])
        if row["model_name"] == "bn" and row["latency_mode"] != "phi_fixed":
            if len(profile) != len(config["fitting"]["profile_dispersion"]) or any(point["status"] != "ok" or not point["converged"] for point in profile):
                problems.append(f"Incomplete/nonconverged profile: case {row['task_id']}, replicate {row['replication']}, {row['latency_mode']}")
            for point in profile:
                improvement = None if point["final_loglik"] is None else point["final_loglik"] - row["final_loglik"]
                profiles.append({**{key: row[key] for key in ("task_id", "truth", "replication", "latency_mode")},
                                 "phi": point["phi"], "final_loglik": point["final_loglik"], "converged": point["converged"],
                                 "profile_improvement_over_free": improvement, "fitted_latency": json.dumps(point.get("fitted_latency"))})
                if improvement is not None and improvement > config["fitting"]["profile_improvement_tolerance"]:
                    problems.append(f"Profile improves on free NB: case {row['task_id']}, replicate {row['replication']}, {row['latency_mode']}, phi={point['phi']}")
    atomic_write_csv(root / "audit/dispersion_profiles.csv", pd.DataFrame(profiles))
    atomic_write_csv(root / "audit/performance_summary.csv", exploratory.performance_summary(frame))
    report = {"protocol_id": design["protocol_id"], "study_id": design["study_id"], "ready": not problems,
              "problems": problems, "fits": len(frame), "converged": int(frame.converged.sum()),
              "interpretation": "phi->0 is the Poisson limit and is not itself a failure. This gate checks fit completion, convergence, and whether fixed-phi/Poisson profile fits expose a better objective. A passed gate is not proof of a global maximum or identification."}
    _write_gate(root / "audit/report.json", report)
    if figures and profiles:
        plot_profiles(pd.DataFrame(profiles), root / "audit")
    return report


def plot_profiles(profiles, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (case_id, mode), rows in profiles.groupby(["task_id", "latency_mode"]):
        fig, axis = plt.subplots(figsize=(7, 4))
        for replication, points in rows.groupby("replication"):
            axis.plot(points.phi, points.profile_improvement_over_free, ".-", alpha=.7, label=f"Replication {replication}")
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_xscale("symlog", linthresh=0.0001)
        axis.set_xlabel(r"Fixed $\phi$ (0 is exact Poisson)")
        axis.set_ylabel("Profile objective minus free-NB objective")
        axis.set_title(f"Case {case_id}, {rows.iloc[0].truth}, {mode} latency")
        axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(directory / f"dispersion_profile_case_{case_id:03d}_{mode}.pdf")
        plt.close(fig)


def summarize(stage, root, config, *, figures=True):
    design, frame = read_results(stage, root, config)
    pairs = confirm.paired_values(frame)
    summary = confirm.contrast_summary(frame, pairs, config)
    directory = root / stage
    atomic_write_csv(directory / "paired_replications.csv", pairs)
    atomic_write_csv(directory / "paired_summary.csv", summary)
    atomic_write_csv(directory / "margin_sensitivity.csv", confirm.margin_sensitivity(summary))
    performance = exploratory.performance_summary(frame)
    atomic_write_csv(directory / "performance_summary.csv", performance)
    primary = summary.loc[summary.primary]
    if stage == "pilot":
        projection, report = confirm.allocation(summary, config)
        atomic_write_csv(directory / "allocation_contrasts.csv", projection)
        report.update(protocol_id=design["protocol_id"], pilot_study_id=design["study_id"])
        if frame.loc[frame.study_part.eq("primary") & frame.latency_mode.eq("estimated"), "status"].ne("ok").any():
            report.update(ready=False, problem="Primary pilot fitting failures require investigation")
        _write_gate(directory / "allocation.json", report)
        return report
    expected = confirm.FAMILY_SIZE
    full = primary.loc[primary.analysis.eq("all_finite")]
    sensitivity = primary.loc[primary.analysis.eq("jointly_converged")]
    main_pass = len(full) == expected // 2 and full.assessment.eq("within_margin").all()
    sensitivity_pass = len(sensitivity) == expected // 2 and sensitivity.assessment.eq("within_margin").all()
    if len(primary) != expected or primary.assessment.eq("incomplete_pairs").any():
        decision = "incomplete"
    elif primary.assessment.eq("exceeds_margin").any():
        decision = "exceeds_margin"
    elif main_pass and sensitivity_pass:
        decision = "within_prespecified_margins"
    else:
        decision = "inconclusive"
    report = {"protocol_id": design["protocol_id"], "study_id": design["study_id"], "stage": stage,
              "replications": design["replications"], "decision": decision,
              "all_finite_pass": bool(main_pass), "jointly_converged_pass": bool(sensitivity_pass),
              "sensitivity_disagreement": bool(main_pass != sensitivity_pass), "primary_comparisons": len(primary),
              "alpha": config["alpha"], "margins": config["margins"], "margin_justification": config["margin_justification"],
              "attempted": len(frame), "failed": int(frame.status.ne("ok").sum()), "converged": int(frame.converged.sum()),
              "minimum_jointly_converged_fraction": float((sensitivity.valid / sensitivity.attempted_pairs).min()),
              "interpretation": "Decision concerns mean paired IBS/cure-RMSE deterioration in the 40 prespecified primary cells, including the NB-under-Poisson limiting-family control. Both all-finite and jointly-converged analyses must pass. Secondary/fixed-parameter/mixture diagnostics do not establish noninferiority. No universal robustness or count-law identification claim."}
    atomic_write_json(directory / "report.json", report)
    # Retain diagnostic compensation comparisons; they do not enter the NI family.
    selected_pairs = pairs.loc[pairs.analysis.eq("all_finite")]
    atomic_write_csv(directory / "latency_compensation.csv", exploratory.latency_compensation(selected_pairs))
    controls = frame.loc[frame.latency_mode.eq("phi_fixed")].merge(
        frame.loc[frame.latency_mode.eq("estimated") & frame.model_name.eq("bn")],
        on=["task_id", "replication"], suffixes=("_fixed_phi", "_estimated_phi"), validate="one_to_one")
    atomic_write_csv(directory / "fixed_dispersion_controls.csv", controls)
    write_summary_artifacts(directory, summary, report, figures=figures)
    return report


def write_summary_artifacts(directory, summary, report, *, figures=True):
    lines = ["# Confirmatory latent-count robustness study", "", f"Decision: **{report['decision']}**", "",
             "Positive deterioration is worse. The primary decision uses simultaneous one-sided bounds for IBS and cure RMSE in all 40 primary cells and both analysis sets.", "",
             f"Margins: {json.dumps(report['margins'])}. Justification: {report['margin_justification']}", "",
             "NB approaches Poisson as dispersion approaches zero. Similar fitted predictions do not establish recovery of the latent-count law.", "",
             "See paired_summary.csv for simultaneous bounds, performance_summary.csv for absolute error/convergence, fixed_dispersion_controls.csv and latency_compensation.csv for diagnostics.", ""]
    primary = summary.loc[summary.primary].copy()
    lines += ["| Analysis | Metric | Largest mean loss | Largest upper bound | Margin |", "|---|---|---:|---:|---:|"]
    for (analysis, metric), rows in primary.groupby(["analysis", "metric"]):
        mean, upper = rows["mean"].max(), rows.simultaneous_upper.max()
        text_mean = f"{mean:.6f}" if pd.notna(mean) else "--"
        text_upper = f"{upper:.6f}" if pd.notna(upper) else "--"
        lines.append(f"| {analysis} | {metric} | {text_mean} | {text_upper} | {report['margins'][metric]} |")
    lines += ["", "The largest mean and largest bound can arise in different cells. See the CSV for cell identities and retained pair counts.", ""]
    (directory / "conclusion.md").write_text("\n".join(lines))
    columns = ["truth", "n", "method", "followup", "model_name", "analysis", "metric", "mean", "mcse", "simultaneous_upper", "margin", "valid", "assessment"]
    primary[columns].to_latex(directory / "confirmatory-contrasts.tex", index=False, float_format="%.6f", na_rep="--")
    if not figures:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in confirm.PRIMARY_METRICS:
        selected = primary.loc[primary.metric.eq(metric)]
        fig, axes = plt.subplots(1, 2, figsize=(13, 15), sharex=True, sharey=True)
        for axis, analysis in zip(axes, confirm.ANALYSES, strict=True):
            rows = selected.loc[selected.analysis.eq(analysis)].sort_values(["task_id", "model_name"])
            y = np.arange(len(rows))
            good = rows[["mean", "simultaneous_lower", "simultaneous_upper"]].notna().all(axis=1)
            visible = rows.loc[good]
            if not visible.empty:
                axis.errorbar(visible["mean"], y[good], xerr=[visible["mean"] - visible.simultaneous_lower, visible.simultaneous_upper - visible["mean"]], fmt=".", markersize=4, linewidth=0.7)
            axis.axvline(report["margins"][metric], color="red", linestyle="--", label="Acceptable deterioration")
            axis.axvline(0, color="gray", linewidth=0.6)
            axis.set_yticks(y, [f"{int(row.task_id)}: {row.truth} / {row.model_name}" for row in rows.itertuples()], fontsize=5)
            axis.set_title(analysis.replace("_", " "))
            axis.set_xlabel("Mean deterioration (simultaneous MC bounds)")
            axis.legend(fontsize=8)
        fig.suptitle(f"Confirmatory {metric}: both analyses must lie below the margin")
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(directory / f"confirmatory_{metric}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def review_historical_margins(path, directory, config):
    """Explain scale/tolerance tradeoffs using old summaries, without NI claims."""
    frame = pd.read_csv(path)
    required = {"latency_mode", "specification", "metric", "mean", "sd", "valid", "attempted_pairs"}
    if not required.issubset(frame):
        raise ValueError(f"Missing historical summary columns: {sorted(required.difference(frame.columns))}")
    selected = frame.loc[frame.latency_mode.eq("estimated") & frame.specification.ne("correct") & frame.metric.isin(confirm.PRIMARY_METRICS)].copy()
    if len(selected) != 160:
        raise ValueError("Historical planning review requires all 160 estimated-latency primary metric comparisons")
    selected["primary"] = True
    selected["analysis"] = "historical_all_finite"
    # Use the same conservative 320-comparison bound construction as the new
    # protocol, while explicitly lacking its jointly-converged sensitivity.
    from scipy.stats import t as student_t
    critical = student_t.ppf(1 - config["alpha"] / confirm.FAMILY_SIZE, selected.valid - 1)
    selected["simultaneous_upper"] = selected["mean"] + critical * selected.sd / np.sqrt(selected.valid)
    selected.loc[selected.valid < 2, "simultaneous_upper"] = np.nan
    sensitivity = confirm.margin_sensitivity(selected)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(directory / "historical_bounds.csv", selected)
    atomic_write_csv(directory / "margin_sensitivity.csv", sensitivity)
    lines = ["# Exploratory review of practical margins", "",
             "These are historical planning results, not the new confirmatory experiment. The NB fitting questions remain unresolved, and jointly-converged pairs cannot be recovered from the supplied summary files.", "",
             "The margins below are illustrative sensitivity values, not accepted scientific defaults. Choosing a margin because it passes this table would not justify a strong robustness claim.", "",
             "| Metric | Candidate absolute margin | Comparisons below margin | Comparisons | Largest upper MC bound |",
             "|---|---:|---:|---:|---:|"]
    for row in sensitivity.itertuples():
        lines.append(f"| {row.metric} | {row.illustrative_margin:g} | {row.within_margin} | {row.comparisons} | {row.largest_upper_bound:.6f} |")
    lines += ["", "How to choose a margin:", "",
              "1. Cure RMSE is on the probability scale. An increase from 0.08 to 0.09 is an increase of one percentage point in RMSE; it is not a one-percentage-point change for every patient.",
              "2. IBS is an average IPCW squared prediction error. Its margin is not a probability-point tolerance. Use an independently justified benchmark improvement or a prediction-error tolerance to anchor it.",
              "3. One possible benchmark rule is to preserve a prespecified fraction of a useful predictive improvement over a simpler model. If an independent benchmark gain is 0.01 IBS and preserving 80% is scientifically required, the allowable loss is 0.002. If the cure-RMSE gain is 0.05, that same illustrative preservation rule gives 0.01. These are hypothetical examples, not adopted margins.",
              "4. Agree the acceptable loss and its justification with the supervisor, then freeze it before the new independent pilot/final results. If no margin can be justified, report the loss bounds and avoid a binary practical-robustness verdict.", ""]
    (directory / "margin_review.md").write_text("\n".join(lines))
    print((directory / "margin_review.md").read_text())
    return sensitivity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "audit-summary", "pilot", "pilot-summary", "final", "report", "margin-review"), default="audit")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task-id", type=int, help="Array position, not the generating case ID")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--exploratory-summary", type=Path, default=original.PROJECT_ROOT / "tmp/paired_summary.csv")
    args = parser.parse_args(argv)
    try:
        config = confirm.load_protocol(args.protocol)
        root = (args.output_dir or confirm.results_root()).resolve()
        if args.stage == "margin-review":
            review_historical_margins(args.exploratory_summary, root / "margin_review", config)
            return 0
        if args.stage in {"audit-summary", "pilot-summary", "report"}:
            report = audit_summary(root, config, figures=not args.no_figures) if args.stage == "audit-summary" else summarize("pilot" if args.stage == "pilot-summary" else "final", root, config, figures=not args.no_figures)
            print(json.dumps(report, indent=2))
            return 0 if args.stage == "report" or report["ready"] else 2
        design = build_design(args.stage, root, config)
        cases = design["cases"]
        if not args.all:
            task = args.task_id if args.task_id is not None else int(os.environ.get("PBS_ARRAY_INDEX", "-1"))
            if not 0 <= task < len(cases):
                raise ValueError(f"Choose array --task-id 0..{len(cases)-1} or --all")
            cases = [cases[task]]
        workers = args.workers if args.workers is not None else get_worker_count()
        if workers < 1:
            raise ValueError("workers must be positive")
        for case in cases:
            run_case(case, design, root / args.stage, workers)
        return 0
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
