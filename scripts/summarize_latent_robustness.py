"""Validate paired robustness experiments and export precision, tables, or plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import latent_robustness_utils as study
from scripts.experiment_utils.artifacts import atomic_write_json, finite_or_none
from scripts.run_latent_robustness_experiment import atomic_write_csv, expected_fits


KEYS = ["task_id", "truth", "study_part", "n", "method", "followup", "latency_mode"]
PAIR_CHECKS = (*study.SEED_COLUMNS, "data_hash", "train_index_hash", "test_index_hash", "horizon", "time_grid_hash")
SECONDARY_METRICS = ("cure_bias", "cure_mse", "grid_cure_rmse", "grid_cure_bias", "tail_survival_rmse", "cure_auc", "fitted_a", "fitted_d", "fitted_p", "fitted_phi")
MODEL_LABELS = {"poisson": "Poisson", "bn": "Negative binomial", "binomial": "Binomial (K=2)"}
TRUTH_LABELS = {"poisson": "Poisson", "nb025": "NB (phi=0.25)", "nb100": "NB (phi=1)", "nb200": "NB (phi=2)", "binomial2": "Binomial (K=2)", "poisson_mixture": "Poisson mixture"}
METRIC_LABELS = {"integrated_brier": "IPCW IBS", "cure_rmse": "Cure RMSE", "survival_rmse": "Survival RMSE", "uno_c_index": "Uno's C-index"}


def read_results(directory, *, require_complete=True):
    directory = Path(directory)
    saved = json.loads((directory / "design.json").read_text())
    current = study.build_design(saved["phase"], replications=saved["replications"], seed=saved["master_seed"])
    if current["study_id"] != saved["study_id"] or study.identity({key: value for key, value in saved.items() if key != "study_id"}) != saved["study_id"]:
        raise ValueError("Saved robustness design/source/software differs from the current study")
    frames, missing = [], []
    for case in saved["cases"]:
        path = directory / f"replications_case_{case['task_id']:03d}.csv"
        metadata_path = path.with_suffix(".metadata.json")
        if not path.exists() or not metadata_path.exists():
            missing.append(case["task_id"])
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("study_id") != saved["study_id"] or metadata.get("task_id") != case["task_id"] or metadata.get("csv_sha256") != study.checksum(path):
            raise ValueError(f"Mismatched case identity or checksum: {path}")
        frame = pd.read_csv(path, dtype={column: "string" for column in PAIR_CHECKS if column.endswith("hash")})
        validate_frame(frame, case, saved)
        frames.append(frame)
    if missing and require_complete:
        raise ValueError(f"Missing robustness case outputs: {missing}")
    return saved, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), missing


def validate_frame(frame, case, design):
    required = {"study_id", "phase", "replication", "model_name", "latency_mode", "status", "converged", "runtime_seconds", *study.SEED_COLUMNS, *case, *study.METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Case {case['task_id']}: missing columns {sorted(missing)}")
    for column in ("replication", "task_id", *study.SEED_COLUMNS, *study.METRICS, "runtime_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    ids = frame["replication"]
    if not np.isfinite(ids).all() or (ids % 1 != 0).any() or set(ids) != set(range(design["replications"])):
        raise ValueError(f"Case {case['task_id']}: incomplete or invalid replication IDs")
    for column, value in {**case, "study_id": design["study_id"], "phase": design["phase"]}.items():
        if value is None:
            valid = frame[column].isna()
        else:
            valid = frame[column].eq(value)
        if not valid.all():
            raise ValueError(f"Case {case['task_id']}: {column} differs from the declared design")
    if not frame.status.isin(["ok", "failed"]).all():
        raise ValueError("Every attempt must have status ok or failed")
    convergence = frame.converged.astype("string").str.lower()
    if not convergence.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError("Every fit must record convergence explicitly")
    frame["converged"] = convergence.isin(["true", "1", "1.0"])
    if (frame.runtime_seconds < 0).any() or not np.isfinite(frame.runtime_seconds).all():
        raise ValueError("Every attempt must record a finite nonnegative runtime")
    expected = expected_fits(case)
    for replication, rows in frame.groupby("replication"):
        if len(rows) != len(expected) or set(zip(rows.latency_mode, rows.model_name)) != expected:
            raise ValueError(f"Case {case['task_id']}, replication {replication}: missing or duplicate fits")
        seeds = study.replication_seeds(case, int(replication), design["master_seed"], phase=design["phase"])
        for column, value in seeds.items():
            if not rows[column].eq(value).all():
                raise ValueError(f"Case {case['task_id']}: wrong shared {column}")
        if rows.status.eq("ok").any():
            for column in PAIR_CHECKS:
                if column not in rows or rows[column].isna().any() or rows[column].nunique() != 1:
                    raise ValueError(f"Case {case['task_id']}, replication {replication}: unpaired {column}")
    for metric in SECONDARY_METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise") if metric in frame else np.nan


def descriptive(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    sd = float(np.std(values, ddof=1)) if count > 1 else np.nan
    mean = float(np.mean(values)) if count else np.nan
    mcse = sd / np.sqrt(count) if count > 1 else np.nan
    return {"valid": count, "mean": finite_or_none(mean), "sd": finite_or_none(sd), "mcse": finite_or_none(mcse),
            "lower_mc95": finite_or_none(mean - 1.96 * mcse), "upper_mc95": finite_or_none(mean + 1.96 * mcse),
            "upper_one_sided_mc95": finite_or_none(mean + 1.6448536269514722 * mcse)}


def performance_summary(frame):
    records = []
    for keys, rows in frame.groupby([*KEYS, "model_name"], sort=False):
        good = rows.loc[rows.status.eq("ok")]
        record = dict(zip([*KEYS, "model_name"], keys, strict=True))
        record.update(attempted=len(rows), successful=len(good), failed=len(rows) - len(good), converged=int(good.converged.sum()),
                      runtime_mean=float(rows.runtime_seconds.mean()), runtime_max=float(rows.runtime_seconds.max()))
        for metric in (*study.METRICS, *SECONDARY_METRICS):
            for name, value in descriptive(good[metric]).items():
                record[f"{metric}_{name}"] = value
        records.append(record)
    return pd.DataFrame(records)


def paired_replications(frame):
    records = []
    for keys, rows in frame.groupby(KEYS, sort=False):
        metadata = dict(zip(KEYS, keys, strict=True))
        reference = study.truth_spec(metadata["truth"])["reference_model"]
        if reference is None:
            continue  # No fitted candidate is the correct family for mixture truth.
        ref = rows.loc[rows.model_name.eq(reference)].set_index("replication")
        for model in study.FIT_MODELS:
            focal = rows.loc[rows.model_name.eq(model)].set_index("replication")
            for metric in study.METRICS:
                good = focal.status.eq("ok") & ref.status.eq("ok") & np.isfinite(focal[metric]) & np.isfinite(ref[metric])
                difference = focal.loc[good, metric] - ref.loc[good, metric]
                if metric == "uno_c_index":
                    difference = -difference
                for replication, value in difference.items():
                    records.append({**metadata, "model_name": model, "reference_model": reference,
                                    "metric": metric, "replication": replication, "deterioration": float(value)})
    return pd.DataFrame(records)


def paired_summary(frame, pairs, *, margins=None):
    margins = margins or {}
    records = []
    for keys, rows in frame.groupby([*KEYS, "model_name"], sort=False):
        metadata = dict(zip([*KEYS, "model_name"], keys, strict=True))
        reference = study.truth_spec(metadata["truth"])["reference_model"]
        if reference is None:
            continue
        selected = pairs
        for name, value in metadata.items():
            selected = selected.loc[selected[name].eq(value)] if not selected.empty else selected
        for metric in study.METRICS:
            values = selected.loc[selected.metric.eq(metric), "deterioration"] if not selected.empty else []
            stats = descriptive(values)
            margin = margins.get(metric)
            robust = None if margin is None or stats["upper_one_sided_mc95"] is None else stats["upper_one_sided_mc95"] < margin
            records.append({**metadata, "reference_model": reference, "metric": metric,
                            "specification": "correct" if metadata["model_name"] == reference else (
                                "poisson_limit" if metadata["truth"] == "poisson" and metadata["model_name"] == "bn" else "misspecified"),
                            "attempted_pairs": len(rows), **stats, "margin": margin,
                            "within_margin": robust, "assessment": "not_assessed" if robust is None else ("within_margin" if robust else "not_established")})
    return pd.DataFrame(records)


def latency_compensation(pairs):
    if pairs.empty:
        return pd.DataFrame()
    keys = [name for name in KEYS if name != "latency_mode"] + ["model_name", "reference_model", "metric", "replication"]
    free = pairs.loc[pairs.latency_mode.eq("estimated")]
    fixed = pairs.loc[pairs.latency_mode.eq("fixed")]
    paired = free.merge(fixed, on=keys, suffixes=("_estimated", "_fixed"), validate="one_to_one")
    paired["estimated_minus_fixed_penalty"] = paired.deterioration_estimated - paired.deterioration_fixed
    group_keys = [name for name in keys if name != "replication"]
    return pd.DataFrame([
        {**dict(zip(group_keys, values, strict=True)), **descriptive(rows.estimated_minus_fixed_penalty)}
        for values, rows in paired.groupby(group_keys, sort=False)
    ])


def pilot_precision(design, summary, missing, *, final_replications=None, ibs_target=0.003, cure_rmse_target=0.005):
    final_replications = study.configured_final_replications() if final_replications is None else final_replications
    if final_replications < 2 or not all(np.isfinite(value) and value > 0 for value in (ibs_target, cure_rmse_target)):
        raise ValueError("Final replications must be at least two and MCSE targets positive")
    targets = {"integrated_brier": ibs_target, "cure_rmse": cure_rmse_target}
    projections = []
    problems = [f"Missing case {case}" for case in missing]
    if not summary.empty:
        for row in summary.to_dict("records"):
            if row["metric"] not in targets or row["specification"] == "correct":
                continue
            target = targets[row["metric"]]
            if row["valid"] < 2 or row["sd"] is None or not np.isfinite(row["sd"]):
                problems.append(f"Case {row['task_id']}/{row['latency_mode']}/{row['model_name']}/{row['metric']}: fewer than two finite pairs")
                required, projected = None, None
            else:
                fraction = row["valid"] / row["attempted_pairs"]
                required = max(2, math.ceil((row["sd"] / target) ** 2 / fraction))
                projected = row["sd"] / np.sqrt(final_replications * fraction)
            projections.append({**{name: row[name] for name in (*KEYS, "model_name", "metric", "valid", "sd", "attempted_pairs")},
                                "target_mcse": target, "projected_mcse": projected, "required_replications": required})
    if not projections:
        problems.append("No paired pilot precision measurements")
    required = max((row["required_replications"] for row in projections if row["required_replications"] is not None), default=None)
    complete = not problems
    ready = complete and required is not None and required <= final_replications
    return pd.DataFrame(projections), {
        "study_id": design["study_id"], "phase": "pilot", "complete": complete, "ready_for_final": ready,
        "decision": "ready" if ready else ("incomplete" if not complete else "insufficient_precision"),
        "final_replications": final_replications, "required_replications": required,
        "targets": targets, "problems": problems,
        "allocation_rule": "ceil((paired SD / target MCSE)^2 / pilot finite-pair fraction), maximum across contrasts",
        "note": "Independent pilot samples are excluded from final estimates. R is not changed automatically. Margins remain unset unless specified.",
    }


def _estimate(row, metric=None):
    mean = row.get(f"{metric}_mean" if metric else "mean")
    mcse = row.get(f"{metric}_mcse" if metric else "mcse")
    if mean is None or not np.isfinite(mean):
        return "--"
    detail = f"{mcse:.4f}" if mcse is not None and np.isfinite(mcse) else "--"
    return f"{mean:.4f} ({detail})"


def render_tables(performance, contrasts, design):
    performance_lines, paired_lines = [], []
    for keys, rows in performance.groupby(["latency_mode", "method", "n", "followup"], sort=False):
        mode, method, n, followup = keys
        label = f"{mode}-{method}-{n}-{followup}"
        caption = (f"Latent-count robustness: {'quadratic' if method == 2 else 'periodic'} cure surface, "
                   f"$n={n}$, {followup} follow-up, {mode} generalized gamma parameters. "
                   f"{design['replications']} planned replications per generating scenario. "
                   "Values are successful finite means (Monte Carlo standard errors); valid counts are retained in the CSV summaries. "
                   "Convergence is converged/successful; failures include every unsuccessful attempt. "
                   "Poisson-mixture truth is outside all three fitted count families.")
        performance_lines.extend([
            r"\begin{table*}[p]", f"\\caption{{{caption}}}", f"\\label{{tab:latent-robustness-{label}}}",
            r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcccccc}", r"\hline",
            r"True count law & Fitted count law & IPCW IBS & Cure RMSE & Survival RMSE & Uno's C-index & Converged & Failed \\", r"\hline",
        ])
        for row in rows.to_dict("records"):
            performance_lines.append(" & ".join([
                TRUTH_LABELS[row["truth"]], MODEL_LABELS[row["model_name"]],
                *[_estimate(row, metric) for metric in study.METRICS],
                f"{row['converged']}/{row['successful']}", str(row["failed"]),
            ]) + r" \\")
        performance_lines.extend([r"\hline", r"\end{tabular}", "}", r"\end{table*}", ""])
        if contrasts.empty:
            continue
        selected = contrasts.loc[contrasts.latency_mode.eq(mode) & contrasts.method.eq(method) & contrasts.n.eq(n) & contrasts.followup.eq(followup)]
        paired_lines.extend([
            r"\begin{table*}[p]",
            f"\\caption{{Paired deterioration relative to the correct count-family fit for {label}. "
            "Positive values indicate worse performance: fitted minus reference for errors, reference minus fitted for Uno's C-index. "
            "Values are means (Monte Carlo standard errors) from finite successful replication pairs. "
            "NB under Poisson truth contains the truth as a limit. No practical robustness verdict is assigned without prespecified margins.}",
            f"\\label{{tab:latent-robustness-paired-{label}}}", r"\centering", r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llcccc}", r"\hline",
            r"True count law & Fitted count law & $\Delta$ IBS & $\Delta$ cure RMSE & $\Delta$ survival RMSE & C-index loss \\", r"\hline",
        ])
        for (truth, model), metrics in selected.groupby(["truth", "model_name"], sort=False):
            lookup = metrics.set_index("metric").to_dict("index")
            paired_lines.append(" & ".join([TRUTH_LABELS[truth], MODEL_LABELS[model], *[_estimate(lookup[metric]) for metric in study.METRICS]]) + r" \\")
        paired_lines.extend([r"\hline", r"\end{tabular}", "}", r"\end{table*}", ""])
    return "\n".join(performance_lines), "\n".join(paired_lines)


def create_figures(performance, contrasts, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode in ("estimated", "fixed", "stress"):
        if mode != "stress" and contrasts.empty:
            continue
        data = performance.loc[performance.study_part.eq("stress")] if mode == "stress" else contrasts.loc[contrasts.latency_mode.eq(mode)]
        if data.empty:
            continue
        panels = list(data.groupby(["method", "n", "followup"], sort=False))
        truths = [truth["name"] for truth in study.TRUTHS if (truth["reference_model"] is None) == (mode == "stress")]
        for metric in study.METRICS:
            fig, axes = plt.subplots(math.ceil(len(panels) / 2), 2, figsize=(12, 3.2 * math.ceil(len(panels) / 2)), squeeze=False)
            all_values = []
            matrices = []
            for keys, rows in panels:
                rows = rows if mode == "stress" else rows.loc[rows.metric.eq(metric)]
                value = f"{metric}_mean" if mode == "stress" else "mean"
                matrix = rows.pivot(index="truth", columns="model_name", values=value).reindex(index=truths, columns=study.FIT_MODELS).to_numpy(dtype=float)
                matrices.append(matrix)
                all_values.extend(matrix[np.isfinite(matrix)].tolist())
            limit = max((abs(value) for value in all_values), default=0) or 0.001
            for axis, (keys, _), matrix in zip(axes.flat, panels, matrices):
                method, n, followup = keys
                kwargs = {"cmap": "viridis"} if mode == "stress" else {"cmap": "RdBu_r", "norm": TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)}
                if mode == "stress":
                    kwargs.update(vmin=min(all_values, default=0), vmax=max(all_values, default=1) or 1)
                mesh = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", **kwargs)
                axis.set_title(f"{'Quadratic' if method == 2 else 'Periodic'}; n={n}; {followup} follow-up", fontsize=10)
                axis.set_xticks(range(3), ["Poisson", "Negative binomial", "Binomial (K=2)"], fontsize=8)
                axis.set_yticks(range(len(truths)), [TRUTH_LABELS[truth] for truth in truths], fontsize=8)
                for i in range(len(truths)):
                    for j in range(3):
                        value = matrix[i, j]
                        axis.text(j, i, f"{value:.4f}" if np.isfinite(value) else "--", ha="center", va="center", fontsize=8,
                                  color="white" if np.isfinite(value) and abs(value) > limit * 0.6 else "black")
                fig.colorbar(mesh, ax=axis, fraction=0.05)
            for axis in list(axes.flat)[len(panels):]:
                axis.set_visible(False)
            title = METRIC_LABELS[metric] if mode == "stress" else f"{METRIC_LABELS[metric]} deterioration (positive = worse)"
            fig.suptitle(f"Latent-count robustness: {mode} — {title}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            stem = output_dir / f"latent_robustness_{mode}_{metric}"
            try:
                for suffix in ("pdf", "png", "svg"):
                    path = stem.with_suffix(f".{suffix}")
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    paths.append(path)
            finally:
                plt.close(fig)
    return paths


def generate_report(*, results_dir=None, output_dir=None, kind="tables", phase="final", final_replications=None,
                    ibs_target=0.003, cure_rmse_target=0.005, ibs_margin=None, cure_rmse_margin=None):
    if kind == "pilot-summary" and phase != "pilot":
        raise ValueError("Pilot precision reports require --phase pilot; final samples cannot be used as the independent pilot")
    directory = Path(results_dir) if results_dir else study.phase_directory(phase)
    design, frame, missing = read_results(directory, require_complete=kind != "pilot-summary")
    if design["phase"] != phase:
        raise ValueError("Requested report phase does not match the saved design")
    margins = {name: value for name, value in (("integrated_brier", ibs_margin), ("cure_rmse", cure_rmse_margin)) if value is not None}
    if any(not np.isfinite(value) or value <= 0 for value in margins.values()):
        raise ValueError("Practical deterioration margins must be positive finite values")
    performance = performance_summary(frame) if not frame.empty else pd.DataFrame()
    pairs = paired_replications(frame) if not frame.empty else pd.DataFrame()
    contrasts = paired_summary(frame, pairs, margins=margins) if not frame.empty else pd.DataFrame()
    if kind == "pilot-summary":
        projections, report = pilot_precision(design, contrasts, missing, final_replications=final_replications, ibs_target=ibs_target, cure_rmse_target=cure_rmse_target)
        destination = Path(output_dir) if output_dir else directory / "reports"
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(destination / "pilot_precision.csv", projections)
        atomic_write_csv(destination / "pilot_performance.csv", performance)
        atomic_write_json(destination / "pilot_precision.json", report)
        (destination / "pilot_precision.txt").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report
    if missing:
        raise ValueError(f"Missing final cases: {missing}")
    if kind == "figures":
        destination = Path(output_dir) if output_dir else (
            study.PROJECT_ROOT / "figures/review/latent_robustness" if phase == "final" else directory / "figures"
        )
        paths = create_figures(performance, contrasts, destination)
        for path in paths:
            print(path)
        return paths
    destination = Path(output_dir) if output_dir else directory / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(destination / "performance_summary.csv", performance)
    atomic_write_csv(destination / "paired_replications.csv", pairs)
    atomic_write_csv(destination / "paired_summary.csv", contrasts)
    atomic_write_csv(destination / "latency_compensation.csv", latency_compensation(pairs))
    absolute, paired = render_tables(performance, contrasts, design)
    (destination / "latent-robustness-performance-table.tex").write_text(absolute + "\n", encoding="utf-8")
    (destination / "latent-robustness-paired-table.tex").write_text(paired + "\n", encoding="utf-8")
    atomic_write_json(destination / "report.json", {
        "study_id": design["study_id"], "phase": phase, "replications": design["replications"],
        "cases": len(design["cases"]), "planned_fits": len(frame), "successful": int(frame.status.eq("ok").sum()),
        "failed": int(frame.status.eq("failed").sum()), "practical_margins": margins,
        "interpretation": "Positive paired deterioration is worse; intervals describe Monte Carlo uncertainty. "
        "Poisson-mixture truth has no correct fitted candidate. Predictive robustness does not identify the true count law. "
        "Finite completed fits include nonconverged fits, whose counts are reported separately. "
        "Without prespecified margins, robustness is not assessed. Per-cell assessments are not a universal robustness claim.",
    })
    print(f"Saved robustness reports in {destination}")
    return performance, contrasts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("tables", "figures", "pilot-summary"), default="tables")
    parser.add_argument("--phase", choices=("pilot", "final"), default="final")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--final-replications", type=int, help="Allocation to assess; defaults to GGKM_ROBUSTNESS_REPLICATIONS or 250.")
    parser.add_argument("--ibs-target", type=float, default=0.003)
    parser.add_argument("--cure-rmse-target", type=float, default=0.005)
    parser.add_argument("--ibs-margin", type=float)
    parser.add_argument("--cure-rmse-margin", type=float)
    parser.add_argument("--allow-insufficient-precision", action="store_true",
                        help="Allow a complete pilot to proceed despite its projected MCSE; incomplete pilots still fail.")
    args = vars(parser.parse_args(argv))
    allow = args.pop("allow_insufficient_precision")
    try:
        result = generate_report(**args)
        if args["kind"] == "pilot-summary" and not (result["ready_for_final"] or allow and result["complete"]):
            print("Pilot does not validate the final allocation; inspect pilot_precision.json. R is not changed automatically.", file=sys.stderr)
            return 2
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
