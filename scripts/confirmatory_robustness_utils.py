"""Frozen protocol and simultaneous paired inference for the confirmation study."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from scripts import latent_robustness_utils as original
from scripts.experiment_utils.artifacts import atomic_write_json


DEFAULT_PROTOCOL = original.PROJECT_ROOT / "config/latent_robustness_confirmatory.json"
PRIMARY_METRICS = ("integrated_brier", "cure_rmse")
ANALYSES = ("all_finite", "jointly_converged")
REVIEW_MARGINS = {"integrated_brier": (0.001, 0.002, 0.003, 0.005, 0.01),
                  "cure_rmse": (0.005, 0.01, 0.015, 0.02)}
AUDIT_CASE_IDS = (4, 20, 28, 29, 30, 31)
FAMILY_SIZE = 40 * 2 * len(PRIMARY_METRICS) * len(ANALYSES)


def results_root():
    return Path(os.environ.get("GGKM_CONFIRMATORY_DIR", original.RESULTS_ROOT / "confirmatory")).expanduser().resolve()


def load_protocol(path=None):
    path = Path(path or os.environ.get("GGKM_CONFIRMATORY_PROTOCOL", DEFAULT_PROTOCOL))
    config = json.loads(path.read_text())
    if config["schema_version"] != 1 or tuple(config["primary_metrics"]) != PRIMARY_METRICS or tuple(config["analyses"]) != ANALYSES:
        raise ValueError("Unsupported confirmatory protocol schema, metrics, or analyses")
    if not 0 < config["alpha"] < 0.5 or not 0 < config["halfwidth_margin_fraction"] < 1:
        raise ValueError("Invalid alpha or precision fraction")
    for name in ("audit_replications", "pilot_replications", "minimum_final_replications"):
        if type(config[name]) is not int or config[name] < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    if config["final_replications"] is not None and (type(config["final_replications"]) is not int or config["final_replications"] < 2):
        raise ValueError("final_replications must be null or an integer >= 2")
    seeds = list(config["seeds"].values())
    if set(config["seeds"]) != {"audit", "pilot", "final"} or len(set(seeds)) != 3 or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Audit, pilot, and final require distinct nonnegative integer seeds")
    if set(seeds) & {original.MASTER_SEED, original.PILOT_MASTER_SEED, original.CALIBRATION_SEED}:
        raise ValueError("Confirmatory seeds must differ from the historical study seeds")
    if set(config["margins"]) != set(PRIMARY_METRICS):
        raise ValueError("Declare both primary margins (null until scientifically chosen)")
    for value in config["margins"].values():
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("Margins must be null or positive finite values")
    fitting = config["fitting"]
    if not fitting["dispersion_starts"] or any(not np.isfinite(x) or x <= 0 for x in fitting["dispersion_starts"]):
        raise ValueError("Dispersion starts must be positive and finite")
    if not fitting["latency_starts"] or any(len(row) != 3 or any(not np.isfinite(x) or x <= 0 for x in row) for row in fitting["latency_starts"]):
        raise ValueError("Each latency start must contain three positive finite values")
    if type(fitting["retries"]) is not int or fitting["retries"] < 0 or type(fitting["retry_multiplier"]) is not int or fitting["retry_multiplier"] < 2:
        raise ValueError("Invalid retry policy")
    profile = fitting["profile_dispersion"]
    if not profile or 0 not in profile or any(not np.isfinite(x) or x < 0 for x in profile):
        raise ValueError("The profile must include exact Poisson (0) and nonnegative finite dispersions")
    if not np.isfinite(fitting["profile_improvement_tolerance"]) or fitting["profile_improvement_tolerance"] <= 0:
        raise ValueError("Invalid profile improvement tolerance")
    return config


def require_margins(config):
    if any(config["margins"][metric] is None for metric in PRIMARY_METRICS) or not isinstance(config["margin_justification"], str) or not config["margin_justification"].strip():
        raise ValueError("Set both practical margins and a scientific margin_justification before the independent pilot/final study; historical losses are not a margin justification.")


def protocol_snapshot(config, *, audit=False):
    # Reuse the original generator and exact tuning grid, recording all their
    # source/library hashes as well as the new adapters and reporting code.
    base = original.build_design("final", replications=2, seed=config["seeds"]["final"])
    for key in ("study_id", "phase", "replications", "master_seed"):
        base.pop(key)
    for name in ("scripts/confirmatory_robustness_utils.py", "scripts/confirmatory_robustness_fitting.py",
                 "scripts/confirmatory_robustness.py", "scripts/summarize_latent_robustness.py"):
        base["source_sha256"][name] = original.checksum(original.PROJECT_ROOT / name)
    frozen_config = {"fitting": config["fitting"], "audit_replications": config["audit_replications"],
                     "audit_seed": config["seeds"]["audit"]} if audit else config
    payload = {"study": "ggkm-latent-count-confirmatory", "protocol": frozen_config, "design": base,
               "simultaneous_family_size": FAMILY_SIZE,
               "interpretation": "One-sided Bonferroni Student-t Monte Carlo bounds; finite-sample approximation. NB contains Poisson in its phi->0 limit. Parameter recovery is separate from prediction."}
    return {**payload, "protocol_id": original.identity(payload)}


def freeze_protocol(root, config, *, audit=False):
    snapshot = protocol_snapshot(config, audit=audit)
    path = Path(root) / ("audit_protocol.json" if audit else "protocol.json")
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(snapshot)):
            raise ValueError("Protocol, source, or software changed: use a new GGKM_CONFIRMATORY_DIR. Existing study outputs are immutable.")
    else:
        atomic_write_json(path, snapshot)
    return snapshot


def variants(case):
    fits = [("estimated", model) for model in original.FIT_MODELS]
    if case["n"] == 600 and case["study_part"] == "primary":
        fits += [("fixed", model) for model in original.FIT_MODELS]
        if case["truth"].startswith("nb"):
            fits.append(("phi_fixed", "bn"))
    return fits


def paired_values(frame, *, analyses=ANALYSES):
    records = []
    keys = ["task_id", "truth", "study_part", "n", "method", "followup", "latency_mode"]
    for values, rows in frame.groupby(keys, sort=False):
        metadata = dict(zip(keys, values, strict=True))
        reference = original.truth_spec(metadata["truth"])["reference_model"]
        if reference is None or metadata["latency_mode"] == "phi_fixed":
            continue
        ref = rows.loc[rows.model_name.eq(reference)].set_index("replication")
        for model in original.FIT_MODELS:
            focal = rows.loc[rows.model_name.eq(model)].set_index("replication")
            if set(focal.index) != set(ref.index):
                raise ValueError("Unpaired replication IDs")
            ref = ref.reindex(focal.index)
            for metric in original.METRICS:
                good = focal.status.eq("ok") & ref.status.eq("ok") & np.isfinite(focal[metric]) & np.isfinite(ref[metric])
                for analysis in analyses:
                    mask = good if analysis == "all_finite" else good & focal.converged & ref.converged
                    differences = focal.loc[mask, metric] - ref.loc[mask, metric]
                    if metric == "uno_c_index":
                        differences = -differences
                    for replication, difference in differences.items():
                        records.append({**metadata, "model_name": model, "reference_model": reference,
                                        "metric": metric, "analysis": analysis, "replication": replication,
                                        "deterioration": float(difference)})
    return pd.DataFrame(records, columns=[*keys, "model_name", "reference_model", "metric", "analysis", "replication", "deterioration"])


def contrast_summary(frame, pairs, config, *, family_size=None):
    from scripts.summarize_latent_robustness import descriptive

    family_size = FAMILY_SIZE if family_size is None else family_size
    records = []
    for (case_id, mode, model), rows in frame.groupby(["task_id", "latency_mode", "model_name"]):
        reference = original.truth_spec(rows.iloc[0].truth)["reference_model"]
        if reference is None or model == reference or mode == "phi_fixed":
            continue
        for analysis in ANALYSES:
            for metric in original.METRICS:
                selected = pairs.loc[pairs.task_id.eq(case_id) & pairs.latency_mode.eq(mode) & pairs.model_name.eq(model) & pairs.analysis.eq(analysis) & pairs.metric.eq(metric)] if not pairs.empty else pd.DataFrame()
                stats = descriptive(selected.deterioration if not selected.empty else [])
                primary = mode == "estimated" and metric in PRIMARY_METRICS
                critical = float(student_t.ppf(1 - config["alpha"] / family_size, stats["valid"] - 1)) if stats["valid"] > 1 else None
                upper = stats["mean"] + critical * stats["mcse"] if critical is not None else None
                lower = stats["mean"] - critical * stats["mcse"] if critical is not None else None
                margin = config["margins"].get(metric)
                complete = stats["valid"] == len(rows) if analysis == "all_finite" else stats["valid"] >= 2
                assessment = "not_assessed"
                if primary and margin is not None:
                    assessment = "incomplete_pairs" if not complete or upper is None else (
                        "within_margin" if upper < margin else "exceeds_margin" if lower > margin else "inconclusive")
                records.append({**{key: rows.iloc[0][key] for key in ("task_id", "truth", "study_part", "n", "method", "followup")},
                                "latency_mode": mode, "model_name": model, "reference_model": reference,
                                "specification": "poisson_limit" if rows.iloc[0].truth == "poisson" and model == "bn" else "misspecified",
                                "analysis": analysis, "metric": metric, "primary": primary,
                                "attempted_pairs": len(rows), **stats, "family_size": family_size,
                                "simultaneous_lower": lower, "simultaneous_upper": upper,
                                "margin": margin, "assessment": assessment})
    return pd.DataFrame(records)


def allocation(summary, config):
    """Freeze R using pilot SDs; no final samples participate in allocation."""
    require_margins(config)
    selected = summary.loc[summary.primary]
    if len(selected) != FAMILY_SIZE or (selected.valid < 2).any():
        raise ValueError("Allocation needs every primary pilot contrast in both analyses with >=2 finite pairs")
    projections = []
    for row in selected.to_dict("records"):
        fraction = row["valid"] / row["attempted_pairs"]
        halfwidth = config["halfwidth_margin_fraction"] * config["margins"][row["metric"]]
        required = max(2, config["minimum_final_replications"])
        # Student-t critical values depend on the projected finite-pair count.
        # Iterate monotonically until the planned halfwidth is small enough.
        while True:
            finite_count = max(2, math.floor(required * fraction))
            critical = float(student_t.ppf(1 - config["alpha"] / FAMILY_SIZE, finite_count - 1))
            needed = math.ceil((critical * row["sd"] / halfwidth) ** 2 / fraction)
            if needed <= required:
                break
            required = needed
        projections.append({**{key: row[key] for key in ("task_id", "model_name", "metric", "analysis", "sd", "valid", "attempted_pairs")},
                            "finite_pair_fraction": fraction, "target_halfwidth": halfwidth,
                            "required_replications": required})
    required = max(row["required_replications"] for row in projections)
    chosen = config["final_replications"] or required
    return pd.DataFrame(projections), {"required_replications": required, "final_replications": chosen,
                                       "ready": chosen >= required,
                                       "rule": "Maximum pilot-SD Bonferroni-t halfwidth requirement across all primary contrasts and both analyses; independent pilot excluded from final estimates."}


def margin_sensitivity(summary):
    """Descriptive planning grid, never a data-driven choice of primary margin."""
    records = []
    primary = summary.loc[summary.primary]
    for (analysis, metric), rows in primary.groupby(["analysis", "metric"]):
        finite = rows.simultaneous_upper.notna()
        for margin in REVIEW_MARGINS[metric]:
            passing = int((rows.simultaneous_upper < margin).sum())
            records.append({"analysis": analysis, "metric": metric, "illustrative_margin": margin,
                            "comparisons": len(rows), "finite_bounds": int(finite.sum()), "within_margin": passing,
                            "all_within": bool(passing == len(rows) and finite.all()),
                            "largest_upper_bound": rows.simultaneous_upper.max(),
                            "interpretation": "Exploratory sensitivity only; candidate margins are not scientifically validated or adopted."})
    return pd.DataFrame(records)
