"""Frozen design and data generation for latent-count misspecification studies."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import numpy as np
from scipy.optimize import brentq
from scipy.stats import gengamma

from evaluate.simulated_data import cure_probability
from scripts import run_pcm_experiment as pcm
from scripts.experiment_utils.artifacts import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data" / "latent_robustness"
N_REPLICATIONS = 250
PILOT_REPLICATIONS = 10
MASTER_SEED = 20261107
PILOT_MASTER_SEED = 20261108
CALIBRATION_SEED = 20261109
CALIBRATION_SIZE = 32768
GG_PARAMETERS = {"a": 1.6, "d": 1.3, "p": 1.1}
FOLLOWUP = {"long": 0.20, "short": 0.50}
FIT_MODELS = ("poisson", "bn", "binomial")
TRUTHS = (
    {"name": "poisson", "distribution": "poisson", "phi": None, "reference_model": "poisson"},
    {"name": "nb025", "distribution": "negative_binomial", "phi": 0.25, "reference_model": "bn"},
    {"name": "nb100", "distribution": "negative_binomial", "phi": 1.0, "reference_model": "bn"},
    {"name": "nb200", "distribution": "negative_binomial", "phi": 2.0, "reference_model": "bn"},
    {"name": "binomial2", "distribution": "binomial", "phi": None, "reference_model": "binomial"},
    {"name": "poisson_mixture", "distribution": "poisson_mixture", "phi": None, "reference_model": None},
)
MIXTURE_WEIGHT = 0.8
MIXTURE_RATE_RATIO = 5.0
BINOMIAL_K = 2
GRID_POINTS = 41
SEED_COLUMNS = ("generation_seed", "count_seed", "activation_seed", "censor_seed", "split_seed", "inner_seed")
METRICS = ("integrated_brier", "cure_rmse", "survival_rmse", "uno_c_index")


def configured_final_replications():
    value = int(os.environ.get("GGKM_ROBUSTNESS_REPLICATIONS", str(N_REPLICATIONS)))
    if value < 2:
        raise ValueError("GGKM_ROBUSTNESS_REPLICATIONS must be an integer of at least two")
    return value


def phase_directory(phase):
    if phase not in {"pilot", "final"}:
        raise ValueError("phase must be pilot or final")
    variable = f"GGKM_ROBUSTNESS_{phase.upper()}_DIR"
    return Path(os.environ.get(variable, str(RESULTS_ROOT / phase))).expanduser().resolve()


def build_experiments():
    return [
        {
            "task_id": index, "truth": truth["name"], "n": n, "method": method,
            "followup": followup, "susceptible_censoring_target": target,
            "reference_model": truth["reference_model"],
            "study_part": "stress" if truth["reference_model"] is None else "primary",
        }
        for index, (truth, n, method, followup, target) in enumerate(
            (truth, n, method, followup, target)
            for truth in TRUTHS for n in (300, 600) for method in (2, 3)
            for followup, target in FOLLOWUP.items()
        )
    ]


def truth_spec(name):
    return next(truth for truth in TRUTHS if truth["name"] == name)


def latency_modes(case):
    return ("estimated", "fixed") if case["n"] == 600 and case["study_part"] == "primary" else ("estimated",)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_design(phase="final", *, replications=None, seed=None):
    if phase not in {"pilot", "final"}:
        raise ValueError("phase must be pilot or final")
    replications = (PILOT_REPLICATIONS if phase == "pilot" else configured_final_replications()) if replications is None else replications
    seed = (PILOT_MASTER_SEED if phase == "pilot" else MASTER_SEED) if seed is None else seed
    if type(replications) is not int or replications < 2:
        raise ValueError("replications must be an integer of at least two")
    if type(seed) is not int or seed < 0 or seed == (MASTER_SEED if phase == "pilot" else PILOT_MASTER_SEED):
        raise ValueError("Use a nonnegative seed distinct from the other study phase")
    source_files = (
        "scripts/latent_robustness_utils.py", "scripts/run_latent_robustness_experiment.py",
        "evaluate/simulated_data.py", "evaluate/util_cross_validation.py", "evaluate/metrics.py",
        "scripts/experiment_utils/models.py",
    )
    library_modules = (
        "skcure.ggkm", "skcure._em", "skcure._latent", "skcure._latency",
        "skcure.distributions.generalized_gamma", "skcure.utils.kernels",
    )
    payload = {
        "schema_version": 1, "study": "ggkm-latent-count-robustness", "phase": phase,
        "replications": replications, "master_seed": seed,
        "calibration_seed": CALIBRATION_SEED, "calibration_size": CALIBRATION_SIZE,
        "gg_parameters": GG_PARAMETERS, "binomial_k": BINOMIAL_K,
        "mixture_weight": MIXTURE_WEIGHT, "mixture_rate_ratio": MIXTURE_RATE_RATIO,
        "truths": TRUTHS, "cases": build_experiments(), "fit_models": FIT_MODELS,
        "fixed_latency_diagnostic_n": 600, "kernel": "rbf",
        "parameter_grids": {name: pcm.parameter_grid(name, "rbf") for name in FIT_MODELS},
        "estimator_parameters": {name: pcm.MODELS[name]().get_params(deep=False) for name in FIT_MODELS},
        "test_size": pcm.TEST_SIZE, "inner_splits": pcm.N_INNER_SPLITS,
        "inner_grid_points": pcm.INNER_GRID_POINTS, "outer_grid_points": pcm.OUTER_GRID_POINTS,
        "observed_time_quantile": 0.90, "minimum_censoring_survival": 0.05,
        "truth_grid_points": GRID_POINTS,
        "software": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "skcure")
        }},
        "source_sha256": {name: checksum(PROJECT_ROOT / name) for name in source_files},
        "library_sha256": {name: checksum(importlib.import_module(name).__file__) for name in library_modules},
    }
    return {**payload, "study_id": identity(payload)}


def write_design(directory, design):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "design.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(design)):
            raise ValueError(f"Different study design, phase, or source version in {directory}; use a separate output directory.")
    else:
        atomic_write_json(path, design)


def count_parameters(pc, truth, *, weight=MIXTURE_WEIGHT, ratio=MIXTURE_RATE_RATIO):
    pc = np.asarray(pc, dtype=float)
    if np.any(~np.isfinite(pc)) or np.any((pc <= 0) | (pc >= 1)):
        raise ValueError("Cure probabilities must lie strictly between zero and one")
    distribution = truth["distribution"]
    if distribution == "poisson":
        return -np.log(pc)
    if distribution == "negative_binomial":
        return np.expm1(-truth["phi"] * np.log(pc)) / truth["phi"]
    if distribution == "binomial":
        return -np.expm1(np.log(pc) / BINOMIAL_K)
    if distribution != "poisson_mixture":
        raise ValueError(f"Unknown count distribution: {distribution}")
    lower, upper = np.zeros_like(pc), -np.log(pc)
    for _ in range(64):
        midpoint = (lower + upper) / 2
        zero_mass = weight * np.exp(-midpoint) + (1 - weight) * np.exp(-ratio * midpoint)
        lower = np.where(zero_mass > pc, midpoint, lower)
        upper = np.where(zero_mass > pc, upper, midpoint)
    return (lower + upper) / 2


def true_survival(X, times, case, *, gg_parameters=GG_PARAMETERS):
    pc = cure_probability(X[:, 0], X[:, 1], case["method"])
    truth = truth_spec(case["truth"])
    parameter = count_parameters(pc, truth)[:, None]
    a, d, p = (gg_parameters[name] for name in ("a", "d", "p"))
    F = gengamma.cdf(np.asarray(times), a=d / p, c=p, scale=a)[None, :]
    if truth["distribution"] == "poisson":
        return np.exp(-parameter * F)
    if truth["distribution"] == "negative_binomial":
        return np.exp(-np.log1p(truth["phi"] * parameter * F) / truth["phi"])
    if truth["distribution"] == "binomial":
        return (1 - parameter * F) ** BINOMIAL_K
    return MIXTURE_WEIGHT * np.exp(-parameter * F) + (1 - MIXTURE_WEIGHT) * np.exp(-MIXTURE_RATE_RATIO * parameter * F)


def replication_seeds(case, replication, master_seed, *, phase="final"):
    # Common covariates/activation uniforms across count laws; no fitted-model
    # identifier enters any data, split, or inner-fold seed.
    stream = {"final": 1, "pilot": 2, "calibration": 3}[phase]
    seeds = np.random.SeedSequence([master_seed, stream, case["n"], case["method"], replication]).generate_state(6)
    result = dict(zip(SEED_COLUMNS, map(int, seeds), strict=True))
    truth_index = next(index for index, truth in enumerate(TRUTHS) if truth["name"] == case["truth"])
    result["count_seed"] = int(np.random.SeedSequence([result["count_seed"], truth_index]).generate_state(1)[0])
    return result


def uncensored_data(n, case, seeds):
    X = np.random.default_rng(seeds["generation_seed"]).uniform(-1, 1, (n, 2))
    pc = cure_probability(X[:, 0], X[:, 1], case["method"])
    truth = truth_spec(case["truth"])
    parameter = count_parameters(pc, truth)
    rng = np.random.default_rng(seeds["count_seed"])
    if truth["distribution"] == "poisson":
        counts = rng.poisson(parameter)
    elif truth["distribution"] == "negative_binomial":
        shape = 1 / truth["phi"]
        counts = rng.negative_binomial(shape, shape / (shape + parameter))
    elif truth["distribution"] == "binomial":
        counts = rng.binomial(BINOMIAL_K, parameter)
    else:
        rate = parameter * np.where(rng.random(n) < MIXTURE_WEIGHT, 1.0, MIXTURE_RATE_RATIO)
        counts = rng.poisson(rate)
    T = np.full(n, np.inf)
    susceptible = counts > 0
    u = np.random.default_rng(seeds["activation_seed"]).uniform(np.finfo(float).eps, 1, n)
    # Exact minimum of N iid activation times without drawing N separate times.
    q = -np.expm1(np.log(u[susceptible]) / counts[susceptible])
    T[susceptible] = gengamma.ppf(q, a=GG_PARAMETERS["d"] / GG_PARAMETERS["p"], c=GG_PARAMETERS["p"], scale=GG_PARAMETERS["a"])
    return {"X": X, "true_count": counts, "true_cure": pc, "true_time": T}


def calibrate_censoring(case, *, sample_size=CALIBRATION_SIZE):
    seeds = replication_seeds({**case, "n": sample_size}, 0, CALIBRATION_SEED, phase="calibration")
    T = uncensored_data(sample_size, case, seeds)["true_time"]
    finite = T[np.isfinite(T)]
    target = case["susceptible_censoring_target"]
    if not 0 < target < 1 or not finite.size:
        raise ValueError("Censoring calibration needs susceptible times and a target in (0, 1)")
    def objective(log_mean):
        return float(np.mean(-np.expm1(-finite / np.exp(log_mean))) - target)
    center = np.log(np.median(finite))
    mean = float(np.exp(brentq(objective, center - 30, center + 30)))
    return {"censor_mean": mean, "calibration_susceptible_size": int(len(finite)),
            "calibration_censoring_fraction": float(np.mean(-np.expm1(-finite / mean)))}


def generate_replication(case, replication, design, censor_mean):
    seeds = replication_seeds(case, replication, design["master_seed"], phase=design["phase"])
    data = uncensored_data(case["n"], case, seeds)
    C = np.random.default_rng(seeds["censor_seed"]).exponential(censor_mean, case["n"])
    data.update(time=np.minimum(data["true_time"], C), event=(data["true_time"] <= C).astype(int), censor_time=C)
    if not np.isfinite(data["time"]).all() or np.any(data["time"] <= 0):
        raise FloatingPointError("Generated observed times must be positive and finite")
    return data, seeds


def array_hash(arrays):
    digest = hashlib.sha256()
    for name, values in sorted(arrays.items()):
        array = np.ascontiguousarray(values)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
