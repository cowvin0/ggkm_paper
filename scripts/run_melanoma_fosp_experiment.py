from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.auto import tqdm

from skcure.models import GGKM, MixtureCureModel, PromotionTimeModel
from skcure.utils.estimator import fit_cure_model
from ggkm_paper.scripts.preprocessing import FOSPMelanomaSurvivalPreprocessor
from skcure.metrics import (
    integrated_brier_score,
    uno_c_index_rmst,
    auc_cure,
)

MODELS = {
    "bernoulli": partial(GGKM, latent_distribution="binomial", K_bin=1),
    "binomial": partial(GGKM, latent_distribution="binomial"),
    "poisson": partial(GGKM, latent_distribution="poisson"),
    "bn": partial(GGKM, latent_distribution="negative_binomial"),
    "mixture_cure": MixtureCureModel,
    "promotion_time": PromotionTimeModel,
}

KERNELS = [
    "linear",
    "rbf",
    "laplacian",
    "exponential",
    "cauchy",
    "sigmoid",
    "polynomial",
    "wavelet",
]

KERNEL_MODELS = {
    "bernoulli",
    "binomial",
    "poisson",
    "bn",
}

PARAMETRIC_MODELS = {
    "mixture_cure",
    "promotion_time",
}


N_OUTER_SPLITS = 50
OUTER_TEST_SIZE = 0.30

N_INNER_SPLITS = 4

INNER_GRID_POINTS = 40
OUTER_GRID_POINTS = 60

OBSERVED_TIME_QUANTILE = 0.90
MINIMUM_CENSORING_SURVIVAL = 0.05

MASTER_SEED = 20260817


# ============================================================
# FOSP melanoma data / preprocessing configuration
# ============================================================

FOLLOWUP = "extended"

INCLUDE_TREATMENTS = True
INCLUDE_EXTRA_TREATMENTS = True
INCLUDE_TOPOGRAPHY = True
INCLUDE_MORPHOLOGY = True
MIN_MORPHOLOGY_FREQUENCY = 50

DATA_PATH_CANDIDATES = [
    Path("data/melanoma_fosp_gomez2023.csv"),
    Path("ggkm/data/melanoma_fosp_gomez2023.csv"),
]

if FOLLOWUP == "extended":
    TIME_COL = "tempo_ext"
    STATUS_COL = "status_ext"
elif FOLLOWUP == "article":
    TIME_COL = "tempo"
    STATUS_COL = "status"
else:
    raise ValueError("FOLLOWUP must be either 'extended' or 'article'.")


def make_preprocessor():
    return FOSPMelanomaSurvivalPreprocessor(
        followup=FOLLOWUP,
        include_treatments=INCLUDE_TREATMENTS,
        include_extra_treatments=INCLUDE_EXTRA_TREATMENTS,
        include_topography=INCLUDE_TOPOGRAPHY,
        include_morphology=INCLUDE_MORPHOLOGY,
        min_morphology_frequency=MIN_MORPHOLOGY_FREQUENCY,
    )


def resolve_data_path():
    for path in DATA_PATH_CANDIDATES:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in DATA_PATH_CANDIDATES)

    raise FileNotFoundError(
        "Could not find melanoma_fosp_gomez2023.csv. " f"Searched: {searched}"
    )


KERNEL_PENALTIES = [
    0.30,
    3.00,
]

KERNEL_GAMMAS = [
    0.25,
    1.00,
]

POLYNOMIAL_DEGREES = [
    2,
    3,
]

POLYNOMIAL_COEF0 = 1.0

SIGMOID_COEF0 = 0.0

WAVELET_A_VALUES = [
    0.50,
    1.00,
    2.00,
    # 10.00,
]

BINOMIAL_K = 2

OUTPUT_DIR = Path(f"data/melanoma_fosp_results/grid_search_paired_50_{FOLLOWUP}")


EXPERIMENT_CONFIG = {
    "followup": FOLLOWUP,
    "include_treatments": INCLUDE_TREATMENTS,
    "include_extra_treatments": INCLUDE_EXTRA_TREATMENTS,
    "include_topography": INCLUDE_TOPOGRAPHY,
    "include_morphology": INCLUDE_MORPHOLOGY,
    "min_morphology_frequency": MIN_MORPHOLOGY_FREQUENCY,
    "n_outer_splits": N_OUTER_SPLITS,
    "outer_test_size": OUTER_TEST_SIZE,
    "n_inner_splits": N_INNER_SPLITS,
    "inner_grid_points": INNER_GRID_POINTS,
    "outer_grid_points": OUTER_GRID_POINTS,
    "observed_time_quantile": OBSERVED_TIME_QUANTILE,
    "minimum_censoring_survival": MINIMUM_CENSORING_SURVIVAL,
    "master_seed": MASTER_SEED,
    "kernel_penalties": KERNEL_PENALTIES,
    "kernel_gammas": KERNEL_GAMMAS,
    "polynomial_degrees": POLYNOMIAL_DEGREES,
    "polynomial_coef0": POLYNOMIAL_COEF0,
    "sigmoid_coef0": SIGMOID_COEF0,
    "wavelet_A_values": WAVELET_A_VALUES,
    "binomial_K": BINOMIAL_K,
}

EXPERIMENT_SIGNATURE = hashlib.sha256(
    json.dumps(
        EXPERIMENT_CONFIG,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()[:16]


def build_experiments():
    experiments = []

    for model_name in MODELS:
        if model_name in KERNEL_MODELS:
            for kernel in KERNELS:
                experiments.append(
                    {
                        "model_name": model_name,
                        "kernel": kernel,
                    }
                )

        else:
            experiments.append(
                {
                    "model_name": model_name,
                    "kernel": None,
                }
            )

    return experiments


def get_experiment(
    experiments,
    task_id,
):
    if task_id < 0 or task_id >= len(experiments):
        raise ValueError(
            f"Invalid PBS_ARRAY_INDEX={task_id}. " f"Expected 0-{len(experiments) - 1}."
        )

    return experiments[task_id]


def load_data():
    data_path = resolve_data_path()

    df = pd.read_csv(data_path).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("FOSP melanoma data contain fewer than two observations.")

    if "id" in df.columns and df["id"].duplicated().any():
        duplicate_ids = (
            df.loc[df["id"].duplicated(keep=False), "id"].drop_duplicates().tolist()
        )

        raise ValueError(
            "Duplicate patient identifiers were found in the FOSP melanoma data: "
            f"{duplicate_ids[:10]}"
        )

    for column in [TIME_COL, STATUS_COL]:
        if column not in df.columns:
            raise ValueError(
                f"FOSP melanoma data are missing required outcome column: {column}"
            )

    time_values = pd.to_numeric(
        df[TIME_COL],
        errors="coerce",
    ).to_numpy(dtype=float)

    status_values = pd.to_numeric(
        df[STATUS_COL],
        errors="coerce",
    ).to_numpy(dtype=float)

    if np.any(~np.isfinite(time_values)):
        raise ValueError(f"FOSP melanoma data contain non-finite values in {TIME_COL}.")

    if np.any(time_values <= 0.0):
        raise ValueError(
            f"FOSP melanoma data contain non-positive survival times in {TIME_COL}."
        )

    if np.any(~np.isfinite(status_values)):
        raise ValueError(
            f"FOSP melanoma data contain non-finite values in {STATUS_COL}."
        )

    if not np.all(
        np.isin(
            status_values,
            [
                0.0,
                1.0,
            ],
        )
    ):
        raise ValueError(f"{STATUS_COL} must contain only 0 and 1.")

    if np.unique(status_values).size != 2:
        raise ValueError(
            "FOSP melanoma data must contain both events " "and censored observations."
        )

    # Validate the complete feature specification once before launching
    # the repeated-holdout experiment. The fitted object is discarded;
    # all actual preprocessing inside CV is fitted on training data only.
    validation_preprocessor = make_preprocessor()
    validation_preprocessor._validate(df)

    return df


# ============================================================
# IPCW evaluation horizon
# ============================================================


def censoring_reliable_horizon(
    t,
    delta,
    minimum_survival=MINIMUM_CENSORING_SURVIVAL,
):
    t = np.asarray(
        t,
        dtype=float,
    ).reshape(-1)

    delta = np.asarray(
        delta,
        dtype=float,
    ).reshape(-1)

    if t.size == 0:
        raise ValueError("Cannot construct censoring survival from empty data.")

    if t.shape != delta.shape:
        raise ValueError("t and delta must have the same shape.")

    if np.any(~np.isfinite(t)):
        raise ValueError("Survival times must be finite.")

    if np.any(t < 0.0):
        raise ValueError("Survival times must be non-negative.")

    if not np.all(
        np.isin(
            delta,
            [
                0.0,
                1.0,
            ],
        )
    ):
        raise ValueError("delta must contain only 0 and 1.")

    if not (0.0 < minimum_survival < 1.0):
        raise ValueError("minimum_survival must lie in (0, 1).")

    order = np.argsort(
        t,
        kind="mergesort",
    )

    sorted_time = t[order]
    sorted_delta = delta[order]

    censoring_event = 1.0 - sorted_delta

    (
        unique_time,
        first_index,
        counts,
    ) = np.unique(
        sorted_time,
        return_index=True,
        return_counts=True,
    )

    survival_after = np.empty(
        unique_time.size,
        dtype=float,
    )

    survival = 1.0
    sample_size = sorted_time.size

    for (
        index,
        (
            start,
            count,
        ),
    ) in enumerate(
        zip(
            first_index,
            counts,
            strict=True,
        )
    ):
        at_risk = sample_size - int(start)

        censoring_count = int(np.sum(censoring_event[start : start + count]))

        if censoring_count > 0:
            survival *= 1.0 - censoring_count / at_risk

        survival_after[index] = survival

    reliable = survival_after >= minimum_survival

    if not np.any(reliable):
        return 0.0

    reliable_index = np.flatnonzero(reliable)[-1]

    return float(unique_time[reliable_index])


def make_evaluation_grid(
    t,
    delta,
    *,
    points,
):
    t = np.asarray(
        t,
        dtype=float,
    ).reshape(-1)

    delta = np.asarray(
        delta,
        dtype=float,
    ).reshape(-1)

    quantile_horizon = float(
        np.quantile(
            t,
            OBSERVED_TIME_QUANTILE,
        )
    )

    reliable_horizon = censoring_reliable_horizon(
        t,
        delta,
        minimum_survival=(MINIMUM_CENSORING_SURVIVAL),
    )

    horizon = min(
        quantile_horizon,
        reliable_horizon,
    )

    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("Training data provide no positive reliable IPCW horizon.")

    t_grid = np.linspace(
        0.0,
        horizon,
        points,
        dtype=float,
    )

    return (
        t_grid,
        horizon,
    )


# ============================================================
# Model construction
# ============================================================


def build_model(
    estimator,
    kernel,
    params,
):

    if kernel is None:
        return estimator(
            **params,
        )

    return estimator(
        kernel=kernel,
        **params,
    )


def fixed_kernel_model_params(
    model_name,
):
    if model_name == "bernoulli":
        return {
            "K_bin": 1,
        }

    if model_name == "binomial":
        return {
            "K_bin": BINOMIAL_K,
        }

    return {}


# ============================================================
# Kernel parameter grids
# ============================================================


def kernel_parameter_grid(
    model_name,
    kernel,
):
    base_params = fixed_kernel_model_params(model_name)

    grid = []

    if kernel == "linear":
        for lambda_reg in KERNEL_PENALTIES:
            grid.append(
                {
                    **base_params,
                    "lambda_reg": lambda_reg,
                }
            )

    elif kernel in {
        "rbf",
        "laplacian",
        "exponential",
        "cauchy",
    }:
        for gamma in KERNEL_GAMMAS:
            for lambda_reg in KERNEL_PENALTIES:
                grid.append(
                    {
                        **base_params,
                        "gamma": gamma,
                        "lambda_reg": lambda_reg,
                    }
                )

    elif kernel == "sigmoid":
        for gamma in KERNEL_GAMMAS:
            for lambda_reg in KERNEL_PENALTIES:
                grid.append(
                    {
                        **base_params,
                        "gamma": gamma,
                        "coef0": SIGMOID_COEF0,
                        "lambda_reg": lambda_reg,
                    }
                )

    elif kernel == "polynomial":
        for gamma in KERNEL_GAMMAS:
            for degree in POLYNOMIAL_DEGREES:
                for lambda_reg in KERNEL_PENALTIES:
                    grid.append(
                        {
                            **base_params,
                            "gamma": gamma,
                            "degree": degree,
                            "coef0": (POLYNOMIAL_COEF0),
                            "lambda_reg": lambda_reg,
                        }
                    )

    elif kernel == "wavelet":
        for A in WAVELET_A_VALUES:
            for lambda_reg in KERNEL_PENALTIES:
                grid.append(
                    {
                        **base_params,
                        "A": A,
                        "lambda_reg": lambda_reg,
                    }
                )

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if not grid:
        raise RuntimeError(f"Empty parameter grid for kernel={kernel}.")

    return grid


# ============================================================
# Inner CV
# ============================================================


def prepare_inner_splits(
    df_outer_train,
    *,
    inner_seed,
):
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=inner_seed,
    )

    prepared = []

    for (
        tr_idx,
        val_idx,
    ) in inner_cv.split(
        df_outer_train,
        df_outer_train[STATUS_COL],
    ):
        df_tr = df_outer_train.iloc[tr_idx].reset_index(drop=True)

        df_val = df_outer_train.iloc[val_idx].reset_index(drop=True)

        preprocessor = make_preprocessor()

        (
            X_tr,
            t_tr,
            d_tr,
        ) = preprocessor.fit(df_tr)

        (
            X_val,
            t_val,
            d_val,
        ) = preprocessor.transform(df_val)

        (
            t_grid_inner,
            _,
        ) = make_evaluation_grid(
            t_tr,
            d_tr,
            points=INNER_GRID_POINTS,
        )

        prepared.append(
            (
                X_tr,
                t_tr,
                d_tr,
                X_val,
                t_val,
                d_val,
                t_grid_inner,
            )
        )

    return prepared


def score_params_on_inner_splits(
    inner_splits,
    *,
    estimator,
    kernel,
    params,
):
    fold_scores = []

    for (
        X_tr,
        t_tr,
        d_tr,
        X_val,
        t_val,
        d_val,
        t_grid_inner,
    ) in inner_splits:
        model = build_model(
            estimator,
            kernel,
            params,
        )

        fit_cure_model(
            model,
            X_tr,
            t_tr,
            d_tr,
        )

        S_val = model.predict_survival(
            X_val,
            t_grid_inner,
        )

        ibs = integrated_brier_score(
            S_val,
            t_val,
            d_val,
            t_tr,
            d_tr,
            t_grid_inner,
        )

        if not np.isfinite(ibs):
            raise FloatingPointError("Non-finite inner integrated Brier score.")

        fold_scores.append(float(ibs))

    if len(fold_scores) != N_INNER_SPLITS:
        raise RuntimeError("Inner tuning did not evaluate every fold.")

    return float(np.mean(fold_scores))


# ============================================================
# Exhaustive kernel-grid tuning
# ============================================================


def tune_kernel_grid(
    df_outer_train,
    *,
    estimator,
    model_name,
    kernel,
    inner_seed,
):
    inner_splits = prepare_inner_splits(
        df_outer_train,
        inner_seed=inner_seed,
    )

    candidates = kernel_parameter_grid(
        model_name,
        kernel,
    )

    scores = []
    failure_reasons = []

    for candidate in candidates:
        try:
            score = score_params_on_inner_splits(
                inner_splits,
                estimator=estimator,
                kernel=kernel,
                params=candidate,
            )

            failure_reason = None

        except Exception as error:
            score = np.inf
            failure_reason = f"{type(error).__name__}: " f"{error}"

        scores.append(float(score))

        failure_reasons.append(failure_reason)

    scores = np.asarray(
        scores,
        dtype=float,
    )

    finite = np.isfinite(scores)

    if not np.any(finite):
        first_failure = next(
            (reason for reason in failure_reasons if reason is not None),
            None,
        )

        if first_failure is not None:
            raise RuntimeError(
                "All kernel-grid candidates failed during "
                "inner tuning. "
                f"First failure: {first_failure}"
            )

        raise RuntimeError("All kernel-grid candidates failed during inner tuning.")

    best_index = int(np.argmin(scores))

    return (
        dict(candidates[best_index]),
        float(scores[best_index]),
        len(candidates),
        int(np.sum(finite)),
    )


# ============================================================
# Tuning dispatcher
# ============================================================


def tune_model(
    df_outer_train,
    *,
    estimator,
    model_name,
    kernel,
    inner_seed,
):

    if model_name in PARAMETRIC_MODELS:

        return (
            {},
            None,
            "none",
            0,
            0,
        )

    if model_name in KERNEL_MODELS:

        (
            best_params,
            best_inner_ibs,
            tuning_evaluations,
            valid_tuning_evaluations,
        ) = tune_kernel_grid(
            df_outer_train,
            estimator=estimator,
            model_name=model_name,
            kernel=kernel,
            inner_seed=inner_seed,
        )

        return (
            best_params,
            best_inner_ibs,
            "grid",
            tuning_evaluations,
            valid_tuning_evaluations,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Split identity and deterministic seeds
# ============================================================


def index_hash(
    indices,
):
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    return hashlib.sha256(indices.tobytes()).hexdigest()[:16]


def derive_inner_seed(
    split_index,
):
    return int(
        np.random.SeedSequence(
            [
                MASTER_SEED,
                split_index,
                1,
            ]
        ).generate_state(
            1
        )[0]
    )


def _finite_or_none(
    value,
):
    if value is None:
        return None

    value = float(value)

    if np.isfinite(value):
        return value

    return None


# ============================================================
# Checkpoints
# ============================================================


def experiment_prefix(
    task_id,
    model_name,
    kernel,
):

    if kernel is None:
        return f"melanoma_fosp_task" f"{task_id}_" f"{model_name}"

    return f"melanoma_fosp_task" f"{task_id}_" f"{model_name}_" f"kernel_{kernel}"


def checkpoint_path(
    task_id,
    model_name,
    kernel,
    split_index,
):
    prefix = experiment_prefix(
        task_id,
        model_name,
        kernel,
    )

    return (
        OUTPUT_DIR / "checkpoints" / prefix / (f"split_" f"{split_index:03d}" f".json")
    )


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


def load_valid_checkpoint(
    path,
    *,
    split_index,
    model_name,
    kernel,
    train_hash,
    test_hash,
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
        or row.get("experiment_signature") != EXPERIMENT_SIGNATURE
    ):
        return None

    return row


# ============================================================
# One outer split
# ============================================================


def run_outer_split_worker(
    df,
    model_name,
    kernel,
    split_index,
    train_idx,
    test_idx,
    checkpoint_file,
):
    started = perf_counter()

    train_idx = np.asarray(
        train_idx,
        dtype=np.int64,
    )

    test_idx = np.asarray(
        test_idx,
        dtype=np.int64,
    )

    train_hash = index_hash(train_idx)
    test_hash = index_hash(test_idx)
    inner_seed = derive_inner_seed(split_index)

    estimator = MODELS[model_name]

    kernel_label = kernel if kernel is not None else "parametric"

    try:
        df_outer_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        (
            best_params,
            best_inner_ibs,
            tuning_method,
            tuning_evaluations,
            valid_tuning_evaluations,
        ) = tune_model(
            df_outer_train,
            estimator=estimator,
            model_name=model_name,
            kernel=kernel,
            inner_seed=inner_seed,
        )

        outer_preprocessor = make_preprocessor()

        (
            X_outer_train,
            t_outer_train,
            d_outer_train,
        ) = outer_preprocessor.fit(df_outer_train)

        (
            X_test,
            t_test,
            d_test,
        ) = outer_preprocessor.transform(df_test)

        (
            t_grid,
            horizon,
        ) = make_evaluation_grid(
            t_outer_train,
            d_outer_train,
            points=OUTER_GRID_POINTS,
        )

        final_model = build_model(
            estimator,
            kernel,
            best_params,
        )

        fit_cure_model(
            final_model,
            X_outer_train,
            t_outer_train,
            d_outer_train,
        )

        S_test = final_model.predict_survival(
            X_test,
            t_grid,
        )

        test_ibs = integrated_brier_score(
            S_test,
            t_test,
            d_test,
            t_outer_train,
            d_outer_train,
            t_grid,
        )

        test_cindex = uno_c_index_rmst(
            S_pred=S_test,
            t_eval=t_test,
            delta_eval=d_test,
            t_train=t_outer_train,
            delta_train=d_outer_train,
            t_grid=t_grid,
            tau=horizon,
        )

        p_cure_hat = final_model.predict_cure_probability(X_test)

        (
            test_auc,
            _,
            _,
            _,
        ) = auc_cure(p_cure_hat)

        mean_predicted_cure = float(np.mean(p_cure_hat))

        if not np.isfinite(test_ibs):
            raise FloatingPointError("Non-finite outer-test integrated Brier score.")

        if not np.isfinite(mean_predicted_cure):
            raise FloatingPointError("Non-finite mean predicted cure probability.")

        elapsed = perf_counter() - started

        converged = bool(
            getattr(
                final_model,
                "converged_",
                True,
            )
        )

        n_iter = getattr(
            final_model,
            "n_iter_",
            None,
        )

        if n_iter is not None:
            try:
                n_iter = int(n_iter)
            except (
                TypeError,
                ValueError,
            ):
                n_iter = None

        row = {
            "split": int(split_index),
            "model_name": model_name,
            "kernel": kernel_label,
            "followup": FOLLOWUP,
            "experiment_signature": EXPERIMENT_SIGNATURE,
            "status": "ok",
            "failure_reason": None,
            "converged": converged,
            "n_iter": n_iter,
            "training_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "n_features": int(X_outer_train.shape[1]),
            "training_event_rate": float(np.mean(d_outer_train)),
            "test_event_rate": float(np.mean(d_test)),
            "train_index_hash": train_hash,
            "test_index_hash": test_hash,
            "inner_seed": int(inner_seed),
            "horizon": float(horizon),
            "integrated_brier": float(test_ibs),
            "uno_c_index": (_finite_or_none(test_cindex)),
            "cure_auc": (_finite_or_none(test_auc)),
            "mean_predicted_cure": (mean_predicted_cure),
            "best_inner_ibs": (_finite_or_none(best_inner_ibs)),
            "best_params": json.dumps(
                best_params,
                sort_keys=True,
            ),
            "tuning_method": tuning_method,
            "tuning_evaluations": int(tuning_evaluations),
            "valid_tuning_evaluations": int(valid_tuning_evaluations),
            "runtime_seconds": float(elapsed),
        }

    except Exception as error:
        elapsed = perf_counter() - started

        row = {
            "split": int(split_index),
            "model_name": model_name,
            "kernel": kernel_label,
            "followup": FOLLOWUP,
            "experiment_signature": EXPERIMENT_SIGNATURE,
            "status": "failed",
            "failure_reason": (f"{type(error).__name__}: " f"{error}"),
            "converged": False,
            "n_iter": None,
            "training_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "n_features": None,
            "training_event_rate": float(df.iloc[train_idx][STATUS_COL].mean()),
            "test_event_rate": float(df.iloc[test_idx][STATUS_COL].mean()),
            "train_index_hash": train_hash,
            "test_index_hash": test_hash,
            "inner_seed": int(inner_seed),
            "horizon": None,
            "integrated_brier": None,
            "uno_c_index": None,
            "cure_auc": None,
            "mean_predicted_cure": None,
            "best_inner_ibs": None,
            "best_params": None,
            "tuning_method": ("grid" if model_name in KERNEL_MODELS else "none"),
            "tuning_evaluations": None,
            "valid_tuning_evaluations": None,
            "runtime_seconds": float(elapsed),
        }

    atomic_write_json(
        checkpoint_file,
        row,
    )

    return row


# ============================================================
# Parallel workers
# ============================================================


def get_worker_count():
    value = os.environ.get(
        "GGKM_OUTER_WORKERS",
        os.environ.get(
            "PBS_NCPUS",
            "1",
        ),
    )

    try:
        workers = int(value)

    except ValueError:
        workers = 1

    return max(
        1,
        workers,
    )


# ============================================================
# 50 paired repeated holdouts
# ============================================================


def repeated_holdout_melanoma_fosp(
    df,
    *,
    task_id,
    model_name,
    kernel,
):
    df = df.reset_index(drop=True)

    outer_splitter = StratifiedShuffleSplit(
        n_splits=N_OUTER_SPLITS,
        test_size=OUTER_TEST_SIZE,
        random_state=MASTER_SEED,
    )

    outer_splits = list(
        outer_splitter.split(
            df,
            df[STATUS_COL],
        )
    )

    rows = []
    pending = []

    for (
        split_index,
        (
            train_idx,
            test_idx,
        ),
    ) in enumerate(outer_splits):
        train_hash = index_hash(train_idx)

        test_hash = index_hash(test_idx)

        checkpoint_file = checkpoint_path(
            task_id,
            model_name,
            kernel,
            split_index,
        )

        checkpoint = load_valid_checkpoint(
            checkpoint_file,
            split_index=split_index,
            model_name=model_name,
            kernel=kernel,
            train_hash=train_hash,
            test_hash=test_hash,
        )

        if checkpoint is not None:
            rows.append(checkpoint)

        else:
            pending.append(
                (
                    split_index,
                    np.asarray(
                        train_idx,
                        dtype=np.int64,
                    ),
                    np.asarray(
                        test_idx,
                        dtype=np.int64,
                    ),
                    checkpoint_file,
                )
            )

    workers = min(
        get_worker_count(),
        max(
            1,
            len(pending),
        ),
    )

    print(
        f"Completed checkpoints: " f"{len(rows)}/" f"{N_OUTER_SPLITS}",
        flush=True,
    )

    print(
        f"Pending outer splits: " f"{len(pending)}",
        flush=True,
    )

    print(
        f"Outer-split workers: " f"{workers}",
        flush=True,
    )

    if pending:

        description = model_name if kernel is None else f"{model_name}-{kernel}"

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_split = {
                executor.submit(
                    run_outer_split_worker,
                    df,
                    model_name,
                    kernel,
                    split_index,
                    train_idx,
                    test_idx,
                    checkpoint_file,
                ): split_index
                for (
                    split_index,
                    train_idx,
                    test_idx,
                    checkpoint_file,
                ) in pending
            }

            progress = tqdm(
                as_completed(future_to_split),
                total=len(future_to_split),
                desc=description,
            )

            for future in progress:
                split_index = future_to_split[future]

                try:
                    row = future.result()

                except Exception as error:
                    raise RuntimeError(
                        "Outer worker crashed "
                        f"for split "
                        f"{split_index}: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    ) from error

                rows.append(row)

                if row["status"] == "ok" and row["integrated_brier"] is not None:
                    progress.set_postfix(
                        split=split_index,
                        ibs=(f"{row['integrated_brier']:.4f}"),
                        refresh=False,
                    )

                else:
                    progress.set_postfix(
                        split=split_index,
                        status="failed",
                        refresh=False,
                    )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            "split",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(results) != N_OUTER_SPLITS:
        raise RuntimeError(
            "The repeated-holdout run did not produce "
            f"{N_OUTER_SPLITS} split rows; "
            f"received {len(results)}."
        )

    if results["split"].nunique() != N_OUTER_SPLITS:
        raise RuntimeError(
            "Duplicate or missing outer split identifiers were detected."
        )

    return results


# ============================================================
# Summary
# ============================================================


def summarize_results(
    results,
):
    valid = results.loc[results["status"] == "ok"].copy()

    def metric_summary(
        column,
    ):
        values = pd.to_numeric(
            valid[column],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            return (
                np.nan,
                np.nan,
            )

        mean = float(values.mean())

        sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan

        return (
            mean,
            sd,
        )

    (
        mean_ibs,
        sd_ibs,
    ) = metric_summary("integrated_brier")

    (
        mean_cindex,
        sd_cindex,
    ) = metric_summary("uno_c_index")

    (
        mean_auc,
        sd_auc,
    ) = metric_summary("cure_auc")

    (
        mean_cure,
        sd_cure,
    ) = metric_summary("mean_predicted_cure")

    runtime = pd.to_numeric(
        valid["runtime_seconds"],
        errors="coerce",
    ).dropna()

    mean_runtime = float(runtime.mean()) if len(runtime) else np.nan

    maximum_runtime = float(runtime.max()) if len(runtime) else np.nan

    return {
        "model_name": (results["model_name"].iloc[0]),
        "kernel": (results["kernel"].iloc[0]),
        "splits": int(len(results)),
        "successful": int((results["status"] == "ok").sum()),
        "failed": int((results["status"] == "failed").sum()),
        "converged": int(results["converged"].fillna(False).astype(bool).sum()),
        "mean_integrated_brier": (mean_ibs),
        "split_sd_integrated_brier": (sd_ibs),
        "mean_uno_c_index": (mean_cindex),
        "split_sd_uno_c_index": (sd_cindex),
        "mean_cure_auc": (mean_auc),
        "split_sd_cure_auc": (sd_auc),
        "mean_predicted_cure": (mean_cure),
        "split_sd_mean_predicted_cure": (sd_cure),
        "mean_runtime_seconds": (mean_runtime),
        "maximum_runtime_seconds": (maximum_runtime),
    }


# ============================================================
# Run experiment
# ============================================================


def run_experiment(
    df,
    *,
    task_id,
    model_name,
    kernel,
):
    return repeated_holdout_melanoma_fosp(
        df,
        task_id=task_id,
        model_name=model_name,
        kernel=kernel,
    )


# ============================================================
# Save final aggregate files
# ============================================================


def save_results(
    results,
    task_id,
    model_name,
    kernel,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = experiment_prefix(
        task_id,
        model_name,
        kernel,
    )

    split_file = OUTPUT_DIR / f"splits_{prefix}.csv"

    summary_file = OUTPUT_DIR / f"summary_{prefix}.csv"

    results.to_csv(
        split_file,
        index=False,
    )

    summary = summarize_results(results)

    pd.DataFrame([summary]).to_csv(
        summary_file,
        index=False,
    )

    return (
        split_file,
        summary_file,
        summary,
    )


# ============================================================
# Main
# ============================================================


def main():
    start = time.time()

    task_id = int(os.environ["PBS_ARRAY_INDEX"])

    experiments = build_experiments()

    exp = get_experiment(
        experiments,
        task_id,
    )

    model_name = exp["model_name"]

    kernel = exp["kernel"]

    print(
        "==================================",
        flush=True,
    )

    print(
        f"Task {task_id}",
        flush=True,
    )

    if kernel is None:

        print(
            f"Model={model_name}",
            flush=True,
        )

    else:

        print(
            f"Model={model_name}, " f"kernel={kernel}",
            flush=True,
        )

    print(
        "",
        flush=True,
    )

    print(
        "FOSP melanoma evaluation design:",
        flush=True,
    )

    print(
        f"  Outer repeated holdouts: " f"{N_OUTER_SPLITS}",
        flush=True,
    )

    print(
        f"  Outer test fraction: " f"{OUTER_TEST_SIZE:.2f}",
        flush=True,
    )

    print(
        f"  Inner folds: " f"{N_INNER_SPLITS}",
        flush=True,
    )

    print(
        f"  Outer workers: " f"{get_worker_count()}",
        flush=True,
    )

    print(
        f"  Inner grid points: " f"{INNER_GRID_POINTS}",
        flush=True,
    )

    print(
        f"  Outer grid points: " f"{OUTER_GRID_POINTS}",
        flush=True,
    )

    print(
        f"  Time quantile: " f"{OBSERVED_TIME_QUANTILE}",
        flush=True,
    )

    print(
        f"  Minimum censoring survival: " f"{MINIMUM_CENSORING_SURVIVAL}",
        flush=True,
    )

    print(
        f"  Master seed: " f"{MASTER_SEED}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        f"Experiment signature: {EXPERIMENT_SIGNATURE}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "FOSP preprocessing:",
        flush=True,
    )

    print(
        f"  Follow-up: {FOLLOWUP} " f"({TIME_COL}, {STATUS_COL})",
        flush=True,
    )

    print(
        f"  Include treatments: {INCLUDE_TREATMENTS}",
        flush=True,
    )

    print(
        f"  Include extra treatments: {INCLUDE_EXTRA_TREATMENTS}",
        flush=True,
    )

    print(
        f"  Include topography: {INCLUDE_TOPOGRAPHY}",
        flush=True,
    )

    print(
        f"  Include morphology: {INCLUDE_MORPHOLOGY}",
        flush=True,
    )

    print(
        "  Minimum morphology frequency: " f"{MIN_MORPHOLOGY_FREQUENCY}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    if model_name in KERNEL_MODELS:

        print(
            "Hyperparameter tuning: exhaustive grid search",
            flush=True,
        )

        candidates = kernel_parameter_grid(
            model_name,
            kernel,
        )

        print(
            f"  Candidates per outer split: " f"{len(candidates)}",
            flush=True,
        )

        for candidate in candidates:
            print(
                f"  {candidate}",
                flush=True,
            )

    else:

        print(
            "Hyperparameter tuning: none",
            flush=True,
        )

    print(
        "==================================",
        flush=True,
    )

    df = load_data()

    print(
        f"Number of observations: " f"{len(df)}",
        flush=True,
    )

    print(
        f"Number of events ({STATUS_COL}): " f"{int(df[STATUS_COL].sum())}",
        flush=True,
    )

    print(
        f"Number censored ({STATUS_COL}): " f"{int((1 - df[STATUS_COL]).sum())}",
        flush=True,
    )

    results = run_experiment(
        df,
        task_id=task_id,
        model_name=model_name,
        kernel=kernel,
    )

    (
        split_file,
        summary_file,
        summary,
    ) = save_results(
        results,
        task_id,
        model_name,
        kernel,
    )

    elapsed = time.time() - start

    print(
        "",
        flush=True,
    )

    print(
        "==================================",
        flush=True,
    )

    print(
        "Finished",
        flush=True,
    )

    print(
        f"Splits: " f"{summary['splits']}",
        flush=True,
    )

    print(
        f"Successful: " f"{summary['successful']}",
        flush=True,
    )

    print(
        f"Failed: " f"{summary['failed']}",
        flush=True,
    )

    print(
        f"Converged: " f"{summary['converged']}/" f"{summary['splits']}",
        flush=True,
    )

    print(
        (
            "IPCW IBS: "
            f"{summary['mean_integrated_brier']:.4f} "
            "("
            f"{summary['split_sd_integrated_brier']:.4f}"
            ")"
        ),
        flush=True,
    )

    print(
        (
            "Mean predicted cure: "
            f"{summary['mean_predicted_cure']:.4f} "
            "("
            f"{summary['split_sd_mean_predicted_cure']:.4f}"
            ")"
        ),
        flush=True,
    )

    print(
        (
            "Runtime per split: "
            f"{summary['mean_runtime_seconds']:.2f} "
            "("
            f"{summary['maximum_runtime_seconds']:.2f}"
            ") seconds"
        ),
        flush=True,
    )

    print(
        f"Total elapsed seconds: " f"{elapsed:.2f}",
        flush=True,
    )

    print(
        f"Total elapsed minutes: " f"{elapsed / 60:.2f}",
        flush=True,
    )

    print(
        f"Total elapsed hours: " f"{elapsed / 3600:.2f}",
        flush=True,
    )

    print(
        f"Saved split results: " f"{split_file}",
        flush=True,
    )

    print(
        f"Saved summary: " f"{summary_file}",
        flush=True,
    )

    print(
        "==================================",
        flush=True,
    )


if __name__ == "__main__":
    main()
