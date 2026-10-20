"""Breast-cancer settings and repeated-holdout experiment entry point."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.auto import tqdm

from skcure.utils.estimator import fit_cure_model
from skcure.metrics import auc_cure
from scripts.preprocessing import BreastCancerSurvivalPreprocessor
from evaluate.metrics import (
    integrated_brier_score,
    uno_c_index_rmst,
)

from scripts.experiment_utils import (
    artifacts,
    evaluation,
    models as experiment_models,
    reporting,
    results as experiment_results,
    runtime,
    tuning,
)
from scripts.experiment_utils.artifacts import (
    array_or_empty as _array_or_empty,
    atomic_write_json,
    atomic_write_npz,
    finite_or_none as _finite_or_none,
    fitted_model_arrays as _fitted_model_arrays,
    index_hash,
    preprocessed_feature_names as _preprocessed_feature_names,
)
from scripts.experiment_utils.models import build_model
from scripts.experiment_utils.results import summarize_results
from scripts.experiment_utils.runtime import get_worker_count

MODELS = experiment_models.make_model_registry(negative_binomial_max_em_iter=10000)

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

KERNEL_MODELS = set(experiment_models.KERNEL_MODEL_NAMES)
PARAMETRIC_MODELS = set(experiment_models.PARAMETRIC_MODEL_NAMES)
SPLINE_MODELS = set(experiment_models.SPLINE_MODEL_NAMES)


N_OUTER_SPLITS = 50
OUTER_TEST_SIZE = 0.30

N_INNER_SPLITS = 4

INNER_GRID_POINTS = 40
OUTER_GRID_POINTS = 60

OBSERVED_TIME_QUANTILE = 0.90
MINIMUM_CENSORING_SURVIVAL = 0.05

MASTER_SEED = 20260817


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


SPLINE_PARAMETER_GRID = [
    {
        "n_knots": 3,
        "degree": 3,
        "lambda_reg": 0.10,
    },
    {
        "n_knots": 3,
        "degree": 3,
        "lambda_reg": 1.00,
    },
    {
        "n_knots": 5,
        "degree": 3,
        "lambda_reg": 0.10,
    },
    {
        "n_knots": 5,
        "degree": 3,
        "lambda_reg": 1.00,
    },
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "breast_cancer_results" / "grid_search_paired_50"


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
    df = (
        pd.read_csv(DATA_DIR / "cancer_de_mama_rhc_e_fap.csv")
        .drop(
            columns=[
                "RhcDataDeNascimento",
                "Limitacao",
                "DiffEmDiasEntreRhcDtNascimentoDtObito",
                "Instrução",
                "EstCong",
                "HistFamCan",
                "SimDataDeNascimento",
                "CosemsCodIbge",
                "Renda",
                "Macroregião",
                "DIFFTRATCONS1",
                "DIFFDIAGTRAT",
                "DIFFDIAGCONS1",
                "Diagnostico",
                "Raça",
                "AnoCateg",
                "Idade",
                "Ano",
                "Estadiamento",
            ]
        )
        .fillna(
            {
                "SimRecordID": "vivo",
                "RhcRecordID": "desconhecido",
                "SimCausaBasicaCategoria": "vivo",
                "SimCausaBasica": "vivo",
            }
        )
        .rename(
            columns={
                "Status": "delta",
            }
        )
        .assign(delta=lambda x: (x.delta == "Morte por cancer de Mama").astype(int))
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(df) < 2:
        raise ValueError("Breast-cancer data contain fewer than two observations.")

    if df["delta"].nunique() != 2:
        raise ValueError(
            "Breast-cancer data must contain both events " "and censored observations."
        )

    if np.any(~np.isfinite(df["Tempo"])):
        raise ValueError("Breast-cancer data contain non-finite survival times.")

    if np.any(df["Tempo"] < 0.0):
        raise ValueError("Breast-cancer data contain negative survival times.")

    df.loc[df["Tempo"] == 0.0, "Tempo"] = 0.5

    return df


def censoring_reliable_horizon(t, delta, minimum_survival=MINIMUM_CENSORING_SURVIVAL):
    return evaluation.censoring_reliable_horizon(t, delta, minimum_survival)


def make_evaluation_grid(t, delta, *, points):
    return evaluation.make_evaluation_grid(
        t, delta, points=points,
        observed_time_quantile=OBSERVED_TIME_QUANTILE,
        minimum_censoring_survival=MINIMUM_CENSORING_SURVIVAL,
        censoring_horizon=censoring_reliable_horizon,
    )


def fixed_kernel_model_params(model_name):
    return experiment_models.fixed_kernel_model_params(model_name, binomial_k=BINOMIAL_K)


def kernel_parameter_grid(model_name, kernel):
    return experiment_models.kernel_parameter_grid(
        model_name, kernel,
        binomial_k=BINOMIAL_K,
        base_params=fixed_kernel_model_params(model_name),
        penalties=KERNEL_PENALTIES,
        gammas=KERNEL_GAMMAS,
        polynomial_degrees=POLYNOMIAL_DEGREES,
        polynomial_coef0=POLYNOMIAL_COEF0,
        wavelet_a_values=WAVELET_A_VALUES,
        sigmoid_coef0=SIGMOID_COEF0,
        allow_sigmoid=True,
    )


def prepare_inner_splits(df_outer_train, *, inner_seed):
    return tuning.prepare_inner_splits(
        df_outer_train, inner_seed=inner_seed,
        n_inner_splits=N_INNER_SPLITS,
        preprocessor_factory=BreastCancerSurvivalPreprocessor,
        make_evaluation_grid=make_evaluation_grid,
        inner_grid_points=INNER_GRID_POINTS,
        splitter_factory=StratifiedKFold,
    )


def score_params_on_inner_splits(inner_splits, *, estimator, kernel, params):
    return tuning.score_params_on_inner_splits(
        inner_splits, estimator=estimator, kernel=kernel, params=params,
        n_inner_splits=N_INNER_SPLITS,
        build_model=build_model,
        fit_model=fit_cure_model,
        integrated_brier_score=integrated_brier_score,
    )


def tune_kernel_grid(df_outer_train, *, estimator, model_name, kernel, inner_seed):
    return tuning.tune_kernel_grid(
        df_outer_train, estimator=estimator, model_name=model_name,
        kernel=kernel, inner_seed=inner_seed,
        prepare_inner_splits=prepare_inner_splits,
        kernel_parameter_grid=kernel_parameter_grid,
        score_params_on_inner_splits=score_params_on_inner_splits,
    )


def tune_spline_grid(df_outer_train, *, estimator, inner_seed):
    return tuning.tune_spline_grid(
        df_outer_train, estimator=estimator, inner_seed=inner_seed,
        prepare_inner_splits=prepare_inner_splits,
        score_params_on_inner_splits=score_params_on_inner_splits,
        spline_parameter_grid=SPLINE_PARAMETER_GRID,
        report_failures=True,
    )


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

    if model_name in SPLINE_MODELS:

        (
            best_params,
            best_inner_ibs,
            tuning_evaluations,
            valid_tuning_evaluations,
        ) = tune_spline_grid(
            df_outer_train,
            estimator=estimator,
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


def derive_inner_seed(split_index):
    return runtime.derive_inner_seed(split_index, master_seed=MASTER_SEED)


def experiment_prefix(
    task_id,
    model_name,
    kernel,
):

    if kernel is None:
        return f"breast_cancer_task" f"{task_id}_" f"{model_name}"

    return f"breast_cancer_task" f"{task_id}_" f"{model_name}_" f"kernel_{kernel}"


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


def plot_data_path(
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

    return OUTPUT_DIR / "plot_data" / prefix / f"split_{split_index:03d}.npz"


def _portable_path(path):
    return artifacts.portable_path(path, project_root=PROJECT_ROOT)


def _resolve_saved_path(path):
    return artifacts.resolve_saved_path(path, project_root=PROJECT_ROOT)


def load_valid_checkpoint(path, *, split_index, model_name, kernel, train_hash, test_hash):
    return artifacts.load_valid_checkpoint(
        path, split_index=split_index, model_name=model_name, kernel=kernel,
        train_hash=train_hash, test_hash=test_hash,
        resolve_path=_resolve_saved_path,
    )


def run_outer_split_worker(
    df,
    model_name,
    kernel,
    split_index,
    train_idx,
    test_idx,
    checkpoint_file,
    plot_data_file,
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

        outer_preprocessor = BreastCancerSurvivalPreprocessor()

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
            test_tpr,
            test_fpr,
            cure_thresholds,
        ) = auc_cure(p_cure_hat)

        eta_test = final_model.predict_eta(X_test)
        risk_test = final_model.predict(X_test)

        mean_predicted_cure = float(np.mean(p_cure_hat))

        if not np.isfinite(test_ibs):
            raise FloatingPointError("Non-finite outer-test integrated Brier score.")

        if not np.isfinite(mean_predicted_cure):
            raise FloatingPointError("Non-finite mean predicted cure probability.")

        if np.any(~np.isfinite(S_test)):
            raise FloatingPointError("Non-finite outer-test survival probabilities.")

        if np.any(~np.isfinite(p_cure_hat)):
            raise FloatingPointError("Non-finite outer-test cure probabilities.")

        if np.any(~np.isfinite(eta_test)) or np.any(~np.isfinite(risk_test)):
            raise FloatingPointError("Non-finite outer-test risk predictions.")

        plot_payload = {
            "split": np.asarray(split_index, dtype=np.int64),
            "model_name": np.asarray(model_name),
            "kernel": np.asarray(kernel_label),
            "train_indices": train_idx,
            "test_indices": test_idx,
            "train_time": np.asarray(t_outer_train, dtype=float),
            "train_event": np.asarray(d_outer_train, dtype=float),
            "test_time": np.asarray(t_test, dtype=float),
            "test_event": np.asarray(d_test, dtype=float),
            "evaluation_time_grid": np.asarray(t_grid, dtype=float),
            "evaluation_horizon": np.asarray(horizon, dtype=float),
            "survival_test": np.asarray(S_test, dtype=float),
            "cure_probability_test": np.asarray(p_cure_hat, dtype=float),
            "eta_test": np.asarray(eta_test, dtype=float),
            "risk_test": np.asarray(risk_test, dtype=float),
            "cure_thresholds": _array_or_empty(cure_thresholds),
            "cure_tpr": _array_or_empty(test_tpr),
            "cure_fpr": _array_or_empty(test_fpr),
            "X_test": np.asarray(X_test, dtype=float),
            "feature_names": _preprocessed_feature_names(
                outer_preprocessor,
                X_test.shape[1],
            ),
            "best_params_json": np.asarray(json.dumps(best_params, sort_keys=True)),
            **_fitted_model_arrays(final_model),
        }

        atomic_write_npz(
            plot_data_file,
            **plot_payload,
        )

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
            "status": "ok",
            "failure_reason": None,
            "converged": converged,
            "n_iter": n_iter,
            "training_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
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
            "plot_data_file": _portable_path(plot_data_file),
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

        try:
            Path(plot_data_file).unlink(missing_ok=True)
        except OSError:
            pass

        row = {
            "split": int(split_index),
            "model_name": model_name,
            "kernel": kernel_label,
            "status": "failed",
            "failure_reason": (f"{type(error).__name__}: " f"{error}"),
            "converged": False,
            "n_iter": None,
            "training_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "training_event_rate": float(df.iloc[train_idx]["delta"].mean()),
            "test_event_rate": float(df.iloc[test_idx]["delta"].mean()),
            "train_index_hash": train_hash,
            "test_index_hash": test_hash,
            "inner_seed": int(inner_seed),
            "horizon": None,
            "integrated_brier": None,
            "uno_c_index": None,
            "cure_auc": None,
            "mean_predicted_cure": None,
            "plot_data_file": None,
            "best_inner_ibs": None,
            "best_params": None,
            "tuning_method": (
                "grid"
                if model_name in KERNEL_MODELS or model_name in SPLINE_MODELS
                else "none"
            ),
            "tuning_evaluations": None,
            "valid_tuning_evaluations": None,
            "runtime_seconds": float(elapsed),
        }

    atomic_write_json(
        checkpoint_file,
        row,
    )

    return row


def repeated_holdout_breast_cancer(
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
            df["delta"],
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

        plot_file = plot_data_path(
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
                    plot_file,
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
                    plot_file,
                ): split_index
                for (
                    split_index,
                    train_idx,
                    test_idx,
                    checkpoint_file,
                    plot_file,
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


def run_experiment(
    df,
    *,
    task_id,
    model_name,
    kernel,
):
    return repeated_holdout_breast_cancer(
        df,
        task_id=task_id,
        model_name=model_name,
        kernel=kernel,
    )


def save_results(results, task_id, model_name, kernel):
    return experiment_results.save_clinical_results(
        results, task_id, model_name, kernel,
        output_dir=OUTPUT_DIR,
        experiment_prefix=experiment_prefix,
        summarize_results=summarize_results,
    )


def main():
    start = time.time()
    task_id = int(os.environ["PBS_ARRAY_INDEX"])
    experiments = build_experiments()
    exp = get_experiment(experiments, task_id)
    model_name = exp["model_name"]
    kernel = exp["kernel"]

    reporting.print_clinical_header(task_id, model_name, kernel)
    reporting.print_clinical_design(
        dataset_name='Breast-cancer',
        outer_splits=N_OUTER_SPLITS,
        outer_test_size=OUTER_TEST_SIZE,
        inner_splits=N_INNER_SPLITS,
        worker_count=get_worker_count,
        inner_grid_points=INNER_GRID_POINTS,
        outer_grid_points=OUTER_GRID_POINTS,
        observed_time_quantile=OBSERVED_TIME_QUANTILE,
        minimum_censoring_survival=MINIMUM_CENSORING_SURVIVAL,
        master_seed=MASTER_SEED,
    )
    reporting.print_clinical_tuning(
        model_name=model_name,
        kernel=kernel,
        kernel_models=KERNEL_MODELS,
        spline_models=SPLINE_MODELS,
        kernel_parameter_grid=kernel_parameter_grid,
        spline_parameter_grid=SPLINE_PARAMETER_GRID,
        show_spline_candidate_count=True,
    )

    df = load_data()
    reporting.print_dataset_counts(df)
    results = run_experiment(
        df, task_id=task_id, model_name=model_name, kernel=kernel,
    )
    split_file, summary_file, summary = save_results(
        results, task_id, model_name, kernel,
    )
    elapsed = time.time() - start
    reporting.print_clinical_completion(summary, elapsed, split_file, summary_file)


if __name__ == "__main__":
    main()
