"""Read breast-cancer holdout results using the experiment's cohort and splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import melanoma_plot_utils as shared

PROJECT_ROOT = shared.PROJECT_ROOT
DATA_PATH = PROJECT_ROOT / "data" / "cancer_de_mama_rhc_e_fap.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "breast_cancer_results" / "grid_search_paired_50"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "review"
N_OUTER_SPLITS = 50
OUTER_TEST_SIZE = 0.30
MASTER_SEED = 20260817
KERNELS = [
    "linear", "rbf", "laplacian", "exponential", "cauchy", "sigmoid",
    "polynomial", "wavelet",
]
KERNEL_LABELS = {**shared.KERNEL_LABELS, "sigmoid": "Sigmoid"}
index_hash = shared.index_hash
parse_best_params = shared.parse_best_params
read_split_file = shared.read_split_file


def load_breast_cancer(data_path=DATA_PATH):
    """Match ``run_breast_cancer_experiment.load_data`` without changing row IDs."""
    frame = (
        pd.read_csv(data_path)
        .drop(columns=[
            "RhcDataDeNascimento", "Limitacao", "DiffEmDiasEntreRhcDtNascimentoDtObito",
            "Instrução", "EstCong", "HistFamCan", "SimDataDeNascimento", "CosemsCodIbge",
            "Renda", "Macroregião", "DIFFTRATCONS1", "DIFFDIAGTRAT", "DIFFDIAGCONS1",
            "Diagnostico", "Raça", "AnoCateg", "Idade", "Ano", "Estadiamento",
        ])
        .fillna({
            "SimRecordID": "vivo", "RhcRecordID": "desconhecido",
            "SimCausaBasicaCategoria": "vivo", "SimCausaBasica": "vivo",
        })
        .rename(columns={"Status": "delta"})
        .assign(delta=lambda data: (data.delta == "Morte por cancer de Mama").astype(int))
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(frame) < 2:
        raise ValueError("Breast-cancer data contain fewer than two observations.")
    if frame["delta"].nunique() != 2:
        raise ValueError("Breast-cancer data must contain both events and censored observations.")
    if np.any(~np.isfinite(frame["Tempo"])):
        raise ValueError("Breast-cancer data contain non-finite survival times.")
    if np.any(frame["Tempo"] < 0.0):
        raise ValueError("Breast-cancer data contain negative survival times.")
    frame.loc[frame["Tempo"] == 0.0, "Tempo"] = 0.5
    return frame


def construct_outer_splits(
    frame, n_splits=N_OUTER_SPLITS, test_size=OUTER_TEST_SIZE, random_state=MASTER_SEED,
):
    return shared.construct_outer_splits(frame, n_splits, test_size, random_state)


def unique_result_file(pattern, results_dir=RESULTS_DIR):
    return shared.unique_result_file(pattern, results_dir)


def verify_split_hashes(frame, outer_splits, model_name):
    try:
        shared.verify_split_hashes(frame, outer_splits, model_name)
    except RuntimeError as error:
        raise RuntimeError(str(error).replace("melanoma CSV", "breast-cancer CSV")) from error


def choose_best_kernel(model_name, results_dir=RESULTS_DIR, *, kernels=None):
    """Choose the highest mean saved cure AUC, with IBS then name breaking ties."""
    rows = []
    for kernel in KERNELS if kernels is None else kernels:
        path = unique_result_file(
            f"splits_breast_cancer_task*_{model_name}_kernel_{kernel}.csv", results_dir,
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
        raise RuntimeError(f"No successful breast-cancer GG-KM cure-AUC results for {model_name}.")
    return pd.DataFrame(rows).sort_values(
        ["mean_auc", "mean_ibs", "kernel"], ascending=[False, True, True], kind="mergesort",
    ).iloc[0].to_dict()


def load_split_predictions(
    row, *, results_dir=RESULTS_DIR, project_root=PROJECT_ROOT,
    expected_train_indices=None, expected_test_indices=None,
):
    """Reuse archive validation and portable paths with breast-cancer metadata."""
    if hasattr(row, "keys"):
        row = dict(row)
    elif hasattr(row, "_asdict"):
        row = dict(row._asdict())
    else:
        row = vars(row).copy()
    if not shared._present(row.get("_result_file")) and shared._present(row.get("model_name")):
        suffix = str(row["model_name"])
        kernel = row.get("kernel")
        if shared._present(kernel) and str(kernel) not in {"parametric", "none", "None"}:
            suffix += f"_kernel_{kernel}"
        matches = sorted(Path(results_dir).glob(f"splits_breast_cancer_task*_{suffix}.csv"))
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous breast-cancer result files for {suffix}.")
        if matches:
            row["_result_file"] = str(matches[0].resolve())
        else:
            filename = f"split_{int(row['split']):03d}.npz"
            archives = sorted((Path(results_dir) / "plot_data").glob(
                f"breast_cancer_task*_{suffix}/{filename}"
            ))
            if len(archives) > 1:
                raise RuntimeError(f"Ambiguous breast-cancer prediction archives for {suffix}.")
            if archives:
                row["_result_file"] = str(
                    Path(results_dir) / f"splits_{archives[0].parent.name}.csv"
                )
    try:
        return shared.load_split_predictions(
            row, results_dir=results_dir, project_root=project_root,
            expected_train_indices=expected_train_indices, expected_test_indices=expected_test_indices,
        )
    except ValueError as error:
        raise ValueError(str(error).replace("melanoma prediction", "breast-cancer prediction")) from error


def full_cohort_times(frame):
    """The breast-cancer preprocessor preserves recorded ``Tempo`` units."""
    time = frame["Tempo"].to_numpy(dtype=float)
    event = frame["delta"].to_numpy(dtype=int)
    if np.any(~np.isfinite(time)) or np.any(time < 0):
        raise ValueError("Breast-cancer follow-up times must be finite and nonnegative.")
    return time, event
