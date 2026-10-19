import optuna
import pandas as pd
import numpy as np
import time
import json

from tqdm.auto import tqdm

from scipy.stats import gengamma
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from skcure.metrics import uno_c_index_rmst, integrated_brier_score, auc_cure
from skcure.utils.estimator import fit_cure_model
from skcure.utils.optuna_utils import (
    _suggest_kernel_ranges,
    _kernel_ranges_from_best_params,
)
from ggkm_paper.evaluate.simulated_data import simulate_pcm, cure_probability
from ggkm_paper.evaluate.util_cross_validation import (
    _make_evaluation_grid,
    _build_estimator,
    _select_grid_parameters,
)


def _true_pcm_survival(
    X_raw,
    t_grid,
    *,
    method,
    generating_distribution,
    generating_phi,
    a=1.6,
    d=1.3,
    p=1.1,
):
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]

    if X_raw.shape[1] == 4:
        x3 = X_raw[:, 2]
        x4 = X_raw[:, 3]
    else:
        x3 = None
        x4 = None

    p_cure = cure_probability(
        x1=x1,
        x2=x2,
        method=method,
        x3=x3,
        x4=x4,
    )

    F = gengamma.cdf(
        t_grid,
        a=d / p,
        c=p,
        scale=a,
    )

    if generating_distribution == "poisson":
        theta = -np.log(p_cure)

        return np.exp(-theta[:, None] * F[None, :])

    if generating_distribution == "negative_binomial":
        phi = generating_phi

        theta = (p_cure ** (-phi) - 1.0) / phi

        return (1.0 + phi * theta[:, None] * F[None, :]) ** (-1.0 / phi)

    raise ValueError(
        "True survival diagnostics are currently "
        "implemented for Poisson and negative binomial."
    )


def cross_validate_pcm(
    n,
    method,
    estimator,
    kernel="rbf",
    n_outer_splits=5,
    n_inner_splits=4,
    n_trials=20,
    t_grid_points=50,
    random_state=42,
    bagging=False,
    bagging_estimator=None,
    kernels=None,
    estimator_name=None,
    optimize_kernel_ranges=False,
    lambda_reg_search_bounds=(1e-5, 1.0),
    gamma_search_bounds=(1e-3, 10.0),
    A_search_bounds=(1e-2, 200.0),
    degree_search_bounds=(2, 6),
    coef0_poly_search_bounds=(0.0, 5.0),
    coef0_sigmoid_search_bounds=(-5.0, 5.0),
    n_jobs=-1,
):

    df = simulate_pcm(n=n, method=method, seed=random_state)

    X = df[["x1", "x2"]].to_numpy()
    t = df["time"].to_numpy()
    delta = df["event"].to_numpy()

    p_cure_true = 1 - df["pi_x"].to_numpy()

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )
    total_trials = n_outer_splits * n_trials

    model_name = "Bagging" if bagging else kernel

    pbar = tqdm(total=total_trials, desc=f"{model_name} nested CV", unit="trial")

    all_test_ibs = []
    all_test_cindex = []
    all_test_auc = []
    all_test_tpr = []
    all_test_fpr = []
    all_bias = []
    all_mse = []
    all_best_params = []

    optuna_callback = lambda study, trial: pbar.update(1)

    for _, (train_idx, test_idx) in enumerate(outer_cv.split(X, delta)):

        X_outer_train, X_test = (X[train_idx], X[test_idx])
        t_outer_train, t_test = (t[train_idx], t[test_idx])
        d_outer_train, d_test = (delta[train_idx], delta[test_idx])
        p_cure_test_true = p_cure_true[test_idx]

        t_grid = np.linspace(
            np.percentile(t_outer_train, 5),
            np.percentile(t_outer_train, 95),
            t_grid_points,
        )

        inner_cv = StratifiedKFold(
            n_splits=n_inner_splits, shuffle=True, random_state=random_state
        )

        def objective(trial):

            params = {}

            if bagging:
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    50,
                    1000,
                )

                if optimize_kernel_ranges:
                    params.update(
                        _suggest_kernel_ranges(
                            trial,
                            lambda_reg_bounds=lambda_reg_search_bounds,
                            gamma_bounds=gamma_search_bounds,
                            degree_bounds=degree_search_bounds,
                            coef0_poly_bounds=coef0_poly_search_bounds,
                            coef0_sigmoid_bounds=coef0_sigmoid_search_bounds,
                        )
                    )

            else:

                params = {
                    "lambda_reg": trial.suggest_float(
                        "lambda_reg",
                        lambda_reg_search_bounds[0],
                        lambda_reg_search_bounds[1],
                        log=True,
                    ),
                }

                if estimator_name == "binomial":
                    params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
                elif estimator_name == "bernoulli":
                    params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

                if kernel in {
                    "rbf",
                    "gaussian",
                    "laplacian",
                    "exponential",
                    "cauchy",
                    "sigmoid",
                    "polynomial",
                }:
                    params["gamma"] = trial.suggest_float(
                        "gamma",
                        gamma_search_bounds[0],
                        gamma_search_bounds[1],
                        log=True,
                    )

                if kernel == "wavelet":
                    params["A"] = trial.suggest_float(
                        "A",
                        A_search_bounds[0],
                        A_search_bounds[1],
                        log=False,
                    )

                if kernel == "polynomial":
                    params["degree"] = trial.suggest_int(
                        "degree",
                        degree_search_bounds[0],
                        degree_search_bounds[1],
                    )

                    params["coef0"] = trial.suggest_float(
                        "coef0",
                        coef0_poly_search_bounds[0],
                        coef0_poly_search_bounds[1],
                    )

                if kernel == "sigmoid":
                    params["coef0"] = trial.suggest_float(
                        "coef0",
                        coef0_sigmoid_search_bounds[0],
                        coef0_sigmoid_search_bounds[1],
                    )

            val_scores = []

            for tr_idx, val_idx in inner_cv.split(X_outer_train, d_outer_train):

                X_tr = X_outer_train[tr_idx]
                X_val = X_outer_train[val_idx]

                t_tr = t_outer_train[tr_idx]
                t_val = t_outer_train[val_idx]

                d_tr = d_outer_train[tr_idx]
                d_val = d_outer_train[val_idx]

                t_grid_inner = np.linspace(
                    np.percentile(t_tr, 5), np.percentile(t_tr, 95), t_grid_points
                )

                try:
                    if bagging:

                        if bagging_estimator is None:
                            model = estimator(
                                kernels=kernels,
                                random_state=random_state,
                                **params,
                            )
                        else:
                            model = estimator(
                                estimator=bagging_estimator,
                                kernels=kernels,
                                random_state=random_state,
                                **params,
                            )

                    else:

                        model = estimator(kernel=kernel, **params)

                    fit_cure_model(model, X_tr, t_tr, d_tr)
                    S_pred = model.predict_survival(X_val, t_grid_inner)
                    ibs = integrated_brier_score(
                        S_pred, t_val, d_val, t_tr, d_tr, t_grid_inner
                    )
                    val_scores.append(ibs)

                except Exception as e:
                    print(f"Trial failed: {e}")
                    return 1.0

            return float(np.mean(val_scores))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[optuna_callback],
            n_jobs=n_jobs,
        )

        best_params = study.best_params
        all_best_params.append(best_params)

        if bagging:

            final_params = {"n_estimators": best_params["n_estimators"]}

            if optimize_kernel_ranges:
                final_params.update(_kernel_ranges_from_best_params(best_params))

            if bagging_estimator is None:
                final_model = estimator(
                    kernels=kernels,
                    random_state=random_state,
                    **final_params,
                )
            else:
                final_model = estimator(
                    estimator=bagging_estimator,
                    kernels=kernels,
                    random_state=random_state,
                    **final_params,
                )

        else:
            final_model = estimator(kernel=kernel, **best_params)

        fit_cure_model(final_model, X_outer_train, t_outer_train, d_outer_train)
        S_test = final_model.predict_survival(X_test, t_grid)

        test_ibs = integrated_brier_score(
            S_test, t_test, d_test, t_outer_train, d_outer_train, t_grid
        )
        all_test_ibs.append(test_ibs)

        test_cindex = uno_c_index_rmst(
            S_pred=S_test,
            t_eval=t_test,
            delta_eval=d_test,
            t_train=t_outer_train,
            delta_train=d_outer_train,
            t_grid=t_grid,
            tau=t_grid[-1],
        )
        all_test_cindex.append(test_cindex)

        p_cure_hat = final_model.predict_cure_probability(X_test)
        test_auc, test_tpr, test_fpr, _ = auc_cure(p_cure_hat)

        all_test_auc.append(test_auc)
        all_test_tpr.append(test_tpr)
        all_test_fpr.append(test_fpr)

        fold_bias = np.mean(p_cure_hat - p_cure_test_true)
        fold_mse = np.mean((p_cure_hat - p_cure_test_true) ** 2)
        all_bias.append(fold_bias)
        all_mse.append(fold_mse)

    pbar.close()
    return {
        "test_ibs": all_test_ibs,
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "test_cindex": all_test_cindex,
        "mean_cindex": float(np.nanmean(all_test_cindex)),
        "std_cindex": float(np.nanstd(all_test_cindex)),
        "test_auc": all_test_auc,
        "mean_auc": float(np.nanmean(all_test_auc)),
        "std_auc": float(np.nanstd(all_test_auc)),
        "test_tpr": all_test_tpr,
        "test_fpr": all_test_fpr,
        "mean_bias": float(np.mean(all_bias)),
        "std_bias": float(np.std(all_bias)),
        "mean_mse": float(np.mean(all_mse)),
        "std_mse": float(np.std(all_mse)),
        "best_params": all_best_params,
    }


def cross_validate_gg_km(
    df,
    estimator,
    preprocessor_factory,
    kernel="rbf",
    n_outer_splits=5,
    n_inner_splits=4,
    n_trials=50,
    t_grid_points=50,
    random_state=42,
    bagging=False,
    bagging_estimator=None,
    kernels=None,
    estimator_name=None,
    optimize_kernel_ranges=False,
    lambda_reg_search_bounds=(1e-5, 1.0),
    gamma_search_bounds=(1e-3, 10.0),
    A_search_bounds=(1e-3, 200.0),
    degree_search_bounds=(2, 10),
    coef0_poly_search_bounds=(0.0, 5.0),
    coef0_sigmoid_search_bounds=(-5.0, 5.0),
    n_jobs=-1,
):

    df = df.reset_index(drop=True)

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )

    all_test_ibs = []
    all_test_cindex = []
    all_test_auc = []
    all_test_tpr = []
    all_test_fpr = []
    all_best_params = []

    model_name = "Bagging" if bagging else kernel

    outer_splits = outer_cv.split(df, df["delta"])

    outer_pbar = tqdm(outer_splits, total=n_outer_splits, desc=f"{model_name} folds")

    for fold, (train_idx, test_idx) in enumerate(outer_pbar):

        df_outer_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        outer_preprocessor = preprocessor_factory()
        X_outer_train, t_outer_train, d_outer_train = outer_preprocessor.fit(
            df_outer_train
        )
        X_test, t_test, d_test = outer_preprocessor.transform(df_test)

        t_lo = np.percentile(t_outer_train, 5)
        t_hi = np.percentile(t_outer_train, 95)
        t_grid = np.linspace(t_lo, t_hi, t_grid_points)

        inner_cv = StratifiedKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=random_state,
        )

        inner_splits = []

        for tr_idx, val_idx in inner_cv.split(df_outer_train, df_outer_train["delta"]):
            df_tr = df_outer_train.iloc[tr_idx].reset_index(drop=True)
            df_val = df_outer_train.iloc[val_idx].reset_index(drop=True)

            inner_preprocessor = preprocessor_factory()
            X_tr, t_tr, d_tr = inner_preprocessor.fit(df_tr)
            X_val, t_val, d_val = inner_preprocessor.transform(df_val)

            t_grid_inner = np.linspace(
                np.percentile(t_tr, 5),
                np.percentile(t_tr, 95),
                t_grid_points,
            )
            inner_splits.append((X_tr, t_tr, d_tr, X_val, t_val, d_val, t_grid_inner))

        def objective(trial):

            params = {}

            # added this part because bagging might not be using the best results from model optimization
            # if estimator_name == "binomial":
            #     params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
            # elif estimator_name == "bernoulli":
            #     params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

            if bagging:
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    50,
                    1000,
                )

                if optimize_kernel_ranges:
                    params.update(
                        _suggest_kernel_ranges(
                            trial,
                            lambda_reg_bounds=lambda_reg_search_bounds,
                            gamma_bounds=gamma_search_bounds,
                            degree_bounds=degree_search_bounds,
                            coef0_poly_bounds=coef0_poly_search_bounds,
                            coef0_sigmoid_bounds=coef0_sigmoid_search_bounds,
                        )
                    )

            else:
                params = {
                    "lambda_reg": trial.suggest_float(
                        "lambda_reg",
                        1e-5,
                        1.0,
                        log=True,
                    ),
                }

                if estimator_name == "binomial":
                    params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
                elif estimator_name == "bernoulli":
                    params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

            if kernel in {
                "rbf",
                "gaussian",
                "laplacian",
                "exponential",
                "cauchy",
                "sigmoid",
            }:
                params["gamma"] = trial.suggest_float(
                    "gamma",
                    gamma_search_bounds[0],
                    gamma_search_bounds[1],
                    log=True,
                )

            if kernel == "wavelet":
                params["A"] = trial.suggest_float(
                    "A",
                    A_search_bounds[0],
                    A_search_bounds[1],
                    log=True,
                )

            if kernel == "polynomial":
                params["gamma"] = trial.suggest_float(
                    "gamma",
                    gamma_search_bounds[0],
                    gamma_search_bounds[1],
                    log=True,
                )

                params["degree"] = trial.suggest_int(
                    "degree",
                    degree_search_bounds[0],
                    degree_search_bounds[1],
                )

                params["coef0"] = trial.suggest_float(
                    "coef0",
                    coef0_poly_search_bounds[0],
                    coef0_poly_search_bounds[1],
                )

            if kernel == "sigmoid":
                params["coef0"] = trial.suggest_float(
                    "coef0",
                    coef0_sigmoid_search_bounds[0],
                    coef0_sigmoid_search_bounds[1],
                )

            val_scores = []

            for (
                X_tr,
                t_tr,
                d_tr,
                X_val,
                t_val,
                d_val,
                t_grid_inner,
            ) in inner_splits:

                try:

                    if bagging:

                        if bagging_estimator is None:
                            model = estimator(
                                kernels=kernels,
                                random_state=random_state,
                                **params,
                            )
                        else:
                            model = estimator(
                                estimator=bagging_estimator,
                                kernels=kernels,
                                random_state=random_state,
                                **params,
                            )

                    else:

                        model = estimator(
                            kernel=kernel,
                            **params,
                        )

                    fit_cure_model(
                        model,
                        X_tr,
                        t_tr,
                        d_tr,
                    )

                    S_pred = model.predict_survival(
                        X_val,
                        t_grid_inner,
                    )

                    ibs = integrated_brier_score(
                        S_pred,
                        t_val,
                        d_val,
                        t_tr,
                        d_tr,
                        t_grid_inner,
                    )

                    val_scores.append(ibs)

                except Exception as e:
                    print(f"Error in optuna trial {e}")
                    return 1.0

            return float(np.mean(val_scores))

        trial_pbar = tqdm(
            total=n_trials,
            desc=f"Fold {fold + 1}",
            leave=False,
        )

        callback = lambda study, trial: trial_pbar.update(1)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

        try:

            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=False,
                callbacks=[callback],
                n_jobs=n_jobs,
            )

        finally:

            trial_pbar.close()

        best_params = study.best_params

        if bagging:

            final_params = {"n_estimators": best_params["n_estimators"]}

            if optimize_kernel_ranges:
                final_params.update(_kernel_ranges_from_best_params(best_params))

            if bagging_estimator is None:
                final_model = estimator(
                    kernels=kernels,
                    random_state=random_state,
                    **final_params,
                )
            else:
                final_model = estimator(
                    estimator=bagging_estimator,
                    kernels=kernels,
                    random_state=random_state,
                    **final_params,
                )

        else:

            final_model = estimator(
                kernel=kernel,
                **best_params,
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
            tau=t_grid[-1],
        )

        p_cure_hat = final_model.predict_cure_probability(X_test)
        test_auc, test_tpr, test_fpr, _ = auc_cure(p_cure_hat)

        all_test_ibs.append(test_ibs)
        all_test_cindex.append(test_cindex)
        all_test_auc.append(test_auc)
        all_test_tpr.append(test_tpr)
        all_test_fpr.append(test_fpr)
        all_best_params.append(best_params)

    return {
        "test_ibs": all_test_ibs,
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "test_cindex": all_test_cindex,
        "mean_cindex": float(np.nanmean(all_test_cindex)),
        "std_cindex": float(np.nanstd(all_test_cindex)),
        "test_auc": all_test_auc,
        "mean_auc": float(np.nanmean(all_test_auc)),
        "std_auc": float(np.nanstd(all_test_auc)),
        "test_tpr": all_test_tpr,
        "test_fpr": all_test_fpr,
        "best_params": all_best_params,
    }


def cross_validate_gg_boost(
    df,
    estimator,
    preprocessor_factory,
    n_outer_splits=5,
    n_inner_splits=4,
    n_trials=50,
    t_grid_points=50,
    random_state=42,
    estimator_name=None,
    learning_rate_search_bounds=(1e-3, 0.5),
    n_estimators_search_bounds=(50, 1000),
    max_depth_search_bounds=(1, 10),
    min_samples_leaf_search_bounds=(5, 100),
    max_features_search_bounds=(0.1, 1.0),
    n_jobs=-1,
):

    df = df.reset_index(drop=True)

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )

    all_test_ibs = []
    all_test_cindex = []
    all_test_auc = []
    all_test_tpr = []
    all_test_fpr = []
    all_best_params = []

    model_name = estimator_name or estimator.__name__

    outer_splits = outer_cv.split(df, df["delta"])

    outer_pbar = tqdm(outer_splits, total=n_outer_splits, desc=f"{model_name} folds")

    for fold, (train_idx, test_idx) in enumerate(outer_pbar):

        df_outer_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        outer_preprocessor = preprocessor_factory()
        X_outer_train, t_outer_train, d_outer_train = outer_preprocessor.fit(
            df_outer_train
        )
        X_test, t_test, d_test = outer_preprocessor.transform(df_test)

        t_lo = np.percentile(t_outer_train, 5)
        t_hi = np.percentile(t_outer_train, 95)
        t_grid = np.linspace(t_lo, t_hi, t_grid_points)

        inner_cv = StratifiedKFold(
            n_splits=n_inner_splits,
            shuffle=True,
            random_state=random_state,
        )

        inner_splits = []

        for tr_idx, val_idx in inner_cv.split(df_outer_train, df_outer_train["delta"]):
            df_tr = df_outer_train.iloc[tr_idx].reset_index(drop=True)
            df_val = df_outer_train.iloc[val_idx].reset_index(drop=True)

            inner_preprocessor = preprocessor_factory()
            X_tr, t_tr, d_tr = inner_preprocessor.fit(df_tr)
            X_val, t_val, d_val = inner_preprocessor.transform(df_val)

            t_grid_inner = np.linspace(
                np.percentile(t_tr, 5),
                np.percentile(t_tr, 95),
                t_grid_points,
            )
            inner_splits.append((X_tr, t_tr, d_tr, X_val, t_val, d_val, t_grid_inner))

        def objective(trial):

            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *learning_rate_search_bounds,
                    log=True,
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *n_estimators_search_bounds,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *max_depth_search_bounds,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    *min_samples_leaf_search_bounds,
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    *max_features_search_bounds,
                ),
            }

            if estimator_name == "binomial":
                params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
            elif estimator_name == "bernoulli":
                params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

            val_scores = []

            for (
                X_tr,
                t_tr,
                d_tr,
                X_val,
                t_val,
                d_val,
                t_grid_inner,
            ) in inner_splits:

                try:

                    model = estimator(
                        random_state=random_state,
                        **params,
                    )

                    fit_cure_model(
                        model,
                        X_tr,
                        t_tr,
                        d_tr,
                    )

                    S_pred = model.predict_survival(
                        X_val,
                        t_grid_inner,
                    )

                    ibs = integrated_brier_score(
                        S_pred,
                        t_val,
                        d_val,
                        t_tr,
                        d_tr,
                        t_grid_inner,
                    )

                    val_scores.append(ibs)

                except Exception as e:
                    print(f"Error in optuna trial {e}")
                    return 1.0

            return float(np.mean(val_scores))

        trial_pbar = tqdm(
            total=n_trials,
            desc=f"Fold {fold + 1}",
            leave=False,
        )

        callback = lambda study, trial: trial_pbar.update(1)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )

        try:

            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=False,
                callbacks=[callback],
                n_jobs=n_jobs,
            )

        finally:

            trial_pbar.close()

        best_params = study.best_params

        final_model = estimator(
            random_state=random_state,
            **best_params,
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
            tau=t_grid[-1],
        )

        p_cure_hat = final_model.predict_cure_probability(X_test)
        test_auc, test_tpr, test_fpr, _ = auc_cure(p_cure_hat)

        all_test_ibs.append(test_ibs)
        all_test_cindex.append(test_cindex)
        all_test_auc.append(test_auc)
        all_test_tpr.append(test_tpr)
        all_test_fpr.append(test_fpr)
        all_best_params.append(best_params)

    return {
        "test_ibs": all_test_ibs,
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "test_cindex": all_test_cindex,
        "mean_cindex": float(np.nanmean(all_test_cindex)),
        "std_cindex": float(np.nanstd(all_test_cindex)),
        "test_auc": all_test_auc,
        "mean_auc": float(np.nanmean(all_test_auc)),
        "std_auc": float(np.nanstd(all_test_auc)),
        "test_tpr": all_test_tpr,
        "test_fpr": all_test_fpr,
        "best_params": all_best_params,
    }


def cross_validate_pcm_boost(
    n,
    method,
    estimator,
    n_outer_splits=5,
    n_inner_splits=4,
    n_trials=20,
    t_grid_points=50,
    random_state=42,
    estimator_name=None,
    learning_rate_search_bounds=(1e-3, 0.5),
    n_estimators_search_bounds=(50, 1000),
    max_depth_search_bounds=(1, 10),
    min_samples_leaf_search_bounds=(5, 100),
    max_features_search_bounds=(0.1, 1.0),
    n_jobs=-1,
):

    df = simulate_pcm(n=n, method=method, seed=random_state)

    X = df[["x1", "x2"]].to_numpy()
    t = df["time"].to_numpy()
    delta = df["event"].to_numpy()

    p_cure_true = 1 - df["pi_x"].to_numpy()

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=random_state
    )
    total_trials = n_outer_splits * n_trials

    model_name = estimator_name or estimator.__name__

    pbar = tqdm(total=total_trials, desc=f"{model_name} nested CV", unit="trial")

    all_test_ibs = []
    all_test_cindex = []
    all_test_auc = []
    all_test_tpr = []
    all_test_fpr = []
    all_bias = []
    all_mse = []
    all_best_params = []

    optuna_callback = lambda study, trial: pbar.update(1)

    for _, (train_idx, test_idx) in enumerate(outer_cv.split(X, delta)):

        X_outer_train, X_test = (X[train_idx], X[test_idx])
        t_outer_train, t_test = (t[train_idx], t[test_idx])
        d_outer_train, d_test = (delta[train_idx], delta[test_idx])
        p_cure_test_true = p_cure_true[test_idx]

        t_grid = np.linspace(
            np.percentile(t_outer_train, 5),
            np.percentile(t_outer_train, 95),
            t_grid_points,
        )

        inner_cv = StratifiedKFold(
            n_splits=n_inner_splits, shuffle=True, random_state=random_state
        )

        def objective(trial):

            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *learning_rate_search_bounds,
                    log=True,
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    *n_estimators_search_bounds,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    *max_depth_search_bounds,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    *min_samples_leaf_search_bounds,
                ),
                "max_features": trial.suggest_float(
                    "max_features",
                    *max_features_search_bounds,
                ),
            }

            if estimator_name == "binomial":
                params["K_bin"] = trial.suggest_int("K_bin", 2, 1000)
            elif estimator_name == "bernoulli":
                params["K_bin"] = trial.suggest_int("K_bin", 1, 1)

            val_scores = []

            for tr_idx, val_idx in inner_cv.split(X_outer_train, d_outer_train):

                X_tr = X_outer_train[tr_idx]
                X_val = X_outer_train[val_idx]

                t_tr = t_outer_train[tr_idx]
                t_val = t_outer_train[val_idx]

                d_tr = d_outer_train[tr_idx]
                d_val = d_outer_train[val_idx]

                t_grid_inner = np.linspace(
                    np.percentile(t_tr, 5), np.percentile(t_tr, 95), t_grid_points
                )

                try:
                    model = estimator(random_state=random_state, **params)

                    fit_cure_model(model, X_tr, t_tr, d_tr)
                    S_pred = model.predict_survival(X_val, t_grid_inner)
                    ibs = integrated_brier_score(
                        S_pred, t_val, d_val, t_tr, d_tr, t_grid_inner
                    )
                    val_scores.append(ibs)

                except Exception as e:
                    print(f"Trial failed: {e}")
                    return 1.0

            return float(np.mean(val_scores))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[optuna_callback],
            n_jobs=n_jobs,
        )

        best_params = study.best_params
        all_best_params.append(best_params)

        final_model = estimator(random_state=random_state, **best_params)

        fit_cure_model(final_model, X_outer_train, t_outer_train, d_outer_train)
        S_test = final_model.predict_survival(X_test, t_grid)

        test_ibs = integrated_brier_score(
            S_test, t_test, d_test, t_outer_train, d_outer_train, t_grid
        )
        all_test_ibs.append(test_ibs)

        test_cindex = uno_c_index_rmst(
            S_pred=S_test,
            t_eval=t_test,
            delta_eval=d_test,
            t_train=t_outer_train,
            delta_train=d_outer_train,
            t_grid=t_grid,
            tau=t_grid[-1],
        )
        all_test_cindex.append(test_cindex)

        p_cure_hat = final_model.predict_cure_probability(X_test)
        test_auc, test_tpr, test_fpr, _ = auc_cure(p_cure_hat)

        all_test_auc.append(test_auc)
        all_test_tpr.append(test_tpr)
        all_test_fpr.append(test_fpr)

        fold_bias = np.mean(p_cure_hat - p_cure_test_true)
        fold_mse = np.mean((p_cure_hat - p_cure_test_true) ** 2)
        all_bias.append(fold_bias)
        all_mse.append(fold_mse)

    pbar.close()
    return {
        "test_ibs": all_test_ibs,
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "test_cindex": all_test_cindex,
        "mean_cindex": float(np.nanmean(all_test_cindex)),
        "std_cindex": float(np.nanstd(all_test_cindex)),
        "test_auc": all_test_auc,
        "mean_auc": float(np.nanmean(all_test_auc)),
        "std_auc": float(np.nanstd(all_test_auc)),
        "test_tpr": all_test_tpr,
        "test_fpr": all_test_fpr,
        "mean_bias": float(np.mean(all_bias)),
        "std_bias": float(np.std(all_bias)),
        "mean_mse": float(np.mean(all_mse)),
        "std_mse": float(np.std(all_mse)),
        "best_params": all_best_params,
    }


def cross_validate_pcm_parametric(
    n,
    method,
    estimator,
    n_outer_splits=5,
    t_grid_points=50,
    random_state=42,
    estimator_params=None,
    estimator_name=None,
):

    df = simulate_pcm(
        n=n,
        method=method,
        seed=random_state,
    )

    X = df[
        [
            "x1",
            "x2",
        ]
    ].to_numpy()

    t = df["time"].to_numpy()
    delta = df["event"].to_numpy()
    p_cure_true = 1.0 - df["pi_x"].to_numpy()

    if estimator_params is None:
        estimator_params = {}

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    model_name = estimator_name if estimator_name is not None else estimator.__name__

    all_test_ibs = []
    all_test_cindex = []

    all_test_auc = []
    all_test_tpr = []
    all_test_fpr = []

    all_bias = []
    all_mse = []

    all_fitted_params = []

    outer_splits = outer_cv.split(
        X,
        delta,
    )

    pbar = tqdm(
        outer_splits,
        total=n_outer_splits,
        desc=f"{model_name} folds",
    )

    for _, (
        train_idx,
        test_idx,
    ) in enumerate(pbar):

        X_outer_train = X[train_idx]
        X_test = X[test_idx]
        t_outer_train = t[train_idx]
        t_test = t[test_idx]
        d_outer_train = delta[train_idx]
        d_test = delta[test_idx]
        p_cure_test_true = p_cure_true[test_idx]

        t_grid = np.linspace(
            np.percentile(
                t_outer_train,
                5,
            ),
            np.percentile(
                t_outer_train,
                95,
            ),
            t_grid_points,
        )

        model = estimator(**estimator_params)
        fit_cure_model(
            model,
            X_outer_train,
            t_outer_train,
            d_outer_train,
        )

        S_test = model.predict_survival(
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
        all_test_ibs.append(test_ibs)

        test_cindex = uno_c_index_rmst(
            S_pred=S_test,
            t_eval=t_test,
            delta_eval=d_test,
            t_train=t_outer_train,
            delta_train=d_outer_train,
            t_grid=t_grid,
            tau=t_grid[-1],
        )

        all_test_cindex.append(test_cindex)
        p_cure_hat = model.predict_cure_probability(X_test)

        (
            test_auc,
            test_tpr,
            test_fpr,
            _,
        ) = auc_cure(p_cure_hat)

        all_test_auc.append(test_auc)
        all_test_tpr.append(test_tpr)
        all_test_fpr.append(test_fpr)

        fold_bias = np.mean(p_cure_hat - p_cure_test_true)
        all_bias.append(fold_bias)

        fold_mse = np.mean((p_cure_hat - p_cure_test_true) ** 2)
        all_mse.append(fold_mse)

        fitted_params = {}

        for name in ("a", "d", "p"):
            fitted_name = name + "_"
            if hasattr(model, fitted_name):
                fitted_params[name] = float(getattr(model, fitted_name))
            elif hasattr(model, name):
                fitted_params[name] = float(getattr(model, name))
        if hasattr(
            model,
            "alpha_",
        ):
            fitted_params["alpha"] = np.asarray(model.alpha_).tolist()
        if hasattr(
            model,
            "beta_",
        ):
            fitted_params["beta"] = np.asarray(model.beta_).tolist()
        if hasattr(
            model,
            "converged_",
        ):
            fitted_params["converged"] = bool(model.converged_)
        if hasattr(
            model,
            "n_iter_",
        ):
            fitted_params["n_iter"] = int(model.n_iter_)
        if hasattr(
            model,
            "final_loglik_",
        ):
            fitted_params["final_loglik"] = float(model.final_loglik_)

        all_fitted_params.append(fitted_params)

    pbar.close()

    return {
        "test_ibs": (all_test_ibs),
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "test_cindex": (all_test_cindex),
        "mean_cindex": float(np.nanmean(all_test_cindex)),
        "std_cindex": float(np.nanstd(all_test_cindex)),
        "test_auc": (all_test_auc),
        "mean_auc": float(np.nanmean(all_test_auc)),
        "std_auc": float(np.nanstd(all_test_auc)),
        "test_tpr": (all_test_tpr),
        "test_fpr": (all_test_fpr),
        "test_bias": (all_bias),
        "mean_bias": float(np.mean(all_bias)),
        "std_bias": float(np.std(all_bias)),
        "test_mse": (all_mse),
        "mean_mse": float(np.mean(all_mse)),
        "std_mse": float(np.std(all_mse)),
        "fitted_params": (all_fitted_params),
    }


def monte_carlo_pcm(
    *,
    n,
    method,
    generating_distribution,
    estimator,
    estimator_name,
    kernel=None,
    parameter_grid=None,
    generating_phi=None,
    generating_K_bin=2,
    replications=250,
    test_size=0.30,
    n_inner_splits=2,
    inner_grid_points=24,
    outer_grid_points=36,
    random_state=20260817,
    return_diagnostics=False,
    surface_grid_points=41,
):
    if parameter_grid is None:
        parameter_grid = [{}]

    generating_distribution = generating_distribution.lower()

    aliases = {
        "nb": "negative_binomial",
        "negative-binomial": "negative_binomial",
        "negative binomial": "negative_binomial",
    }

    generating_distribution = aliases.get(
        generating_distribution,
        generating_distribution,
    )

    distribution_codes = {
        "poisson": 1,
        "negative_binomial": 2,
        "binomial": 3,
        "bernoulli": 4,
    }

    if generating_distribution not in distribution_codes:
        raise ValueError("unsupported generating distribution")

    if method in {1, 2, 3}:
        covariate_columns = [
            "x1",
            "x2",
        ]

    elif method in {4, 5}:
        covariate_columns = [
            "x1",
            "x2",
            "x3",
            "x4",
        ]

    else:
        raise ValueError("method must be 1, 2, 3, 4, or 5")

    phi_seed = 0 if generating_phi is None else int(round(1000 * generating_phi))

    surface_axis = None
    X_surface_raw = None
    true_cure_surface = None

    if method in {1, 2, 3}:

        surface_axis = np.linspace(
            -1.0,
            1.0,
            surface_grid_points,
        )

        (
            x1_surface,
            x2_surface,
        ) = np.meshgrid(
            surface_axis,
            surface_axis,
            indexing="xy",
        )

        X_surface_raw = np.column_stack(
            [
                x1_surface.ravel(),
                x2_surface.ravel(),
            ]
        )

        true_cure_surface = cure_probability(
            x1=X_surface_raw[:, 0],
            x2=X_surface_raw[:, 1],
            method=method,
        )

        true_cure_surface = np.asarray(
            true_cure_surface,
            dtype=float,
        )

    rows = []

    diagnostic_replication = []
    diagnostic_generation_seed = []
    diagnostic_split_seed = []
    diagnostic_inner_seed = []

    diagnostic_train_index = []
    diagnostic_test_index = []

    diagnostic_X_test_raw = []

    diagnostic_t_test = []
    diagnostic_d_test = []

    diagnostic_true_cure = []
    diagnostic_predicted_cure = []

    diagnostic_t_grid = []

    diagnostic_true_survival = []
    diagnostic_predicted_survival = []

    diagnostic_surface_prediction = []

    diagnostic_scaler_mean = []
    diagnostic_scaler_scale = []

    for replication in range(replications):

        replication_start = time.perf_counter()

        seed_sequence = np.random.SeedSequence(
            [
                random_state,
                n,
                method,
                distribution_codes[generating_distribution],
                phi_seed,
                generating_K_bin,
                replication,
            ]
        )

        (
            generation_seed,
            split_seed,
            inner_seed,
        ) = [
            int(value)
            for value in seed_sequence.generate_state(
                3,
                dtype=np.uint32,
            )
        ]

        try:

            data = simulate_pcm(
                n=n,
                method=method,
                distribution=generating_distribution,
                a=1.6,
                d=1.3,
                p=1.1,
                phi=(0.25 if generating_phi is None else generating_phi),
                K_bin=generating_K_bin,
                censor_mean=3.5,
                seed=generation_seed,
            )

            X = data[covariate_columns].to_numpy(
                dtype=float,
            )

            t = data["time"].to_numpy(
                dtype=float,
            )

            delta = data["event"].to_numpy(
                dtype=int,
            )

            true_cure = data["p_cure"].to_numpy(
                dtype=float,
            )

            indices = np.arange(n)

            (
                train_index,
                test_index,
            ) = train_test_split(
                indices,
                test_size=test_size,
                stratify=delta,
                random_state=split_seed,
            )

            X_train_raw = X[train_index]
            X_test_raw = X[test_index]

            t_train = t[train_index]
            t_test = t[test_index]

            d_train = delta[train_index]
            d_test = delta[test_index]

            true_cure_test = true_cure[test_index]

            if len(parameter_grid) == 1:

                best_params = dict(parameter_grid[0])
                best_inner_ibs = np.nan

            else:

                (
                    best_params,
                    best_inner_ibs,
                ) = _select_grid_parameters(
                    X_train_raw,
                    t_train,
                    d_train,
                    estimator,
                    kernel,
                    parameter_grid,
                    n_inner_splits=n_inner_splits,
                    t_grid_points=inner_grid_points,
                    random_state=inner_seed,
                )

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            final_model = _build_estimator(
                estimator,
                kernel,
                best_params,
            )

            fit_cure_model(
                final_model,
                X_train,
                t_train,
                d_train,
            )

            t_grid = _make_evaluation_grid(
                t_train,
                d_train,
                points=outer_grid_points,
            )

            survival = final_model.predict_survival(
                X_test,
                t_grid,
            )

            survival = np.asarray(
                survival,
                dtype=float,
            )

            true_survival = _true_pcm_survival(
                X_test_raw,
                t_grid,
                method=method,
                generating_distribution=(generating_distribution),
                generating_phi=(generating_phi),
                a=1.6,
                d=1.3,
                p=1.1,
            )

            true_survival = np.asarray(
                true_survival,
                dtype=float,
            )

            if true_survival.shape != survival.shape:
                raise RuntimeError(
                    "true and predicted survival "
                    "matrices have different shapes: "
                    f"{true_survival.shape} versus "
                    f"{survival.shape}."
                )

            if np.any(~np.isfinite(true_survival)):
                raise FloatingPointError("Non-finite true survival " "probabilities.")

            if np.any(~np.isfinite(survival)):
                raise FloatingPointError(
                    "Non-finite predicted survival " "probabilities."
                )

            test_ibs = integrated_brier_score(
                survival,
                t_test,
                d_test,
                t_train,
                d_train,
                t_grid,
            )

            test_cindex = uno_c_index_rmst(
                S_pred=survival,
                t_eval=t_test,
                delta_eval=d_test,
                t_train=t_train,
                delta_train=d_train,
                t_grid=t_grid,
                tau=t_grid[-1],
            )

            predicted_cure = final_model.predict_cure_probability(X_test)
            predicted_cure = np.asarray(
                predicted_cure,
                dtype=float,
            )

            if np.any(~np.isfinite(predicted_cure)):
                raise FloatingPointError("Non-finite predicted cure " "probabilities.")

            (
                test_auc,
                test_tpr,
                test_fpr,
                _,
            ) = auc_cure(predicted_cure)

            cure_error = predicted_cure - true_cure_test
            cure_bias = float(np.mean(cure_error))
            cure_mse = float(np.mean(cure_error**2))
            cure_rmse = float(np.sqrt(cure_mse))
            cure_mae = float(np.mean(np.abs(cure_error)))

            survival_error = survival - true_survival
            survival_bias = float(np.mean(survival_error))
            survival_mse = float(np.mean(survival_error**2))
            survival_rmse = float(np.sqrt(survival_mse))
            survival_mae = float(np.mean(np.abs(survival_error)))

            predicted_cure_surface = None

            if X_surface_raw is not None:

                X_surface = scaler.transform(X_surface_raw)

                predicted_cure_surface = final_model.predict_cure_probability(X_surface)
                predicted_cure_surface = np.asarray(
                    predicted_cure_surface,
                    dtype=float,
                )

                if np.any(~np.isfinite(predicted_cure_surface)):
                    raise FloatingPointError(
                        "Non-finite predicted cure " "probabilities on surface grid."
                    )

            raw_converged = getattr(
                final_model,
                "converged_",
                None,
            )

            if raw_converged is None:
                converged = np.nan
            else:
                converged = bool(raw_converged)

            raw_n_iter = getattr(
                final_model,
                "n_iter_",
                np.nan,
            )

            try:
                n_iter = float(raw_n_iter)
            except (TypeError, ValueError):
                n_iter = np.nan

            fitted_a = getattr(
                final_model,
                "a",
                np.nan,
            )

            fitted_d = getattr(
                final_model,
                "d",
                np.nan,
            )

            fitted_p = getattr(
                final_model,
                "p",
                np.nan,
            )

            fitted_phi = getattr(
                final_model,
                "phi_",
                None,
            )

            if fitted_phi is None:
                fitted_phi = getattr(
                    final_model,
                    "phi",
                    np.nan,
                )

            def _finite_scalar_or_nan(
                value,
            ):
                try:
                    value = float(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    return np.nan

                return value if np.isfinite(value) else np.nan

            fitted_a = _finite_scalar_or_nan(fitted_a)
            fitted_d = _finite_scalar_or_nan(fitted_d)
            fitted_p = _finite_scalar_or_nan(fitted_p)
            fitted_phi = _finite_scalar_or_nan(fitted_phi)

            if return_diagnostics:

                diagnostic_replication.append(int(replication))
                diagnostic_generation_seed.append(int(generation_seed))
                diagnostic_split_seed.append(int(split_seed))
                diagnostic_inner_seed.append(int(inner_seed))
                diagnostic_train_index.append(
                    np.asarray(
                        train_index,
                        dtype=np.int64,
                    )
                )

                diagnostic_test_index.append(
                    np.asarray(
                        test_index,
                        dtype=np.int64,
                    )
                )

                diagnostic_X_test_raw.append(
                    np.asarray(
                        X_test_raw,
                        dtype=float,
                    )
                )

                diagnostic_t_test.append(
                    np.asarray(
                        t_test,
                        dtype=float,
                    )
                )

                diagnostic_d_test.append(
                    np.asarray(
                        d_test,
                        dtype=np.int8,
                    )
                )

                diagnostic_true_cure.append(
                    np.asarray(
                        true_cure_test,
                        dtype=float,
                    )
                )

                diagnostic_predicted_cure.append(
                    np.asarray(
                        predicted_cure,
                        dtype=float,
                    )
                )

                diagnostic_t_grid.append(
                    np.asarray(
                        t_grid,
                        dtype=float,
                    )
                )

                diagnostic_true_survival.append(
                    np.asarray(
                        true_survival,
                        dtype=float,
                    )
                )

                diagnostic_predicted_survival.append(
                    np.asarray(
                        survival,
                        dtype=float,
                    )
                )

                diagnostic_scaler_mean.append(
                    np.asarray(
                        scaler.mean_,
                        dtype=float,
                    )
                )

                diagnostic_scaler_scale.append(
                    np.asarray(
                        scaler.scale_,
                        dtype=float,
                    )
                )

                if predicted_cure_surface is not None:
                    diagnostic_surface_prediction.append(predicted_cure_surface)

            rows.append(
                {
                    "replication": replication,
                    "generation_seed": generation_seed,
                    "split_seed": split_seed,
                    "inner_seed": inner_seed,
                    "n": n,
                    "method": method,
                    "generating_distribution": (generating_distribution),
                    "generating_phi": (generating_phi),
                    "model": estimator_name,
                    "kernel": kernel,
                    "status": "ok",
                    "error": None,
                    "converged": converged,
                    "n_iter": n_iter,
                    "fitted_a": fitted_a,
                    "fitted_d": fitted_d,
                    "fitted_p": fitted_p,
                    "fitted_phi": fitted_phi,
                    "train_size": len(train_index),
                    "test_size": len(test_index),
                    "train_event_rate": float(np.mean(d_train)),
                    "test_event_rate": float(np.mean(d_test)),
                    "best_inner_ibs": (best_inner_ibs),
                    "best_params": json.dumps(
                        best_params,
                        sort_keys=True,
                    ),
                    "test_ibs": float(test_ibs),
                    "test_cindex": float(test_cindex),
                    "test_auc": float(test_auc),
                    "cure_bias": cure_bias,
                    "cure_mse": cure_mse,
                    "cure_rmse": cure_rmse,
                    "cure_mae": cure_mae,
                    "survival_bias": (survival_bias),
                    "survival_mse": (survival_mse),
                    "survival_rmse": (survival_rmse),
                    "survival_mae": (survival_mae),
                    "test_tpr": json.dumps(np.asarray(test_tpr).tolist()),
                    "test_fpr": json.dumps(np.asarray(test_fpr).tolist()),
                    "runtime_seconds": (time.perf_counter() - replication_start),
                }
            )

        except Exception as error:

            rows.append(
                {
                    "replication": replication,
                    "generation_seed": generation_seed,
                    "split_seed": split_seed,
                    "inner_seed": inner_seed,
                    "n": n,
                    "method": method,
                    "generating_distribution": (generating_distribution),
                    "generating_phi": (generating_phi),
                    "model": estimator_name,
                    "kernel": kernel,
                    "status": "failed",
                    "error": (f"{type(error).__name__}: " f"{error}"),
                    "converged": False,
                    "n_iter": np.nan,
                    "fitted_a": np.nan,
                    "fitted_d": np.nan,
                    "fitted_p": np.nan,
                    "fitted_phi": np.nan,
                    "train_size": np.nan,
                    "test_size": np.nan,
                    "train_event_rate": np.nan,
                    "test_event_rate": np.nan,
                    "best_inner_ibs": np.nan,
                    "best_params": None,
                    "test_ibs": np.nan,
                    "test_cindex": np.nan,
                    "test_auc": np.nan,
                    "cure_bias": np.nan,
                    "cure_mse": np.nan,
                    "cure_rmse": np.nan,
                    "cure_mae": np.nan,
                    "survival_bias": np.nan,
                    "survival_mse": np.nan,
                    "survival_rmse": np.nan,
                    "survival_mae": np.nan,
                    "test_tpr": None,
                    "test_fpr": None,
                    "runtime_seconds": (time.perf_counter() - replication_start),
                }
            )

    results = pd.DataFrame(rows)

    if not return_diagnostics:
        return results

    successful_count = len(diagnostic_replication)

    diagnostics = {
        "replication": np.asarray(
            diagnostic_replication,
            dtype=np.int32,
        ),
        "generation_seed": np.asarray(
            diagnostic_generation_seed,
            dtype=np.uint32,
        ),
        "split_seed": np.asarray(
            diagnostic_split_seed,
            dtype=np.uint32,
        ),
        "inner_seed": np.asarray(
            diagnostic_inner_seed,
            dtype=np.uint32,
        ),
    }

    if successful_count > 0:

        diagnostics.update(
            {
                "train_index": np.stack(
                    diagnostic_train_index,
                    axis=0,
                ),
                "test_index": np.stack(
                    diagnostic_test_index,
                    axis=0,
                ),
                "X_test_raw": np.stack(
                    diagnostic_X_test_raw,
                    axis=0,
                ),
                "t_test": np.stack(
                    diagnostic_t_test,
                    axis=0,
                ),
                "d_test": np.stack(
                    diagnostic_d_test,
                    axis=0,
                ),
                "true_cure_test": np.stack(
                    diagnostic_true_cure,
                    axis=0,
                ),
                "predicted_cure_test": np.stack(
                    diagnostic_predicted_cure,
                    axis=0,
                ),
                "t_grid": np.stack(
                    diagnostic_t_grid,
                    axis=0,
                ),
                "true_survival_test": np.stack(
                    diagnostic_true_survival,
                    axis=0,
                ),
                "predicted_survival_test": np.stack(
                    diagnostic_predicted_survival,
                    axis=0,
                ),
                "scaler_mean": np.stack(
                    diagnostic_scaler_mean,
                    axis=0,
                ),
                "scaler_scale": np.stack(
                    diagnostic_scaler_scale,
                    axis=0,
                ),
            }
        )

    else:

        test_count = int(np.ceil(n * test_size))
        train_count = n - test_count

        dimension = len(covariate_columns)
        diagnostics.update(
            {
                "train_index": np.empty(
                    (
                        0,
                        train_count,
                    ),
                    dtype=np.int64,
                ),
                "test_index": np.empty(
                    (
                        0,
                        test_count,
                    ),
                    dtype=np.int64,
                ),
                "X_test_raw": np.empty(
                    (
                        0,
                        test_count,
                        dimension,
                    ),
                    dtype=float,
                ),
                "t_test": np.empty(
                    (
                        0,
                        test_count,
                    ),
                    dtype=float,
                ),
                "d_test": np.empty(
                    (
                        0,
                        test_count,
                    ),
                    dtype=np.int8,
                ),
                "true_cure_test": np.empty(
                    (
                        0,
                        test_count,
                    ),
                    dtype=float,
                ),
                "predicted_cure_test": np.empty(
                    (
                        0,
                        test_count,
                    ),
                    dtype=float,
                ),
                "t_grid": np.empty(
                    (
                        0,
                        outer_grid_points,
                    ),
                    dtype=float,
                ),
                "true_survival_test": np.empty(
                    (
                        0,
                        test_count,
                        outer_grid_points,
                    ),
                    dtype=float,
                ),
                "predicted_survival_test": np.empty(
                    (
                        0,
                        test_count,
                        outer_grid_points,
                    ),
                    dtype=float,
                ),
                "scaler_mean": np.empty(
                    (
                        0,
                        dimension,
                    ),
                    dtype=float,
                ),
                "scaler_scale": np.empty(
                    (
                        0,
                        dimension,
                    ),
                    dtype=float,
                ),
            }
        )

    if method in {1, 2, 3}:

        diagnostics["surface_axis"] = np.asarray(
            surface_axis,
            dtype=float,
        )

        diagnostics["surface_grid_raw"] = np.asarray(
            X_surface_raw,
            dtype=float,
        )

        diagnostics["true_cure_surface"] = np.asarray(
            true_cure_surface,
            dtype=float,
        )

        if len(diagnostic_surface_prediction) > 0:

            diagnostics["predicted_cure_surface"] = np.stack(
                diagnostic_surface_prediction,
                axis=0,
            )

        else:

            diagnostics["predicted_cure_surface"] = np.empty(
                (
                    0,
                    surface_grid_points**2,
                ),
                dtype=float,
            )

    return (
        results,
        diagnostics,
    )


def summarize_monte_carlo(
    results,
):
    metrics = [
        "test_ibs",
        "test_cindex",
        "test_auc",
        "cure_bias",
        "cure_mse",
        "cure_rmse",
        "cure_mae",
        "survival_bias",
        "survival_mse",
        "survival_rmse",
        "survival_mae",
    ]

    valid = results.loc[results["status"] == "ok"].copy()

    first = results.iloc[0]

    summary = {
        "model": first["model"],
        "n": int(first["n"]),
        "method": int(first["method"]),
        "generating_distribution": (first["generating_distribution"]),
        "generating_phi": (first["generating_phi"]),
        "kernel": first["kernel"],
        "replications": len(results),
        "successful_replications": len(valid),
        "failed_replications": int(len(results) - len(valid)),
    }

    if "converged" in valid.columns:

        recorded_convergence = valid["converged"].notna()
        summary["convergence_recorded"] = int(recorded_convergence.sum())

        if np.any(recorded_convergence):
            convergence_values = valid.loc[
                recorded_convergence,
                "converged",
            ].astype(bool)

            summary["converged_replications"] = int(convergence_values.sum())
            summary["nonconverged_successful_replications"] = int(
                (~convergence_values).sum()
            )

        else:
            summary["converged_replications"] = 0
            summary["nonconverged_successful_replications"] = 0

    for metric in metrics:

        values = pd.to_numeric(
            valid[metric],
            errors="coerce",
        )

        values = values[np.isfinite(values)]

        count = len(values)

        summary[f"mean_{metric}"] = float(values.mean()) if count else np.nan
        summary[f"sd_{metric}"] = float(values.std(ddof=1)) if count > 1 else np.nan
        summary[f"mcse_{metric}"] = (
            float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else np.nan
        )

    return pd.DataFrame([summary])


def _select_gg_parametric_parameters(
    df_outer_train,
    *,
    estimator,
    preprocessor_factory,
    parameter_grid,
    n_inner_splits,
    t_grid_points,
    random_state,
):

    inner_cv = StratifiedKFold(
        n_splits=n_inner_splits,
        shuffle=True,
        random_state=random_state,
    )

    inner_splits = []

    for (
        train_idx,
        validation_idx,
    ) in inner_cv.split(
        df_outer_train,
        df_outer_train["delta"],
    ):

        df_train = df_outer_train.iloc[train_idx].reset_index(drop=True)
        df_validation = df_outer_train.iloc[validation_idx].reset_index(drop=True)
        preprocessor = preprocessor_factory()

        (
            X_train,
            t_train,
            d_train,
        ) = preprocessor.fit(df_train)

        (
            X_validation,
            t_validation,
            d_validation,
        ) = preprocessor.transform(df_validation)

        t_lo = np.percentile(
            t_train,
            5,
        )

        t_hi = np.percentile(
            t_train,
            95,
        )

        if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
            raise ValueError("Invalid inner-CV evaluation " "time interval.")

        t_grid = np.linspace(
            t_lo,
            t_hi,
            t_grid_points,
        )

        inner_splits.append(
            (
                X_train,
                t_train,
                d_train,
                X_validation,
                t_validation,
                d_validation,
                t_grid,
            )
        )

    candidate_results = []

    for params in parameter_grid:

        fold_ibs = []

        candidate_failed = False

        for (
            X_train,
            t_train,
            d_train,
            X_validation,
            t_validation,
            d_validation,
            t_grid,
        ) in inner_splits:

            try:

                model = estimator(**params)

                fit_cure_model(
                    model,
                    X_train,
                    t_train,
                    d_train,
                )

                S_validation = model.predict_survival(
                    X_validation,
                    t_grid,
                )

                ibs = integrated_brier_score(
                    S_validation,
                    t_validation,
                    d_validation,
                    t_train,
                    d_train,
                    t_grid,
                )

                if not np.isfinite(ibs):
                    candidate_failed = True
                    break

                fold_ibs.append(float(ibs))

            except Exception:

                candidate_failed = True
                break

        if candidate_failed or len(fold_ibs) != n_inner_splits:

            mean_ibs = np.inf

        else:

            mean_ibs = float(np.mean(fold_ibs))

        candidate_results.append(
            {
                "params": dict(params),
                "mean_ibs": mean_ibs,
                "fold_ibs": fold_ibs,
            }
        )

    valid_candidates = [
        candidate
        for candidate in candidate_results
        if np.isfinite(candidate["mean_ibs"])
    ]

    if not valid_candidates:

        raise RuntimeError(
            "All parameter candidates failed " "during inner cross-validation."
        )

    best_candidate = min(
        valid_candidates,
        key=lambda candidate: (candidate["mean_ibs"]),
    )

    return (
        dict(best_candidate["params"]),
        float(best_candidate["mean_ibs"]),
        candidate_results,
    )


def cross_validate_gg_parametric(
    df,
    estimator,
    preprocessor_factory,
    n_outer_splits=5,
    n_inner_splits=4,
    t_grid_points=50,
    random_state=42,
    estimator_params=None,
    parameter_grid=None,
    estimator_name=None,
) -> dict:

    df = df.reset_index(drop=True)

    if estimator_params is None:
        estimator_params = {}

    if parameter_grid is None:

        parameter_grid = [dict(estimator_params)]

        tune_parameters = False

    else:

        parameter_grid = [dict(params) for params in parameter_grid]

        if len(parameter_grid) == 0:
            raise ValueError("parameter_grid cannot be empty.")

        tune_parameters = len(parameter_grid) > 1

    outer_cv = StratifiedKFold(
        n_splits=n_outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_test_ibs = []
    all_test_cindex = []
    all_test_auc = []
    all_test_tpr = []
    all_test_fpr = []

    all_best_params = []
    all_best_inner_ibs = []
    all_inner_results = []

    model_label = estimator_name if estimator_name is not None else estimator.__name__

    outer_splits = outer_cv.split(
        df,
        df["delta"],
    )

    outer_pbar = tqdm(
        outer_splits,
        total=n_outer_splits,
        desc=f"{model_label} folds",
    )

    for (
        fold,
        (
            train_idx,
            test_idx,
        ),
    ) in enumerate(outer_pbar):

        df_outer_train = df.iloc[train_idx].reset_index(drop=True)

        df_test = df.iloc[test_idx].reset_index(drop=True)

        if tune_parameters:

            inner_seed = random_state + fold

            (
                best_params,
                best_inner_ibs,
                inner_results,
            ) = _select_gg_parametric_parameters(
                df_outer_train,
                estimator=estimator,
                preprocessor_factory=(preprocessor_factory),
                parameter_grid=(parameter_grid),
                n_inner_splits=(n_inner_splits),
                t_grid_points=(t_grid_points),
                random_state=(inner_seed),
            )

        else:

            best_params = dict(parameter_grid[0])

            best_inner_ibs = np.nan
            inner_results = []

        outer_preprocessor = preprocessor_factory()

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

        t_lo = np.percentile(
            t_outer_train,
            5,
        )

        t_hi = np.percentile(
            t_outer_train,
            95,
        )

        if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
            raise ValueError(
                f"Invalid outer evaluation " f"time interval in fold {fold}."
            )

        t_grid = np.linspace(
            t_lo,
            t_hi,
            t_grid_points,
        )

        final_model = estimator(**best_params)

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
            tau=t_grid[-1],
        )

        p_cure_hat = final_model.predict_cure_probability(X_test)

        (
            test_auc,
            test_tpr,
            test_fpr,
            _,
        ) = auc_cure(p_cure_hat)

        all_test_ibs.append(test_ibs)
        all_test_cindex.append(test_cindex)
        all_test_auc.append(test_auc)
        all_test_tpr.append(test_tpr)
        all_test_fpr.append(test_fpr)
        all_best_params.append(best_params)
        all_best_inner_ibs.append(best_inner_ibs)
        all_inner_results.append(inner_results)

    return {
        "test_ibs": all_test_ibs,
        "mean_ibs": float(np.mean(all_test_ibs)),
        "std_ibs": float(np.std(all_test_ibs)),
        "test_cindex": (all_test_cindex),
        "mean_cindex": float(np.nanmean(all_test_cindex)),
        "std_cindex": float(np.nanstd(all_test_cindex)),
        "test_auc": all_test_auc,
        "mean_auc": float(np.nanmean(all_test_auc)),
        "std_auc": float(np.nanstd(all_test_auc)),
        "test_tpr": all_test_tpr,
        "test_fpr": all_test_fpr,
        "best_params": (all_best_params),
        "best_inner_ibs": (all_best_inner_ibs),
        "inner_cv_results": (all_inner_results),
    }
