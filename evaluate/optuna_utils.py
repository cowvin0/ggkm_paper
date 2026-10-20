def _suggest_range(trial, base_name, low_bound, high_bound, log=False, is_int=False):
    if is_int:
        lo = trial.suggest_int(f"{base_name}_low", low_bound, high_bound)
        hi = trial.suggest_int(f"{base_name}_high", lo, high_bound)
    else:
        lo = trial.suggest_float(f"{base_name}_low", low_bound, high_bound, log=log)
        hi = trial.suggest_float(f"{base_name}_high", lo, high_bound, log=log)

    return (lo, hi)


def _range_from_best_params(best_params, base_name):
    return (
        best_params[f"{base_name}_low"],
        best_params[f"{base_name}_high"],
    )


def _suggest_kernel_ranges(
    trial,
    lambda_reg_bounds,
    gamma_bounds,
    degree_bounds,
    coef0_poly_bounds,
    coef0_sigmoid_bounds,
):
    return {
        "lambda_reg_range": _suggest_range(
            trial, "lambda_reg_range", *lambda_reg_bounds, log=True
        ),
        "gamma_range": _suggest_range(trial, "gamma_range", *gamma_bounds, log=True),
        "degree_range": _suggest_range(
            trial, "degree_range", *degree_bounds, is_int=True
        ),
        "coef0_poly_range": _suggest_range(
            trial, "coef0_poly_range", *coef0_poly_bounds
        ),
        "coef0_sigmoid_range": _suggest_range(
            trial, "coef0_sigmoid_range", *coef0_sigmoid_bounds
        ),
    }


def _kernel_ranges_from_best_params(best_params):
    return {
        "lambda_reg_range": _range_from_best_params(best_params, "lambda_reg_range"),
        "gamma_range": _range_from_best_params(best_params, "gamma_range"),
        "degree_range": _range_from_best_params(best_params, "degree_range"),
        "coef0_poly_range": _range_from_best_params(best_params, "coef0_poly_range"),
        "coef0_sigmoid_range": _range_from_best_params(
            best_params, "coef0_sigmoid_range"
        ),
    }
