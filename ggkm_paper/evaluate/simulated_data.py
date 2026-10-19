import numpy as np
import pandas as pd

from scipy.special import expit
from scipy.stats import gengamma


def cure_linear_predictor(
    x1,
    x2,
    method,
    x3=None,
    x4=None,
):

    if method == 1:
        return 0.15 - 1.00 * x1 + 0.75 * x2

    elif method == 2:
        return 0.60 - 1.45 * x1**2 - 1.05 * x2**2 + 0.55 * x1 * x2

    elif method == 3:
        return 0.10 + 1.15 * np.sin(np.pi * x1) + 0.80 * np.cos(np.pi * x2)

    elif method == 4:
        if x3 is None or x4 is None:
            raise ValueError("method 4 requires x3 and x4.")

        C_int = -0.232065875712158

        return C_int + 1.60 * x1 * x2 - 1.30 * x3 * x4 + 0.70 * x1 - 0.50 * x4

    elif method == 5:
        if x3 is None or x4 is None:
            raise ValueError("method 5 requires x3 and x4.")

        C_rad = -1.356859019137574

        return C_rad + 3.00 * np.exp(-0.80 * (x1**2 + x2**2 + x3**2 + x4**2))

    raise ValueError("method must be 1, 2, 3, 4, or 5")


def cure_probability(
    x1,
    x2,
    method,
    x3=None,
    x4=None,
):

    eta = cure_linear_predictor(
        x1=x1,
        x2=x2,
        method=method,
        x3=x3,
        x4=x4,
    )

    return expit(eta)


def gg_inverse(
    q,
    a,
    d,
    p,
):

    q = np.clip(
        q,
        1e-12,
        1.0 - 1e-12,
    )

    return gengamma.ppf(
        q,
        a=d / p,
        c=p,
        scale=a,
    )


def simulate_pcm(
    n=300,
    method=1,
    distribution="poisson",
    a=1.6,
    d=1.3,
    p=1.1,
    phi=0.25,
    K_bin=2,
    censor_mean=3.5,
    seed=None,
):

    if (
        not isinstance(
            n,
            (int, np.integer),
        )
        or n < 1
    ):
        raise ValueError("n must be a positive integer.")

    if method not in {
        1,
        2,
        3,
        4,
        5,
    }:
        raise ValueError("method must be 1, 2, 3, 4, or 5")

    for value, name in (
        (a, "a"),
        (d, "d"),
        (p, "p"),
        (censor_mean, "censor_mean"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0.")

    rng = np.random.default_rng(seed)

    distribution = str(distribution).strip().lower()

    aliases = {
        "nb": "negative_binomial",
        "negative-binomial": "negative_binomial",
        "negative binomial": "negative_binomial",
    }

    distribution = aliases.get(
        distribution,
        distribution,
    )

    valid_distributions = {
        "poisson",
        "negative_binomial",
        "binomial",
        "bernoulli",
    }

    if distribution not in valid_distributions:
        raise ValueError(
            "distribution must be one of "
            "'poisson', 'negative_binomial', "
            "'binomial', or 'bernoulli'."
        )

    if distribution == "negative_binomial":
        if not np.isfinite(phi) or phi <= 0.0:
            raise ValueError("phi must be finite and > 0.")

    if distribution == "binomial":
        if (
            not isinstance(
                K_bin,
                (int, np.integer),
            )
            or K_bin < 1
        ):
            raise ValueError("K_bin must be a positive integer.")

        K_bin = int(K_bin)

    if distribution == "bernoulli":
        K_bin = 1

    if method in {
        1,
        2,
        3,
    }:
        x1 = rng.uniform(
            -1.0,
            1.0,
            size=n,
        )

        x2 = rng.uniform(
            -1.0,
            1.0,
            size=n,
        )

        x3 = None
        x4 = None

    else:
        x1 = rng.uniform(
            -1.0,
            1.0,
            size=n,
        )

        x2 = rng.uniform(
            -1.0,
            1.0,
            size=n,
        )

        x3 = rng.uniform(
            -1.0,
            1.0,
            size=n,
        )

        x4 = rng.uniform(
            -1.0,
            1.0,
            size=n,
        )

    eta_true = cure_linear_predictor(
        x1=x1,
        x2=x2,
        method=method,
        x3=x3,
        x4=x4,
    )

    p_cure = cure_probability(
        x1=x1,
        x2=x2,
        method=method,
        x3=x3,
        x4=x4,
    )

    p_cure = np.clip(
        p_cure,
        1e-12,
        1.0 - 1e-12,
    )

    pi_x = 1.0 - p_cure

    theta = None
    theta_star = None

    if distribution == "poisson":
        theta = -np.log(p_cure)

    elif distribution == "negative_binomial":
        theta = (p_cure ** (-phi) - 1.0) / phi

    elif distribution in {
        "binomial",
        "bernoulli",
    }:
        theta_star = 1.0 - p_cure ** (1.0 / K_bin)

    T = np.full(
        n,
        np.inf,
        dtype=float,
    )

    C = np.empty(
        n,
        dtype=float,
    )

    time = np.empty(
        n,
        dtype=float,
    )

    event = np.zeros(
        n,
        dtype=int,
    )

    cured = np.zeros(
        n,
        dtype=int,
    )

    q_values = np.full(
        n,
        np.nan,
        dtype=float,
    )

    for i in range(n):

        U = rng.uniform(
            0.0,
            1.0,
        )

        if U >= 1.0 - p_cure[i]:
            T[i] = np.inf
            cured[i] = 1

        else:

            if distribution == "poisson":
                q = -np.log1p(-U) / theta[i]

            elif distribution == "negative_binomial":
                q = ((1.0 - U) ** (-phi) - 1.0) / (phi * theta[i])

            elif distribution in {
                "binomial",
                "bernoulli",
            }:
                q = (1.0 - (1.0 - U) ** (1.0 / K_bin)) / theta_star[i]

            else:
                raise AssertionError("Unexpected distribution.")

            q = np.clip(
                q,
                1e-12,
                1.0 - 1e-12,
            )

            q_values[i] = q

            T[i] = gg_inverse(
                q=q,
                a=a,
                d=d,
                p=p,
            )

        C[i] = rng.exponential(scale=censor_mean)

        time[i] = min(
            T[i],
            C[i],
        )

        event[i] = int(T[i] <= C[i])

    data = {
        "time": time,
        "event": event,
        "x1": x1,
        "x2": x2,
        "eta_true": eta_true,
        "p_cure": p_cure,
        "pi_x": pi_x,
        "cured": cured,
        "T_true": T,
        "C": C,
        "q": q_values,
        "distribution": distribution,
        "method": method,
        "seed": seed,
    }

    if method in {
        4,
        5,
    }:
        data["x3"] = x3
        data["x4"] = x4

    if theta is not None:
        data["theta"] = theta
    if theta_star is not None:
        data["theta_star"] = theta_star
    if distribution == "negative_binomial":
        data["phi"] = phi
    if distribution in {
        "binomial",
        "bernoulli",
    }:
        data["K_bin"] = K_bin

    data["gg_a"] = a
    data["gg_d"] = d
    data["gg_p"] = p

    return pd.DataFrame(data)
