"""Paired regret and descriptive ranks for the primary simulation tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import METRICS, PRIMARY_SCENARIOS, Method, Results


REGRET_METHODS = (
    Method("spline_bn", "parametric"),
    Method("spline_poisson", "parametric"),
    Method("poisson", "rbf"),
    Method("bn", "rbf"),
)


def _check_pair_seeds(results, scenario, method, reference, replication_ids):
    """When recorded, confirm matched rows use the same data and holdout."""
    frame = results.frame(scenario, method).set_index("replication")
    reference_frame = results.frame(scenario, reference).set_index("replication")
    for column in ("generation_seed", "split_seed"):
        present = column in frame.columns
        reference_present = column in reference_frame.columns
        if present != reference_present:
            raise ValueError(
                f"Cannot pair {scenario}: {column} is recorded for only one "
                f"of {method} and {reference}."
            )
        if not present:
            continue
        seeds = pd.to_numeric(frame.loc[replication_ids, column], errors="raise")
        reference_seeds = pd.to_numeric(
            reference_frame.loc[replication_ids, column], errors="raise",
        )
        if (
            not np.isfinite(seeds.to_numpy()).all()
            or not np.isfinite(reference_seeds.to_numpy()).all()
            or not np.array_equal(seeds.to_numpy(), reference_seeds.to_numpy())
        ):
            raise ValueError(
                f"Cannot pair {scenario}: {column} does not match for "
                f"{method} and {reference}."
            )


def calculate_regret(results: Results, scenarios=PRIMARY_SCENARIOS):
    """Compute the statistics specified by simulation-regret-table.tex.

    Reference selection and ranks use each method's finite successful mean.
    Regret uses shared successful replication IDs and preserves its sign when
    the methods' finite cohorts differ. Scenarios receive equal weight.
    """
    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("Regret requires at least one scenario.")

    rows = []
    for metric in METRICS:
        regrets = {method: [] for method in REGRET_METHODS}
        ranks = {method: [] for method in REGRET_METHODS}
        higher_is_better = metric == "test_auc"

        for scenario in scenarios:
            values = {
                method: results.metric(scenario, method, metric)
                for method in REGRET_METHODS
            }
            means = pd.Series(
                {method: float(series.mean()) for method, series in values.items()},
                dtype=float,
            )
            if not np.isfinite(means.to_numpy()).all():
                raise ValueError(
                    f"Cannot calculate regret for {scenario}, {metric}: "
                    "each retained method needs a finite successful value."
                )

            reference = means.idxmax() if higher_is_better else means.idxmin()
            scenario_ranks = means.rank(method="average", ascending=not higher_is_better)

            for method in REGRET_METHODS:
                paired = pd.concat(
                    [values[method].rename("method"), values[reference].rename("reference")],
                    axis=1, join="inner",
                )
                _check_pair_seeds(results, scenario, method, reference, paired.index)
                if method == reference:
                    mean, mcse = 0.0, 0.0
                else:
                    if len(paired) < 2:
                        raise ValueError(
                            f"Cannot calculate paired regret for {scenario}, {metric}, "
                            f"{method}: fewer than two finite replication pairs "
                            f"with reference {reference}."
                        )
                    difference = paired["method"] - paired["reference"]
                    if higher_is_better:
                        difference = -difference
                    mean = float(difference.mean())
                    mcse = float(difference.std(ddof=1) / np.sqrt(len(difference)))
                regrets[method].append((scenario, mean, mcse))
                ranks[method].append(float(scenario_ranks.loc[method]))

        for method in REGRET_METHODS:
            scenario_regrets = regrets[method]
            worst = max(scenario_regrets, key=lambda item: item[1])
            mean_regret = float(np.mean([item[1] for item in scenario_regrets]))
            mean_mcse = float(
                np.sqrt(sum(item[2] ** 2 for item in scenario_regrets)) / len(scenarios)
            )
            rows.append({
                "metric": metric,
                "method": method,
                "worst_mean": worst[1],
                "worst_halfwidth": 1.96 * worst[2],
                "worst_scenario": worst[0],
                "mean_regret": mean_regret,
                "mean_halfwidth": 1.96 * mean_mcse,
                "mean_rank": float(np.mean(ranks[method])),
            })
    return rows
