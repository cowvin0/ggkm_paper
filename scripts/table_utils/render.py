"""Render the manuscript's simulation tables from current replication results."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .data import ALL_METHODS, NB_METHODS, POISSON_METHODS, Method, Results, Scenario


SURFACE_NAMES = {
    1: "Linear", 2: "Quadratic", 3: "Periodic", 4: "Interaction", 5: "Radial",
}
GENERATING_NAMES = {"poisson": "P", "nb025": "NB(0.25)", "nb100": "NB(1.00)"}
KERNEL_NAMES = {
    "linear": "linear", "rbf": "RBF", "laplacian": "Laplacian",
    "exponential": "exponential", "cauchy": "Cauchy",
    "polynomial": "polynomial", "wavelet": "wavelet",
}
KERNEL_ABBREVIATIONS = {
    "linear": "lin", "rbf": "RBF", "laplacian": "lap",
    "exponential": "exp", "cauchy": "cau", "polynomial": "poly", "wavelet": "wav",
}
METRIC_NAMES = {
    "test_ibs": "IPCW integrated Brier score", "cure_rmse": "Cure RMSE",
    "test_auc": "Cure AUC",
}
METRIC_ALIASES = {"ibs": "test_ibs", "cure_auc": "test_auc"}

_KERNEL_LEGEND = (
    "lin, RBF, lap, exp, cau, poly, and wav denote the linear, radial-basis, "
    "Laplacian, exponential, Cauchy, polynomial, and wavelet kernels."
)
_MISSING_ESTIMATE_NOTE = (
    "A dash indicates an unavailable estimate or Monte Carlo standard error. "
    "Standard errors require at least two finite replications."
)
_PRIMARY_BEST_CAPTION = """Best observed mean predictive performance in each simulation scenario
among Poisson and negative-binomial GG-KM and spline models.
For IPCW IBS and cure RMSE, smaller values are better, whereas for cure AUC,
larger values are better. Parenthesized values are Monte Carlo standard errors.
The displayed optimum is a descriptive comparison and is not a claim of
superiority when paired uncertainty includes zero.
GG, Spl, P, and NB denote generalized-gamma kernel machine, spline,
Poisson, and negative binomial, respectively.
For GG-KM fits, the selected kernel is shown in parentheses."""
_HIGHDIM_BEST_CAPTION = """Best observed mean predictive performance in each high-dimensional simulation
scenario. For IPCW IBS and cure RMSE, smaller values are better, whereas for
cure AUC, larger values are better. Parenthesized values are Monte Carlo
standard errors. The comparison includes Poisson and negative-binomial GG-KM
models, Poisson and negative-binomial spline models, the mixture-cure model,
and the promotion-time model. Int and Rad denote Interaction and Radial
surfaces, respectively. GG, Spl, P, and NB denote generalized-gamma kernel
machine, spline, Poisson, and negative binomial, respectively. For GG-KM fits,
the selected kernel is shown in parentheses."""
_REGRET_CAPTION = r"""Paired regret and descriptive rank summaries over the 18 primary simulation
scenarios for the Poisson and negative-binomial models.
The comparison is restricted to spline models and GG-KM models using the RBF
kernel. Within each scenario and metric, the reference is the retained method
with the best finite mean. For IPCW integrated Brier score and cure RMSE,
regret is the method value minus the reference value; for cure AUC, regret is
the reference value minus the method value, so that smaller regret is better
for all metrics. Regret uses only replication pairs for which both methods are
finite. The worst regret identifies its scenario and reports a 95\% Monte Carlo
half-width; mean regret weights scenarios equally and propagates their Monte
Carlo standard errors. Mean rank is based on scenario means and is descriptive."""


def _number(value: Any, precision: int) -> str:
    """Format finite values without accidentally printing NaN or infinity."""
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{precision}f}"


def _estimate(mean: float, mcse: float) -> str:
    value = _number(mean, 4)
    return "--" if value == "--" else f"{value} ({_number(mcse, 4)})"


def _paired_estimate(mean: float, halfwidth: float) -> str:
    value = _number(mean, 5)
    return "--" if value == "--" else rf"{value} $\pm$ {_number(halfwidth, 5)}"


def _row(cells: Iterable[str]) -> str:
    return " & ".join(cells) + r" \\"


def _begin_table(
    caption: str,
    labels: Sequence[str],
    columns: str,
    *,
    position: str = "p",
    resize: bool = True,
    size: str | None = "scriptsize",
) -> list[str]:
    lines = [rf"\begin{{table*}}[{position}]", r"\caption{", caption, "}"]
    lines.extend(rf"\label{{{label}}}" for label in labels)
    lines.append(r"\centering")
    if size:
        lines.append(rf"\{size}")
    if not resize:
        lines.append(r"\setlength{\tabcolsep}{4pt}")
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([rf"\begin{{tabular}}{{{columns}}}", r"\hline"])
    return lines


def _end_table(*, resize: bool = True) -> list[str]:
    if resize:
        return [r"\hline", r"\end{tabular}%", "}", r"\end{table*}"]
    return [r"\hline", r"\end{tabular}", r"\end{table*}"]


def _scenario_cells(scenario: Scenario) -> list[str]:
    return [SURFACE_NAMES[scenario.method], GENERATING_NAMES[scenario.generating], str(scenario.n)]


def _compact_method(method: Method, *, best: bool = False, primary: bool = True) -> str:
    if method.model in {"poisson", "bn"}:
        prefix = "GG-P" if method.model == "poisson" else "GG-NB"
        if best:
            kernel = method.kernel
            if primary and kernel == "polynomial":
                kernel = "poly"
            return f"{prefix} ({kernel})"
        return f"{prefix} {KERNEL_ABBREVIATIONS[method.kernel]}"
    return {
        "spline_poisson": "Spl-P", "spline_bn": "Spl-NB",
        "promotion_time": "PT", "mixture_cure": "Mix",
    }[method.model]


def _long_method(method: Method, *, diagnostics: bool = False) -> str:
    if method.model in {"poisson", "bn"}:
        law = "Poisson" if method.model == "poisson" else "Negative binomial"
        kernel = KERNEL_NAMES[method.kernel]
        return f"GG-KM {law} -- {kernel}" if diagnostics else f"GG-KM {law} ({kernel})"
    return {
        "spline_poisson": "Spline Poisson", "spline_bn": "Spline Negative binomial",
        "promotion_time": "Promotion time", "mixture_cure": "Mixture cure",
    }[method.model]


def _metric_caption(metric: str, *, primary: bool, poisson: bool, planned: int) -> str:
    experiment = "primary" if primary else "high-dimensional"
    if metric == "test_ibs":
        introduction = f"Mean IPCW integrated Brier scores across the {experiment} Monte Carlo experiment"
        explanation = "Lower values indicate better prediction over the common test-set evaluation horizon."
    elif metric == "cure_rmse":
        introduction = f"Mean cure-probability root mean squared errors across the {experiment} Monte Carlo experiment"
        explanation = "Lower values indicate more accurate estimation of the known individual cure probabilities."
    else:
        raise ValueError(f"Unsupported metric table: {metric!r}")
    if poisson:
        included = (
            "This table includes all Poisson GG-KM kernel specifications, the Poisson spline, "
            "the promotion-time model, and the mixture-cure model. GG, P, Spl, PT, and Mix "
            "denote GG-KM, Poisson, spline, promotion time, and mixture cure, respectively; "
        )
    else:
        included = (
            "This table includes all negative-binomial GG-KM kernel specifications and the "
            "negative-binomial spline. GG, NB, and Spl denote GG-KM, negative binomial, "
            "and spline, respectively; "
        )
    return (
        f"{introduction}, with Monte Carlo standard errors in parentheses.\n"
        "Means and Monte Carlo standard errors are computed over successful replications\n"
        f"with finite metric values, out of {planned} planned replications per\n"
        f"scenario-method configuration. {explanation}\n{included}{_KERNEL_LEGEND}\n"
        f"{_MISSING_ESTIMATE_NOTE}"
    )


def render_metric_tables(
    results: Results,
    scenarios: Sequence[Scenario],
    metric: str,
    *,
    primary: bool,
    planned_replications: int = 250,
) -> str:
    """Render the Poisson and negative-binomial panels for IBS or cure RMSE."""
    metric = METRIC_ALIASES.get(metric, metric)
    suffix = {"test_ibs": "ibs", "cure_rmse": "cure-rmse"}[metric]
    prefix = "simulation" if primary else "highdim"
    base_label = f"tab:supp-{prefix}-{suffix}"
    tables = []
    for poisson, methods in ((True, POISSON_METHODS), (False, NB_METHODS)):
        labels = [base_label, base_label + "-poisson"] if poisson else [base_label + "-negative-binomial"]
        lines = _begin_table(
            _metric_caption(metric, primary=primary, poisson=poisson, planned=planned_replications),
            labels, "lll" + "r" * len(methods),
        )
        lines.extend([
            _row(["True surface", "Generating law", "$n$"] + [_compact_method(m) for m in methods]),
            r"\hline",
        ])
        previous_surface = None
        for scenario in scenarios:
            if previous_surface is not None and scenario.method != previous_surface:
                lines.append(r"\hline")
            cells = _scenario_cells(scenario)
            for method in methods:
                mean, mcse, _ = results.summary(scenario, method, metric)
                cells.append(_estimate(mean, mcse))
            lines.append(_row(cells))
            previous_surface = scenario.method
        lines.extend(_end_table())
        tables.append("\n".join(lines))
    return "\n\n\n".join(tables) + "\n"


def render_best_table(results: Results, scenarios: Sequence[Scenario], *, primary: bool) -> str:
    """Select each metric's finite optimum using full-precision scenario means."""
    methods = [m for m in ALL_METHODS if not primary or m.model not in {"promotion_time", "mixture_cure"}]
    caption = _PRIMARY_BEST_CAPTION if primary else _HIGHDIM_BEST_CAPTION
    caption += "\n" + _MISSING_ESTIMATE_NOTE
    label = "tab:simulation-best" if primary else "tab:highdim-best"
    lines = _begin_table(
        caption, [label], "lllccc", position="tbp", resize=not primary,
        size="small" if primary else None,
    )
    lines.extend([
        _row(["True surface", "Generating law", "$n$", "Lowest mean IPCW IBS", "Lowest mean cure RMSE", "Highest mean cure AUC"]),
        r"\hline",
    ])
    previous_surface = None
    for scenario in scenarios:
        if previous_surface is not None and scenario.method != previous_surface:
            lines.append(r"\hline")
        cells = _scenario_cells(scenario)
        for metric in ("test_ibs", "cure_rmse", "test_auc"):
            candidates = []
            for method in methods:
                mean, mcse, _ = results.summary(scenario, method, metric)
                if math.isfinite(mean):
                    candidates.append((method, mean, mcse))
            if not candidates:
                cells.append("--")
                continue
            best = (max if metric == "test_auc" else min)(candidates, key=lambda item: item[1])
            method, mean, mcse = best
            cells.append(f"{_compact_method(method, best=True, primary=primary)}: {_estimate(mean, mcse)}")
        lines.append(_row(cells))
        previous_surface = scenario.method
    lines.extend(_end_table(resize=not primary))
    return "\n".join(lines) + "\n"


def _latex_bound(value: float) -> str:
    mantissa, exponent = f"{value:.10e}".split("e")
    coefficient = float(mantissa)
    leading = "" if coefficient == 1 else f"{coefficient:g} " + r"\times "
    return f"${leading}10^{{{int(exponent)}}}$"


def render_diagnostics_table(
    results: Results,
    scenarios: Sequence[Scenario],
    *,
    primary: bool,
    planned_replications: int = 250,
    latency_lower_bound: float = 1e-5,
    phi_lower_bound: float = 1e-8,
) -> str:
    """Render fit status, bound contacts, and runtime in the existing method order."""
    experiment = "primary" if primary else "high-dimensional"
    planned = len(scenarios) * planned_replications
    caption = (
        f"Numerical diagnostics aggregated over all {len(scenarios)} {experiment} simulation scenarios.\n"
        f"Each method configuration has {planned:,} planned fits, corresponding to\n"
        f"{planned_replications} replications in each scenario. Converged fits are successful fits satisfying\n"
        "the explicit convergence criterion; successful nonconverged fits completed\n"
        "without failure but did not satisfy that criterion. Failed denotes fits with\n"
        r"\texttt{status = failed}. Latency-bound contacts count final successful fits" "\n"
        "with at least one of the generalized-gamma parameters $a$, $d$, or $p$ at the\n"
        f"lower optimization bound of {_latex_bound(latency_lower_bound)}. The $\\phi$-bound column counts contacts\n"
        f"with the negative-binomial dispersion lower bound of {_latex_bound(phi_lower_bound)}; a dash indicates\n"
        "that the dispersion parameter is not applicable. Runtime over all recorded fits\n"
        "is the mean with the maximum in parentheses, in seconds."
    )
    prefix = "simulation" if primary else "highdim"
    lines = _begin_table(caption, [f"tab:supp-{prefix}-diagnostics"], "lcccccc")
    lines.extend([
        _row(["Method", "Converged", "Successful nonconverged", "Failed", "Latency bound", r"$\phi$ low", "Runtime (s)"]),
        r"\hline",
    ])
    groups = [
        [m for m in ALL_METHODS if m.model == "poisson"],
        [m for m in ALL_METHODS if m.model == "bn"],
        [Method(name, "parametric") for name in ("spline_poisson", "spline_bn", "promotion_time", "mixture_cure")],
    ]
    for group_index, methods in enumerate(groups):
        if group_index:
            lines.append(r"\hline")
        for method in methods:
            values = results.diagnostics(
                scenarios, method, latency_lower_bound=latency_lower_bound,
                phi_lower_bound=phi_lower_bound,
            )
            phi = str(values["phi_bound"]) if method.model in {"bn", "spline_bn"} else "--"
            runtime = _number(values["runtime_mean"], 2)
            if runtime != "--":
                runtime += f" ({_number(values['runtime_max'], 2)})"
            lines.append(_row([
                _long_method(method, diagnostics=True),
                f"{values['converged']}/{values['total']}",
                str(values["nonconverged"]), str(values["failed"]),
                str(values["latency_bound"]), phi, runtime,
            ]))
    lines.extend(_end_table())
    return "\n".join(lines) + "\n"


def render_regret_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render already-computed paired regret and descriptive ranks."""
    lines = _begin_table(_REGRET_CAPTION, ["tab:simulation-regret"], "llccc", position="tbp", size=None)
    lines.extend([
        _row(["Metric", "Method", r"Worst paired regret ($\pm$95\%; scenario)", r"Mean paired regret ($\pm$95\%)", "Mean rank"]),
        r"\hline",
    ])
    previous_metric = None
    for row in rows:
        metric = METRIC_ALIASES.get(row["metric"], row["metric"])
        if previous_metric is not None and metric != previous_metric:
            lines.append(r"\hline")
        worst = _paired_estimate(row["worst_mean"], row["worst_halfwidth"])
        scenario = row["worst_scenario"]
        if worst != "--" and scenario is not None:
            surface, generating, n = _scenario_cells(scenario)
            worst += f" ({surface}, {generating}, $n={n}$)"
        lines.append(_row([
            METRIC_NAMES[metric], _long_method(row["method"]), worst,
            _paired_estimate(row["mean_regret"], row["mean_halfwidth"]),
            _number(row["mean_rank"], 2),
        ]))
        previous_metric = metric
    lines.extend(_end_table())
    return "\n".join(lines) + "\n"
