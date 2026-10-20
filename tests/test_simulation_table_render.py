"""Check the manuscript table contract independently of numerical aggregation."""

import re

import pytest

from scripts.table_utils.data import Method, Scenario
from scripts.table_utils.render import (
    render_best_table,
    render_diagnostics_table,
    render_metric_tables,
)


class FakeResults:
    def __init__(self, summaries=None):
        self.summaries = summaries or {}
        self.summary_calls = []
        self.diagnostic_calls = []

    def summary(self, scenario, method, metric):
        self.summary_calls.append((scenario, method, metric))
        return self.summaries.get((method, metric), (0.5, 0.01, 3))

    def diagnostics(self, scenarios, method, **bounds):
        self.diagnostic_calls.append((tuple(scenarios), method, bounds))
        return dict(
            converged=7, nonconverged=1, failed=2, total=10,
            latency_bound=3, phi_bound=4, runtime_mean=1.234, runtime_max=5.678,
        )


def _data_rows(latex):
    return [
        line.removesuffix(r" \\").split(" & ")
        for line in latex.splitlines()
        if line.startswith(("Linear &", "Quadratic &", "Periodic &", "Interaction &", "Radial &"))
    ]


@pytest.mark.parametrize("primary,prefix", [(True, "simulation"), (False, "highdim")])
@pytest.mark.parametrize("metric,suffix", [("test_ibs", "ibs"), ("cure_rmse", "cure-rmse")])
def test_metric_tables_preserve_labels_columns_and_scenario_order(primary, prefix, metric, suffix):
    # An intentional nondefault input order checks that rows stay aligned with inputs.
    scenarios = (Scenario(2 if primary else 5, "nb100", 600), Scenario(1 if primary else 4, "poisson", 300))
    results = FakeResults()

    output = render_metric_tables(results, scenarios, metric, primary=primary, planned_replications=37)

    assert re.findall(r"\\label\{([^}]+)\}", output) == [
        f"tab:supp-{prefix}-{suffix}",
        f"tab:supp-{prefix}-{suffix}-poisson",
        f"tab:supp-{prefix}-{suffix}-negative-binomial",
    ]
    assert re.findall(r"\\begin\{tabular\}\{([^}]+)\}", output) == ["lllrrrrrrrrrr", "lllrrrrrrrr"]
    headers = [line for line in output.splitlines() if line.startswith("True surface &")]
    assert headers[0].removesuffix(r" \\").split(" & ") == [
        "True surface", "Generating law", "$n$", "GG-P lin", "GG-P RBF", "GG-P lap",
        "GG-P exp", "GG-P cau", "GG-P poly", "GG-P wav", "Spl-P", "PT", "Mix",
    ]
    assert headers[1].removesuffix(r" \\").split(" & ") == [
        "True surface", "Generating law", "$n$", "GG-NB lin", "GG-NB RBF", "GG-NB lap",
        "GG-NB exp", "GG-NB cau", "GG-NB poly", "GG-NB wav", "Spl-NB",
    ]
    first_surface, second_surface = ("Quadratic", "Linear") if primary else ("Radial", "Interaction")
    assert [row[:3] for row in _data_rows(output)] == [
        [first_surface, "NB(1.00)", "600"], [second_surface, "P", "300"],
        [first_surface, "NB(1.00)", "600"], [second_surface, "P", "300"],
    ]
    assert output.count("37 planned replications") == 2
    assert len({method for _, method, _ in results.summary_calls}) == 18


@pytest.mark.parametrize("primary,expected_count,label", [
    (True, 16, "tab:simulation-best"), (False, 18, "tab:highdim-best"),
])
def test_best_table_compares_existing_model_set(primary, expected_count, label):
    promotion = Method("promotion_time", "parametric")
    mixture = Method("mixture_cure", "parametric")
    results = FakeResults({
        (promotion, "test_ibs"): (0.01, 0.002, 3),
        (mixture, "cure_rmse"): (0.02, 0.003, 3),
        (promotion, "test_auc"): (0.99, 0.004, 3),
    })

    output = render_best_table(results, (Scenario(1 if primary else 4, "poisson", 300),), primary=primary)

    compared = {method for _, method, _ in results.summary_calls}
    assert len(compared) == expected_count
    assert (promotion in compared) is (not primary)
    assert (mixture in compared) is (not primary)
    assert re.findall(r"\\label\{([^}]+)\}", output) == [label]
    assert r"\begin{tabular}{lllccc}" in output
    if primary:
        assert _data_rows(output)[0][3:] == ["GG-P (linear): 0.5000 (0.0100)"] * 3
        assert r"\small" in output
        assert r"\setlength{\tabcolsep}{4pt}" in output
    else:
        assert _data_rows(output)[0][3:] == [
            "PT: 0.0100 (0.0020)", "Mix: 0.0200 (0.0030)", "PT: 0.9900 (0.0040)",
        ]
        assert r"\resizebox{\textwidth}{!}" in output


@pytest.mark.parametrize("primary,kernel_name", [(True, "poly"), (False, "polynomial")])
def test_best_selection_uses_unrounded_means_and_metric_direction(primary, kernel_name):
    p_linear, nb_linear = Method("poisson", "linear"), Method("bn", "linear")
    p_poly, nb_poly = Method("poisson", "polynomial"), Method("bn", "polynomial")
    results = FakeResults({
        (p_linear, "test_ibs"): (0.200049, 0.001, 3),
        (nb_linear, "test_ibs"): (0.200041, 0.002, 3),
        (p_poly, "cure_rmse"): (0.100049, 0.003, 3),
        (nb_poly, "cure_rmse"): (0.100041, 0.004, 3),
        (p_poly, "test_auc"): (0.700041, 0.005, 3),
        (nb_poly, "test_auc"): (0.700049, 0.006, 3),
    })

    output = render_best_table(results, (Scenario(1 if primary else 4, "poisson", 300),), primary=primary)

    assert _data_rows(output)[0][3:] == [
        "GG-NB (linear): 0.2000 (0.0020)",
        f"GG-NB ({kernel_name}): 0.1000 (0.0040)",
        f"GG-NB ({kernel_name}): 0.7000 (0.0060)",
    ]


def test_missing_means_and_single_replication_mcse_are_explained():
    class SparseResults(FakeResults):
        def summary(self, scenario, method, metric):
            if method == Method("poisson", "rbf") and metric != "test_auc":
                return (0.2, float("nan"), 1)
            return (float("nan"), float("nan"), 0)

    scenarios = (Scenario(1, "poisson", 300),)
    metric_table = render_metric_tables(SparseResults(), scenarios, "test_ibs", primary=True)
    best_table = render_best_table(SparseResults(), scenarios, primary=True)

    assert _data_rows(metric_table)[0][3:5] == ["--", "0.2000 (--)"]
    assert _data_rows(best_table)[0][3:] == [
        "GG-P (rbf): 0.2000 (--)", "GG-P (rbf): 0.2000 (--)", "--",
    ]
    for output in (metric_table, best_table):
        assert "unavailable estimate or Monte Carlo standard error" in output
        assert "at least two finite replications" in output
        assert "nan" not in output.lower()


@pytest.mark.parametrize("primary,prefix", [(True, "simulation"), (False, "highdim")])
def test_diagnostics_preserve_all18_method_order_and_actual_denominators(primary, prefix):
    scenarios = (Scenario(1 if primary else 4, "poisson", 300), Scenario(2 if primary else 5, "nb025", 600))
    results = FakeResults()

    output = render_diagnostics_table(
        results, scenarios, primary=primary, planned_replications=13,
        latency_lower_bound=2e-5, phi_lower_bound=3e-8,
    )

    rows = [line.split(" & ") for line in output.splitlines() if line.startswith(("GG-KM ", "Spline ", "Promotion time &", "Mixture cure &"))]
    assert [row[0] for row in rows] == [
        "GG-KM Poisson -- linear", "GG-KM Poisson -- RBF", "GG-KM Poisson -- Laplacian",
        "GG-KM Poisson -- exponential", "GG-KM Poisson -- Cauchy", "GG-KM Poisson -- polynomial",
        "GG-KM Poisson -- wavelet", "GG-KM Negative binomial -- linear",
        "GG-KM Negative binomial -- RBF", "GG-KM Negative binomial -- Laplacian",
        "GG-KM Negative binomial -- exponential", "GG-KM Negative binomial -- Cauchy",
        "GG-KM Negative binomial -- polynomial", "GG-KM Negative binomial -- wavelet",
        "Spline Poisson", "Spline Negative binomial", "Promotion time", "Mixture cure",
    ]
    assert re.findall(r"\\label\{([^}]+)\}", output) == [f"tab:supp-{prefix}-diagnostics"]
    assert r"\begin{tabular}{lcccccc}" in output
    assert "26 planned fits" in output
    assert "13 replications in each scenario" in output
    assert r"$2 \times 10^{-5}$" in output
    assert r"$3 \times 10^{-8}$" in output
    assert rows[0][1:] == ["7/10", "1", "2", "3", "--", r"1.23 (5.68) \\"]
    assert rows[7][5] == "4"
    assert len(results.diagnostic_calls) == 18
    assert all(call[0] == scenarios for call in results.diagnostic_calls)
    assert all(call[2] == {"latency_lower_bound": 2e-5, "phi_lower_bound": 3e-8} for call in results.diagnostic_calls)
