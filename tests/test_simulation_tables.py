"""Verify paired regret and complete regeneration of the manuscript tables."""

from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import create_simulation_tables as generator
from scripts.table_utils.data import (
    ALL_METHODS, HIGHDIM_SCENARIOS, PRIMARY_SCENARIOS, Results, Scenario,
)
from scripts.table_utils.regret import REGRET_METHODS, calculate_regret


def _save(directory, scenario, method, frame, task):
    path = directory / (
        f"replications_task{task}_{method.model}_n{scenario.n}_"
        f"method{scenario.method}_gen{scenario.generating}_kernel{method.kernel}.csv"
    )
    frame.to_csv(path, index=False)
    return path


def _frame(losses):
    losses = np.asarray(losses, dtype=float)
    count = len(losses)
    return pd.DataFrame({
        "replication": np.arange(count),
        "generation_seed": np.arange(count) + 100,
        "split_seed": np.arange(count) + 200,
        "status": ["ok"] * count,
        "test_ibs": losses,
        "cure_rmse": losses,
        "test_auc": 1 - losses,
    })


def test_regret_pairs_finite_cohorts_and_propagates_scenario_uncertainty(tmp_path):
    scenarios = tuple(Scenario(i, "poisson", 300) for i in (1, 2, 3))
    losses = (
        ([1, 2, 3], [2, 3, 4], [1, 4, 4], [2, 2, 5]),
        ([4, 4, 4], [2, 2, 2], [1, 4, 4], [3, 3, 3]),
        ([2, np.nan, 8, 2], [1, 2, np.nan, 1], [0, np.nan, 6, 3], [4, 4, 4, 4]),
    )
    for scenario_index, scenario in enumerate(scenarios):
        for method_index, method in enumerate(REGRET_METHODS):
            # Binary fractions keep the deliberately tied scenario means exact.
            frame = _frame(np.asarray(losses[scenario_index][method_index]) / 16)
            _save(tmp_path, scenario, method, frame.iloc[::-1], scenario_index * 4 + method_index)

    rows = calculate_regret(Results(tmp_path), scenarios)

    expected_means = np.array([0.1, 1 / 30, 1 / 12, 7 / 45]) * 0.625
    expected_halfwidths = np.array([0, 0, 0.123674003975206, 0.043555555555556]) * 0.625
    expected_ranks = [17 / 6, 5 / 3, 2.5, 3]
    expected_worst = np.array([0.2, 0.1, 0.1, 4 / 15]) * 0.625
    for metric in ("test_ibs", "cure_rmse", "test_auc"):
        selected = [row for row in rows if row["metric"] == metric]
        assert [row["method"] for row in selected] == list(REGRET_METHODS)
        np.testing.assert_allclose([row["mean_regret"] for row in selected], expected_means)
        np.testing.assert_allclose([row["mean_halfwidth"] for row in selected], expected_halfwidths, atol=1e-14)
        np.testing.assert_allclose([row["mean_rank"] for row in selected], expected_ranks)
        np.testing.assert_allclose([row["worst_mean"] for row in selected], expected_worst)
        assert selected[0]["worst_scenario"] == scenarios[1]
        assert selected[3]["worst_scenario"] == scenarios[2]


@pytest.mark.parametrize("seed_column", ["generation_seed", "split_seed"])
def test_regret_rejects_replications_generated_with_different_seeds(tmp_path, seed_column):
    scenario = PRIMARY_SCENARIOS[0]
    for index, method in enumerate(REGRET_METHODS):
        frame = _frame([0.1 + index / 100, 0.2 + index / 100])
        if index == 1:
            frame.loc[0, seed_column] += 1
        _save(tmp_path, scenario, method, frame, index)

    with pytest.raises(ValueError, match=f"{seed_column} does not match"):
        calculate_regret(Results(tmp_path), [scenario])


def test_regret_preserves_negative_paired_differences(tmp_path):
    scenario = PRIMARY_SCENARIOS[0]
    losses = (
        [0.1, 0.2, 0.3, np.nan], [0.05, np.nan, 0.25, 0.9],
        [0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4],
    )
    for index, (method, values) in enumerate(zip(REGRET_METHODS, losses)):
        _save(tmp_path, scenario, method, _frame(values), index)

    rows = calculate_regret(Results(tmp_path), [scenario])
    spline_poisson = next(
        row for row in rows if row["metric"] == "test_ibs" and row["method"] == REGRET_METHODS[1]
    )
    # Its marginal mean .40 exceeds the reference mean .20, yet the paired
    # mean is (.05 - .10 + .25 - .30)/2 = -.05 and must retain its sign.
    assert spline_poisson["mean_regret"] == pytest.approx(-0.05)


@pytest.fixture(scope="module")
def complete_pcm_results(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pcm-table-results")
    paths = []
    for scenario_index, scenario in enumerate(PRIMARY_SCENARIOS + HIGHDIM_SCENARIOS):
        for method_index, method in enumerate(ALL_METHODS):
            base = 0.1 + method_index * 0.005 + scenario_index * 0.001
            frame = _frame([base, base + 0.02, base + 0.04, 99])
            frame.loc[3, "status"] = "failed"
            frame["converged"] = [True, False, True, False]
            frame["runtime_seconds"] = [2.0, 4.0, 6.0, 100.0]
            frame["fitted_a"] = [1e-5, 1, 1, np.nan]
            frame["fitted_d"] = [1, 1, 1, np.nan]
            frame["fitted_p"] = [1, 1, 1, np.nan]
            frame["fitted_phi"] = [1e-8, 1.5e-8, 0.2, np.nan]
            paths.append(_save(directory, scenario, method, frame, len(paths)))
    return directory, paths


def test_cli_generates_all_nine_tables_with_existing_labels(complete_pcm_results, tmp_path):
    directory, _ = complete_pcm_results
    script = Path(generator.__file__).resolve()
    output = tmp_path / "tables"

    completed = subprocess.run(
        [sys.executable, str(script), "--results-dir", str(directory),
         "--output-dir", str(output), "--planned-replications", "4"],
        cwd=tmp_path, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    names = {
        "simulation-best-table.tex", "highdim-best-table.tex", "simulation-regret-table.tex",
        "simulation-ibs-table.tex", "simulation-cure-rmse-table.tex", "simulation-diagnostics-table.tex",
        "highdim-ibs-table.tex", "highdim-cure-rmse-table.tex", "highdim-diagnostics-table.tex",
    }
    assert {path.name for path in output.iterdir()} == names
    for name in names:
        text = (output / name).read_text()
        original = (generator.PROJECT_ROOT / "sections" / "generated" / name).read_text()
        active_original = "\n".join(line for line in original.splitlines() if not line.lstrip().startswith("%"))
        assert re.findall(r"\\label\{([^}]+)\}", text) == re.findall(r"\\label\{([^}]+)\}", active_original)
        assert text.count(r"\begin{table*}") == text.count(r"\end{table*}")
        assert not re.search(r"\b(?:nan|inf)\b", text, flags=re.IGNORECASE)
    diagnostics = (output / "simulation-diagnostics-table.tex").read_text()
    assert "36/72 & 18 & 18 & 18" in diagnostics
    assert "28.00 (100.00)" in diagnostics
    assert "4 planned replications" in (output / "highdim-ibs-table.tex").read_text()


def test_missing_input_does_not_replace_existing_tables(tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()
    _save(directory, PRIMARY_SCENARIOS[0], ALL_METHODS[0], _frame([0.1, 0.2]), 0)
    output = tmp_path / "tables"
    output.mkdir()
    existing = output / "simulation-best-table.tex"
    existing.write_text("existing results\n")

    with pytest.raises(FileNotFoundError, match="Missing PCM replication file"):
        generator.generate_tables(directory, output)

    assert existing.read_text() == "existing results\n"
    assert list(output.iterdir()) == [existing]


def test_inferred_replication_count_can_be_checked(complete_pcm_results):
    directory, _ = complete_pcm_results
    results = Results(directory)
    scenarios = PRIMARY_SCENARIOS + HIGHDIM_SCENARIOS
    assert generator.validate_replications(results, scenarios) == 4
    with pytest.raises(ValueError, match="Expected 250 planned replications"):
        generator.validate_replications(results, scenarios, 250)


@pytest.mark.parametrize("damage", ["missing_row", "shifted_ids"])
def test_incomplete_replication_sets_are_rejected(complete_pcm_results, tmp_path, damage):
    _, paths = complete_pcm_results
    for path in paths[:len(ALL_METHODS)]:
        shutil.copy2(path, tmp_path / path.name)
    damaged_path = tmp_path / paths[0].name
    frame = pd.read_csv(damaged_path)
    if damage == "missing_row":
        frame = frame.iloc[:-1]
        message = "inconsistent numbers"
    else:
        frame["replication"] += 1
        message = "expected replication IDs"
    frame.to_csv(damaged_path, index=False)

    with pytest.raises(ValueError, match=message):
        generator.validate_replications(Results(tmp_path), [PRIMARY_SCENARIOS[0]])
