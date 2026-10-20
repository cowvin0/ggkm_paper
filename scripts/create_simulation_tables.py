"""Regenerate the nine simulation LaTeX tables from PCM replication CSVs.

Run on the cluster after the PCM experiments have finished. The default input
is this repository's data/results directory; the default destination replaces
the corresponding tables in sections/generated. No models are fitted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .table_utils import render
    from .table_utils.data import ALL_METHODS, HIGHDIM_SCENARIOS, PRIMARY_SCENARIOS, Results
    from .table_utils.regret import calculate_regret
else:
    from table_utils import render
    from table_utils.data import ALL_METHODS, HIGHDIM_SCENARIOS, PRIMARY_SCENARIOS, Results
    from table_utils.regret import calculate_regret


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = PROJECT_ROOT / "sections" / "generated"


def validate_replications(results, scenarios, planned_replications=None):
    """Infer the run size and require every planned replication to be recorded."""
    if planned_replications is not None and planned_replications < 1:
        raise ValueError("--planned-replications must be a positive integer.")
    scenarios = tuple(scenarios)
    results.validate(scenarios, ALL_METHODS)
    counts = {
        len(results.frame(scenario, method))
        for scenario in scenarios for method in ALL_METHODS
    }
    if len(counts) != 1:
        raise ValueError(
            "Replication files contain inconsistent numbers of attempted fits "
            f"({', '.join(str(count) for count in sorted(counts))}). "
            "Use completed PCM runs with the same planned replication count."
        )
    recorded = counts.pop()
    if planned_replications is not None and recorded != planned_replications:
        raise ValueError(
            f"Expected {planned_replications} planned replications per configuration, "
            f"but the CSV files contain {recorded}."
        )
    expected_ids = np.arange(recorded, dtype=np.int64)
    for scenario in scenarios:
        for method in ALL_METHODS:
            frame = results.frame(scenario, method)
            if not np.array_equal(np.sort(frame["replication"].to_numpy()), expected_ids):
                raise ValueError(
                    f"{results.path(scenario, method)}: expected replication IDs "
                    f"0 through {recorded - 1}; incomplete or shifted IDs cannot "
                    "be counted as planned fits."
                )
    return recorded


def render_tables(results, *, planned_replications):
    """Compute every output in memory before replacing any existing table."""
    tables = {}
    for prefix, scenarios, primary in (
        ("simulation", PRIMARY_SCENARIOS, True),
        ("highdim", HIGHDIM_SCENARIOS, False),
    ):
        tables[f"{prefix}-best-table.tex"] = render.render_best_table(
            results, scenarios, primary=primary,
        )
        for suffix, metric in (("ibs", "test_ibs"), ("cure-rmse", "cure_rmse")):
            tables[f"{prefix}-{suffix}-table.tex"] = render.render_metric_tables(
                results, scenarios, metric, primary=primary,
                planned_replications=planned_replications,
            )
        tables[f"{prefix}-diagnostics-table.tex"] = render.render_diagnostics_table(
            results, scenarios, primary=primary,
            planned_replications=planned_replications,
        )
    tables["simulation-regret-table.tex"] = render.render_regret_table(
        calculate_regret(results),
    )
    return tables


def _write_tables(tables, output_dir):
    """Stage every complete LaTeX document and then replace its destination."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for name, contents in tables.items():
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output_dir,
                prefix=f".{name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                pending.append((temporary, output_dir / name))
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, destination in pending:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
    return tuple(output_dir / name for name in tables)


def generate_tables(
    results_dir=RESULTS_DIR, output_dir=OUTPUT_DIR, *, planned_replications=None,
):
    results = Results(results_dir)
    count = validate_replications(
        results, PRIMARY_SCENARIOS + HIGHDIM_SCENARIOS, planned_replications,
    )
    tables = render_tables(results, planned_replications=count)
    paths = _write_tables(tables, Path(output_dir).expanduser().resolve())
    print(
        f"Read {len(ALL_METHODS)} configurations in each of "
        f"{len(PRIMARY_SCENARIOS) + len(HIGHDIM_SCENARIOS)} scenarios "
        f"({count} attempted replications per configuration)."
    )
    for path in paths:
        print(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory containing PCM replications_task*.csv files (default: data/results).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for the nine .tex files (default: sections/generated).",
    )
    parser.add_argument(
        "--planned-replications", type=int, default=None,
        help="Require this number of replications; otherwise infer it from the CSV files.",
    )
    args = parser.parse_args(argv)
    try:
        generate_tables(
            args.results_dir, args.output_dir,
            planned_replications=args.planned_replications,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
