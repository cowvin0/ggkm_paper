"""Regenerate melanoma and breast cancer LaTeX tables from saved split CSVs.

Run on the cluster after the clinical experiments finish. Existing tables in
sections/generated are replaced only after all selected inputs are validated
and both tables are rendered. No models are fitted and no raw cohorts are read.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

if __package__:
    from .table_utils.clinical import ClinicalResults, render_performance_table
else:
    from table_utils.clinical import ClinicalResults, render_performance_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MELANOMA_RESULTS_DIR = PROJECT_ROOT / "data" / "melanoma_results" / "grid_search"
BREAST_CANCER_RESULTS_DIR = PROJECT_ROOT / "data" / "breast_cancer_results" / "grid_search_paired_50"
OUTPUT_DIR = PROJECT_ROOT / "sections" / "generated"


def _write_tables(tables: dict[str, str], output_dir: Path) -> tuple[Path, ...]:
    """Stage every rendered table before replacing any existing destination."""
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
    *, dataset="all", melanoma_results_dir=MELANOMA_RESULTS_DIR,
    breast_cancer_results_dir=BREAST_CANCER_RESULTS_DIR, output_dir=OUTPUT_DIR,
    planned_splits=None, include_sigmoid=False,
):
    if dataset not in ("all", "melanoma", "breast-cancer"):
        raise ValueError("--dataset must be all, melanoma, or breast-cancer.")
    if include_sigmoid and dataset == "melanoma":
        raise ValueError("--include-sigmoid applies only to the breast cancer table.")
    tables = {}
    for selected, directory in (
        ("melanoma", melanoma_results_dir), ("breast-cancer", breast_cancer_results_dir),
    ):
        if dataset not in ("all", selected):
            continue
        results = ClinicalResults(
            directory, selected, include_sigmoid=include_sigmoid and selected == "breast-cancer",
        )
        count = results.validate(planned_splits)
        tables[f"{selected}-performance-table.tex"] = render_performance_table(
            results, planned_splits=count,
        )
    paths = _write_tables(tables, Path(output_dir).expanduser().resolve())
    for path in paths:
        print(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("all", "melanoma", "breast-cancer"), default="all",
        help="Table(s) to regenerate (default: both).",
    )
    parser.add_argument(
        "--melanoma-results-dir", type=Path, default=MELANOMA_RESULTS_DIR,
        help="Directory containing splits_melanoma_task*.csv (default: data/melanoma_results/grid_search).",
    )
    parser.add_argument(
        "--breast-cancer-results-dir", type=Path, default=BREAST_CANCER_RESULTS_DIR,
        help="Directory containing splits_breast_cancer_task*.csv (default: data/breast_cancer_results/grid_search_paired_50).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for .tex tables (default: sections/generated).",
    )
    parser.add_argument(
        "--planned-splits", type=int,
        help="Require this number of attempted holdouts per method; otherwise infer it from the CSVs.",
    )
    parser.add_argument(
        "--include-sigmoid", action="store_true",
        help="Add breast cancer sigmoid Poisson/NB rows (omitted in the original active table).",
    )
    args = parser.parse_args(argv)
    try:
        generate_tables(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
