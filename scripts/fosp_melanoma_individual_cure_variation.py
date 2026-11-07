"""Plot FOSP melanoma patients' held-out cure means and holdout variation.

Each patient contributes predictions only from successful outer test sets.
Panel A shows patient means and panel B shows sample standard deviations across
those appearances. The reference comparison uses linear, spline, polynomial,
and RBF predictors, each with Poisson and negative-binomial count laws.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fosp_melanoma_plot_utils import (
    DATA_PATH,
    KERNELS,
    OUTPUT_DIR,
    RESULTS_DIR,
    construct_outer_splits,
    load_fosp_melanoma,
)
from scripts.fosp_melanoma_refit_utils import obtain_split_predictions
from scripts.melanoma_individual_cure_variation import (
    collect_individual_predictions as _collect_individual_predictions,
    create_model_specs as _create_model_specs,
    plot_individual_cure_variation as _plot_individual_cure_variation,
    summarize_individual_predictions,
)


OUTPUT_STEM = "fosp_melanoma_individual_cure_variation"


def create_model_specs(results_dir=RESULTS_DIR, ggkm_kernel="rbf"):
    """Resolve the reference figure's eight FOSP melanoma comparisons."""
    return _create_model_specs(
        results_dir, ggkm_kernel=ggkm_kernel, dataset_name="fosp_melanoma",
        kernels=KERNELS,
    )


def collect_individual_predictions(
    specifications,
    *,
    df,
    results_dir=RESULTS_DIR,
    project_root=PROJECT_ROOT,
    outer_splits,
    refit_missing=False,
):
    """Collect FOSP melanoma holdouts using the FOSP melanoma split design."""
    return _collect_individual_predictions(
        specifications,
        df=df,
        results_dir=results_dir,
        project_root=project_root,
        outer_splits=outer_splits,
        refit_missing=refit_missing,
        prediction_loader=obtain_split_predictions,
        cohort_name="FOSP melanoma",
    )


def plot_individual_cure_variation(
    statistics, specifications, *, output_dir=OUTPUT_DIR
):
    """Write the FOSP melanoma two-panel figure as PDF, PNG, and SVG."""
    return _plot_individual_cure_variation(
        statistics, specifications, output_dir=output_dir,
        title="FOSP melanoma", output_stem=OUTPUT_STEM,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--ggkm-kernel", choices=KERNELS, default="rbf",
        help="GG-KM predictor in the individual comparison (reference figure: rbf).",
    )
    parser.add_argument(
        "--refit-missing",
        action="store_true",
        help="Refit saved best parameters when legacy CSV results have no prediction NPZ.",
    )
    args = parser.parse_args(argv)

    df = load_fosp_melanoma(args.data_path)
    outer_splits = construct_outer_splits(df)
    specifications = create_model_specs(args.results_dir, ggkm_kernel=args.ggkm_kernel)
    predictions = collect_individual_predictions(
        specifications,
        df=df,
        results_dir=args.results_dir,
        outer_splits=outer_splits,
        refit_missing=args.refit_missing,
    )
    statistics = summarize_individual_predictions(predictions, n_patients=len(df))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = args.output_dir / f"{OUTPUT_STEM}_individual_statistics.csv"
    statistics.to_csv(statistics_path, index=False)
    paths = plot_individual_cure_variation(
        statistics, specifications, output_dir=args.output_dir
    )
    for specification in specifications:
        group = statistics.loc[statistics["method"].eq(specification["name"])]
        n_means = int(group["mean_predicted_cure"].notna().sum())
        n_sd = int(group["sd_predicted_cure"].notna().sum())
        print(
            f"{specification['name']} ({specification['kernel'] or 'parametric'}): "
            f"{n_means}/{len(df)} patient means; {n_sd}/{len(df)} sample deviations"
        )
    for path in [*paths, statistics_path]:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
