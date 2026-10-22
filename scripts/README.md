# Experiment scripts

Use [run_workflow.py](run_workflow.py) to run all experiments, figures, and tables
through one entry point. Choose `--mode cluster` to submit the PBS jobs with
dependencies, or `--mode local` to execute the same routines sequentially. Add
`--dry-run` to preview the workflow or `--stage postprocess` to use existing
experiment results. See [the workflow instructions](../pbs_jobs/README.md) for
examples, dataset selection, and cluster environment settings.

The entry points retain their dataset-specific settings, data loading, experiment
ordering, and execution logic:

- `run_breast_cancer_experiment.py`: breast-cancer repeated holdouts.
- `run_melanoma_experiment.py`: melanoma repeated holdouts.
- `run_pcm_experiment.py`: simulated PCM experiments.
- `preprocessing.py`: dataset preprocessors and feature encoders.

Shared boilerplate lives in `experiment_utils/`:

| Module | Responsibility |
| --- | --- |
| `models.py` | Model registries, model construction, and parameter grids |
| `evaluation.py` | Censoring horizons and evaluation time grids |
| `tuning.py` | Inner folds, scoring, and exhaustive grid search |
| `artifacts.py` | Atomic checkpoints, diagnostic arrays, and saved paths |
| `results.py` | Result summaries and CSV/NPZ output |
| `reporting.py` | Console progress and completion reports |
| `runtime.py` | Deterministic inner seeds and PBS worker counts |

Adjust experiment constants in the entry-point files. Their helper functions
forward current settings to the shared modules, so existing imports and small-run
overrides still work. Each runner keeps its own model registry and parameter
grids, including the differences between the two clinical experiments.

Run the same entry-point modules as before; `PBS_ARRAY_INDEX`, input paths,
output directories, file formats, and checkpoint resume rules are unchanged.

## Primary simulation precision pilot

The independent pilot uses all 18 primary data-generating scenarios and the 18
model configurations reported in the manuscript: **324 cells and five planned
replications per cell**. It reuses the final PCM generator, model definitions,
tuning grids, train/test split, and evaluation settings. Its master seed is
`20260818`, separate from the final experiment's `20260817`. Pilot observations
are saved separately and never enter the final simulation tables or estimates.

Run the pilot and its precision summary on the cluster:

```bash
python scripts/run_workflow.py --mode cluster --datasets simulation --stage pilot
```

Use `--mode local` to run the same pilot sequentially on a local machine. The
complete workflow (`--stage all`, the default) and `--stage experiments` now
include the pilot and summary before the final simulation. Figures-only,
tables-only, and postprocessing runs use the existing final results directly.

Pilot files are saved under `data/pilot_results/primary/`:

- `pilot_design.json` records all cells, tuning grids, seeds, software versions,
  and source checksums.
- `replications_pilot_task*.csv` retains every planned attempt, including failed
  fits, its metrics, and generation/split/tuning seeds. Each CSV has a metadata
  file containing its study identity and checksum.
- `pilot_cell_summary.csv` gives successful and failed counts, finite metric
  counts, means, sample standard deviations, and precision projections by cell.
- `pilot_precision_report.json` and `.txt` give the largest within-cell standard
  deviation for each metric, the replication requirements, the worst cells,
  and whether the configured final replication count meets the targets.

For each cell, sample standard deviations use successful, finite observations
with `ddof=1`. The report takes the largest cell standard deviation and computes
`ceil((maximum_sd / target_mcse)**2)`, using targets **0.003 for IBS** and
**0.005 for cure-probability MSE**. It uses `cure_mse` directly. Cure AUC has a
projected MCSE but does not determine the required allocation. Projections use
the current final replication count, initially 250; the pilot never changes it
automatically or rounds its requirement to a convenient study size.

Every cell must contain all five attempted replication IDs and at least four
successful, finite values for each reported metric. Missing or insufficient
cells produce an incomplete report. Mismatched cell identities, shared seeds,
checksums, or study metadata are rejected. If the data are incomplete or 250
replications do not meet the IBS/MSE targets, the summary exits unsuccessfully
and prevents the dependent final simulation from starting. Inspect the report
and choose the final allocation explicitly before proceeding.

Completed pilot cells are reused when their study identity and checksums match.
Use a separate output directory for a different seed, fitting design, or
software/source version. To run one cell manually, or recompute only the report:

```bash
python scripts/run_simulation_pilot.py --task-id 0
python scripts/summarize_simulation_pilot.py
```

The runner accepts `--all`, `--seed`, `--replications`, `--output-dir`, and
`--overwrite`; the summary accepts `--results-dir`, `--output-dir`,
`--final-replications`, `--minimum-successful`, and the two MCSE targets.
Source/settings checks allow reassessing the same pilot at another final
replication count without pooling or refitting its observations. The source and
software used to read the pilot must match its recorded fitting design.

These scripts recompute the pilot under the current implementation and the
full design selected for this workflow. They do not impose the historical
standard deviations quoted in the manuscript; the measured values and resulting
allocation can differ.

## PCM review figures

The plotting scripts read saved PCM experiment outputs; they do not rerun model
fitting. Their default input is the repository's `data/results/` directory, and
all figures and CSV summaries are saved in `figures/review/`. Paths are anchored
to the repository, so they also work when launched from another directory.

Run these commands from the repository root on the cluster after the required
PCM results have been saved:

```bash
python scripts/create_paired_plot_1_2_3_methods.py
python scripts/create_paired_plot_interaction_quadratic_surfaces.py
python scripts/recover_cure_surfaces_figure.py
```

| Script | Input and output |
| --- | --- |
| `create_paired_plot_1_2_3_methods.py` | Replication CSVs for methods 1–3 (linear, quadratic, periodic); saves `simulation_rbf_paired_differences_with_auc.{pdf,png,svg}`, a paired summary CSV, and a file audit CSV |
| `create_paired_plot_interaction_quadratic_surfaces.py` | Replication CSVs for methods 4–5 (interaction and radial, as in the supplied code); saves `highdim_rbf_vs_linear_spline_polynomial_paired_differences_with_auc.{pdf,png,svg,csv}` |
| `recover_cure_surfaces_figure.py` | Diagnostic NPZ archives in `data/results/diagnostics/` for Poisson-generated data, `n=600`, methods 1–3, and the linear/spline/polynomial/RBF Poisson models; saves `simulation_cure_surface_recovery.{pdf,png}` |

Both paired plots compare RBF GG-KM against linear GG-KM, additive splines, and
polynomial GG-KM for the Poisson and negative-binomial count laws. They use the
three generating scenarios and sample sizes 300 and 600. Differences are RBF
minus comparator; the intervals are the paired mean ± 1.96 Monte Carlo standard
errors.

If results are stored elsewhere on the cluster, pass their directory explicitly:

```bash
python scripts/create_paired_plot_1_2_3_methods.py --results-dir /path/to/results
```

All three scripts accept `--results-dir` and `--output-dir`; use `--help` to see
their defaults. Keep each experiment's original `replications_task*.csv` and
`diagnostics_task*.npz` filenames. Each required experiment must resolve to one
input file.

## PCM simulation tables

Regenerate all nine simulation and high-dimensional LaTeX files on the cluster:

```bash
python scripts/create_simulation_tables.py
```

The command reads `replications_task*.csv` from the repository's `data/results/`
and replaces these files in `sections/generated/`, keeping the table labels and
layouts used by the manuscript:

- `simulation-best-table.tex` and `highdim-best-table.tex`
- `simulation-regret-table.tex`
- `simulation-ibs-table.tex` and `highdim-ibs-table.tex`
- `simulation-cure-rmse-table.tex` and `highdim-cure-rmse-table.tex`
- `simulation-diagnostics-table.tex` and `highdim-diagnostics-table.tex`

For a different input directory or a separate output directory:

```bash
python scripts/create_simulation_tables.py \
  --results-dir /path/to/results \
  --output-dir /path/to/generated-tables \
  --planned-replications 250
```

The replication count is inferred from the CSVs unless
`--planned-replications` is supplied to enforce the expected study size. Every
configuration must contain the same number of attempted replications, with IDs
starting at zero and no gaps. Failed fits remain recorded attempts. The command
validates all required inputs and computes all nine tables before replacing any
existing table. It requires the 18 configurations and 30 scenarios represented
by these tables (540 CSVs); the Bernoulli and binomial configurations are ignored.
NPZ archives and the experiment's summary CSVs are not required.

The calculations live in `table_utils/`:

- `data.py` reads results and calculates means, Monte Carlo standard errors,
  convergence counts, lower-bound contacts, and runtimes.
- `regret.py` pairs replication IDs and calculates regrets, uncertainty, and ranks.
- `render.py` defines the LaTeX layouts, captions, and method labels.

Means and Monte Carlo standard errors use successful fits with a finite value
for that metric; MCSE is the sample standard deviation divided by the square
root of the valid count. The best-method table selects the unrounded minimum
IBS/RMSE or maximum AUC. The primary best-method comparison includes GG-KM and
spline Poisson/NB models; the high-dimensional best-method comparison additionally
includes promotion time and mixture cure, following its supplied caption. The
detailed metric and diagnostic tables include all 18 configurations.

Regret uses the four RBF/spline Poisson/NB methods. Within each scenario and
metric, the reference has the best finite mean; paired differences use shared
successful replication IDs and check data-generation and split seeds when
recorded. Mean regret weights scenarios equally, with propagated MCSE
`sqrt(sum(scenario_mcse**2)) / number_of_scenarios`. Ranks use scenario means
and average ties. Undefined paired uncertainty stops generation rather than
silently omitting a scenario.

Diagnostics count successful converged fits, successful nonconverged fits, and
failed fits separately. Bound contacts count successful fits with at least one
latency parameter at `1e-5`, or an applicable NB dispersion at `1e-8`, using a
tight relative tolerance. Runtime includes all recorded attempts, including
failures. Missing diagnostic fields raise an error; unavailable metric means or
MCSEs display as `--`. Generated captions use the current replication count.

## Melanoma review figures

Run these scripts on the cluster after the melanoma experiment has saved its
split results and prediction archives:

```bash
python scripts/melanoma_auc_plot.py
python scripts/melanoma_individual_cure_variation.py
```

Both default to `data/melanoma_results/grid_search/` for results,
`data/melanoma.csv` for the original cohort, and `figures/review/` for outputs.
They accept `--results-dir`, `--data-path`, and `--output-dir`, and can be
launched from any working directory. Preserve the cohort's original row order;
stored training/test hashes are checked against the reconstructed holdouts.

`melanoma_auc_plot.py` reproduces the supplied three-panel survival/IBS/cure-AUC
figure and the separate two-panel kernel comparison:

- `melanoma_predictive_assessment_ibs_auc.{pdf,png,svg}`
- `melanoma_ggkm_kernel_comparison_ibs_auc.{pdf,png,svg}`
- `melanoma_predictive_assessment_ibs_auc_summary.csv`

Its main comparison follows the supplied code: mixture cure, promotion time,
both spline models, and one GG-KM model per count law. GG-KM kernels are selected
by highest finite successful mean cure AUC, then lowest mean IBS, then kernel
name. The kernel comparison shows all seven kernels under both count laws.
Saved IBS/AUC values are retained. Saved survival curves are interpolated onto
a 250-point grid within the minimum supported horizon across displayed fits;
this avoids refitting and does not extrapolate beyond the saved grids.

`melanoma_individual_cure_variation.py` recreates the two violin panels in
`melanoma_individual_cure_variation.{pdf,png,svg}` and saves
`melanoma_individual_cure_variation_individual_statistics.csv`. It compares the
linear, spline, polynomial, and RBF predictors under Poisson and NB count laws,
matching the supplied figure. RBF remains the default for this supplementary
comparison; `--ggkm-kernel` selects an explicit alternative. NB distributions
are hatched. Patient means use only that patient's finite outer-test predictions;
sample deviations use `ddof=1` and require at least two appearances. Patients
never predicted and singleton deviations remain missing, and the CSV retains
the appearance counts.

The scripts use the runner's `plot_data/` NPZ archives, including original
`test_indices`, rather than fitting models again. Keep that directory together
with the `splits_melanoma_task*.csv` files when moving results. If an older run
has only CSVs, explicitly enable reconstruction:

```bash
python scripts/melanoma_auc_plot.py --results-dir /path/to/results --refit-missing
python scripts/melanoma_individual_cure_variation.py --results-dir /path/to/results --refit-missing
```

The optional fallback refits missing predictions with saved best parameters and
the current model/preprocessing code, without retuning or changing the saved
results. Malformed or mismatched archives still raise an error. Refit predictions
can differ from the original run if model code or dependencies have changed.
Shared archive validation and reconstruction live in `melanoma_plot_utils.py`
and `melanoma_refit_utils.py`.

## Breast cancer review figures

Generate the corresponding breast cancer figures on the cluster:

```bash
python scripts/breast_cancer_auc_plot.py
python scripts/breast_cancer_individual_cure_variation.py
```

Defaults are `data/breast_cancer_results/grid_search_paired_50/` for results,
`data/cancer_de_mama_rhc_e_fap.csv` for the cohort, and `figures/review/` for
outputs. Both scripts accept `--results-dir`, `--data-path`, `--output-dir`,
and the explicit legacy `--refit-missing` option. The cohort cleaning,
deduplication, event coding, zero-time adjustment, and original patient indices
match `run_breast_cancer_experiment.py`. The holdouts use 30% of the cohort for
testing, and survival plots preserve the recorded `Tempo` units.

The predictive script saves:

- `breast_cancer_predictive_assessment_ibs_auc.{pdf,png,svg}`
- `breast_cancer_ggkm_kernel_comparison_ibs_auc.{pdf,png,svg}`
- `breast_cancer_predictive_assessment_ibs_auc_summary.csv`

As in the melanoma script, GG-KM kernels in the main comparison are selected by
highest mean cure AUC, then lowest mean IBS. The kernel figure includes all
eight breast-cancer kernels, including sigmoid. For older runs, select the
available kernels explicitly, for example:

```bash
python scripts/breast_cancer_auc_plot.py \
  --kernels linear rbf laplacian exponential cauchy polynomial wavelet
```

The individual script saves
`breast_cancer_individual_cure_variation.{pdf,png,svg}` and
`breast_cancer_individual_cure_variation_individual_statistics.csv`. It compares
the same eight linear/spline/polynomial/RBF methods as the melanoma figure,
with means and sample deviations calculated from each patient's held-out
appearances. `--ggkm-kernel` changes the RBF comparison explicitly. Shared plot
rendering and patient summaries are reused from the melanoma modules; the
breast-specific helpers supply the correct data, split design, and optional
refit preprocessor.

## Clinical performance tables

Regenerate both supplied clinical tables with one command:

```bash
python scripts/create_clinical_performance_tables.py
```

This replaces `sections/generated/melanoma-performance-table.tex` and
`sections/generated/breast-cancer-performance-table.tex`, preserving their
labels, columns, method ordering, and decimal precision. It reads the clinical
`splits_*.csv` files from the same default results directories as the plots;
patient data and NPZ files are unnecessary.

For a single cohort or different results locations:

```bash
python scripts/create_clinical_performance_tables.py \
  --dataset breast-cancer \
  --breast-cancer-results-dir /path/to/breast/results \
  --output-dir /path/to/generated-tables \
  --planned-splits 50
```

Use `--dataset melanoma` and `--melanoma-results-dir` for melanoma alone.
`--planned-splits` is optional; the attempted split count is otherwise inferred
from complete CSVs. Split IDs and training/test hashes must agree across the
selected methods. All selected inputs are checked and tables rendered before
any existing table is replaced.

Table parentheses contain sample split standard deviations, rather than Monte
Carlo standard errors. Metric summaries use successful fits with finite values.
Convergence is the number converged among successful fits, and runtime is the
successful-fit mean with its maximum in parentheses. The default tables retain
the 18 active method rows in the supplied files. Add `--include-sigmoid` to
include the two breast cancer sigmoid rows that were commented out in the
reference table. Missing estimates or undefined standard deviations appear as
`--`; missing convergence information raises an error.
