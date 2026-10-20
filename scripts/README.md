# Experiment scripts

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
