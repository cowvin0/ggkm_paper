# Experiment, figure, and table jobs

Use [the workflow launcher](../scripts/run_workflow.py) from the repository root to
choose local execution or PBS submission. The launcher connects each dataset's
figure and table jobs to its experiment array. The figure jobs save to
`figures/review/`; the table jobs replace the corresponding files in
`sections/generated/`.

The simulation workflow first runs a separate diagnostic pilot and checks its
projected Monte Carlo precision before starting the final simulation array.
The pilot uses five planned replicates in each of 324 cells (18 primary
scenarios crossed with the 18 reported model configurations). Its samples and
outputs are separate from the final 250-replicate experiment.
Raw pilot replications, their recorded design and checksums, and the CSV, JSON,
and text precision reports are saved in `data/pilot_results/primary/`. The
reported standard deviations and projected errors come from the newly run
pilot; historical numbers in the manuscript are not imposed on its results.

| Dataset | Experiment array | Figures | Tables |
| --- | --- | --- | --- |
| Simulation | `run_experiment.pbs` (0–959) | `create_simulation_figures.pbs` | `create_simulation_tables.pbs` |
| Melanoma | `run_melanoma_experiment.pbs` (0–31) | `create_melanoma_figures.pbs` | `create_melanoma_table.pbs` |
| Breast cancer | `run_breast_cancer_experiment.pbs` (0–31) | `create_breast_cancer_figures.pbs` | `create_breast_cancer_table.pbs` |
| FOSP melanoma | `run_fosp_melanoma_experiment.pbs` (0–31) | `create_fosp_melanoma_figures.pbs` | `create_fosp_melanoma_table.pbs` |
| Latent-count robustness | `run_latent_robustness_experiment.pbs` (0–47) | `create_latent_robustness_figures.pbs` | `create_latent_robustness_tables.pbs` |

The additional simulation jobs are `run_simulation_pilot.pbs` (array 0–323) and
`summarize_simulation_pilot.pbs`. The summary checks that every pilot cell has
at least four successful replicates and that 250 final replicates satisfy the
IBS and cure-probability MSE precision targets. A failed check stops the final
simulation from starting; it does not change the final replication count.

The simulation figure job runs the paired methods, interaction/quadratic
surfaces, and recovered cure surfaces scripts. Each clinical figure job runs
the predictive performance and individual cure variation scripts. The simulation
table job regenerates all simulation and high-dimensional tables; each clinical
table job regenerates its dataset's performance table.

All figure jobs also run `scripts/create_score_plots.py` to export IBS and Uno's
C-index separately, each as PDF, PNG, and SVG, while retaining the existing
combined IBS/cure-AUC figures. Clinical outputs include separate model and
seven-kernel comparisons. Simulation outputs include separate primary and
higher-dimensional paired contrasts with the same Monte Carlo intervals.
The additional figures use saved CSV scores and do not refit models. For FOSP,
the main standalone files are `fosp_melanoma_predictive_assessment_ibs.*` and
`fosp_melanoma_predictive_assessment_uno_c_index.*` in `figures/review/`.

Breast cancer uses the same seven kernels, model parameter grids, and three
inner folds as melanoma. Sigmoid is excluded. Updated breast cancer outputs go
to `data/breast_cancer_results/grid_search_paired_50_melanoma_design/`, so the
previous eight-kernel, four-fold results are kept separate. The breast cohort's
30% outer holdout and preprocessing are retained. The figure job writes the
predictive assessment, seven-kernel comparison, and individual cure variation
as PDF, PNG, and SVG in `figures/review/`.

FOSP melanoma uses `data/melanoma_fosp_gomez2023.csv`, preserving its row order
and the extended follow-up (`tempo_ext` in years, `status_ext` as the event).
Its 50 paired 70/30 holdouts, three inner folds, models, tuning grids, scoring,
and plots follow the same clinical workflow. Stage, topography, and morphology
use the breast-cancer preprocessor's smoothed survival target encoding: one
RMST feature per category column, five-fold out-of-fold training encodings,
`smooth=10`, and `random_state=42`. Test mappings and continuous scaling use
training data only; binary sex and treatment indicators are retained. Results
go to `data/fosp_melanoma_results/grid_search/`, figures use the
`fosp_melanoma_` prefix, and the table is
`sections/generated/fosp-melanoma-performance-table.tex`.

Wavelet tuning retains `A=0.5,1,2` and includes `A=10,15,20` in both clinical
and simulation workflows. Clinical checkpoint settings identify the grid so
earlier fits cannot silently bypass the new candidates. Tuning continues to
minimize IPCW IBS. Tables additionally report the already saved Uno's C-index,
using negative RMST risk scores and training-sample censoring weights; cure
AUC and the existing plots are retained. Simulation outputs also include
per-method Uno C-index tables and the metric in best-method and regret tables.

## One launcher for local and cluster execution

Activate the Python environment containing the experiment dependencies before
launching the workflow. Experiment stages import each runner's current
`build_experiments()` to determine its array size, including during a dry run.
For example, on the cluster:

```bash
source "$HOME/venvs/master_thesis/bin/activate"
python scripts/run_workflow.py --mode cluster --dry-run
```

The dry run prints the selected jobs, current task counts, and commands without
running experiments or submitting jobs. To submit all experiments, figures, and
tables:

```bash
python scripts/run_workflow.py --mode cluster
```

The launcher submits experiment arrays and connects each dataset's figure and
table jobs with an `afterok` dependency on its complete array. For simulation,
the dependency chain is pilot array, pilot summary, final experiment array,
then figures and tables. Clinical experiment arrays can run independently of
the simulation pilot. All array tasks must finish successfully before the
dependent jobs can run; see the
[PBS Professional User's Guide, §§8.4.5–8.4.6](https://help.altair.com/2022.1.0/PBS%20Professional/PBSUserGuide2022.1.pdf).

To run only the diagnostic pilot and its summary on the cluster:

```bash
python scripts/run_workflow.py --mode cluster --datasets simulation --stage pilot
```

Use `--mode local` with the same arguments to run the pilot locally. Neither
command runs the final simulation or rewrites its existing results.

To regenerate only figures and tables from existing results on the cluster:

```bash
python scripts/run_workflow.py --mode cluster --stage postprocess
```

To choose one cohort and a queue:

```bash
python scripts/run_workflow.py --mode cluster --datasets melanoma --queue YOUR_QUEUE
```

Local execution uses the same Python routines and output paths. It runs array
tasks sequentially, with one outer-holdout worker per clinical task by default:

```bash
python scripts/run_workflow.py --mode local --dry-run
python scripts/run_workflow.py --mode local
```

To regenerate only the breast cancer figures and tables locally:

```bash
python scripts/run_workflow.py --mode local --datasets breast-cancer --stage postprocess
```

Postprocessing reads saved CSV/NPZ results; those files must be available on the
machine running it. Use `--stage figures` or `--stage tables` for only one output
type, or `--stage experiments` to run experiments, including the simulation
pilot and its precision check. Postprocessing stages never run the pilot.
`--datasets` accepts
any combination of `simulation`, `melanoma`, `breast-cancer`, `fosp-melanoma`,
and `latent-robustness`.

To preview or submit only the new application:

```bash
python scripts/run_workflow.py --mode cluster --datasets fosp-melanoma --dry-run
python scripts/run_workflow.py --mode cluster --datasets fosp-melanoma
```

Use `--python /path/to/venv/bin/python` to select a different interpreter and
`--workers N` to set concurrent clinical outer holdouts per task. Cluster jobs
currently request four CPUs, so the launcher accepts at most four workers in
cluster mode. Each launch records commands and progress or submitted job IDs
in `logs/workflow/<run>/workflow.json`; local task logs are stored alongside it.
Use `--log-dir /path/to/new-directory` to choose a new location. Run
`python scripts/run_workflow.py --help` for all options.

## Cluster environment

All jobs share [common.sh](common.sh). Its defaults match the original jobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GGKM_PROJECT_ROOT` | Launcher: repository root; direct submission: PBS submission directory | Repository root. A submission directory containing `ggkm_paper/` also works. |
| `GGKM_MODULES` | `gcc/14.2.0 python/3.12.8 pcre2/10.44` | Modules loaded after `module purge`; set to an empty string to skip modules. |
| `GGKM_VENV` | `$HOME/venvs/master_thesis` | Virtual environment directory; set to an empty string to skip activation. |
| `GGKM_PYTHON` | Launcher: selected `--python` (default: launcher's interpreter); direct submission: `python` after activation | Python interpreter used by the job. |
| `GGKM_R_HOME` | `$HOME/master_thesis/ggkm/utils/R-4.4.3` | R installation; a nonempty value also adds its `lib/` to `LD_LIBRARY_PATH`. |
| `GGKM_WORKERS` | Breast cancer and FOSP melanoma: 2; other experiment jobs: allocated `PBS_NCPUS`, or 4; figure/table jobs: 1 | Number of concurrent outer holdouts used by the clinical runners, passed through `PBS_NCPUS`. |

Export module, virtual environment, and R settings before running the launcher.
For example:

```bash
export GGKM_PROJECT_ROOT=/path/to/ggkm_paper
export GGKM_VENV=/path/to/venv
export GGKM_R_HOME=/path/to/R
```

The launcher exports its selected Python interpreter to every PBS job. The
shared job setup loads modules and activates `GGKM_VENV` before invoking that
interpreter. Select it with `--python`; an existing `GGKM_PYTHON` setting is
replaced by the launcher's selection. Local mode uses the current environment
directly.

The PBS headers retain the experiment CPU, memory, and walltime requests, with
the breast cancer memory request corrected to `64gb`. Figure jobs request one
CPU and `16gb`; table jobs request one CPU and `4gb`. Edit their `#PBS -l` lines
to suit the cluster's queues and limits.
The FOSP experiment uses the breast-cancer allocation: four CPUs, `64gb`, and
336 hours per task.

The pilot array requests four CPUs, `16gb`, and 72 hours per task; its summary
requests one CPU, `4gb`, and one hour.

## Latent-count misspecification study

The robustness workflow is independent of the existing paper experiments:

```bash
python scripts/run_workflow.py --mode cluster --datasets latent-robustness --dry-run
python scripts/run_workflow.py --mode cluster --datasets latent-robustness
```

It is also included in the default all-datasets workflow. Its dependency chain
is `run_latent_robustness_pilot.pbs` (0–47),
`summarize_latent_robustness_pilot.pbs`,
`run_latent_robustness_experiment.pbs` (0–47), then the corresponding figure and
table jobs. Each array task is one generating scenario and fits all three count
laws to every replication, so they share the actual data and splits.

The 40 primary scenarios cross Poisson, NB with `phi=0.25,1,2`, and binomial
`K=2` truth with `n=300,600`, quadratic/periodic cure surfaces, and long/short
follow-up. Eight additional two-component Poisson-mixture scenarios stress
truth outside all fitted count families. All preserve the same cure surfaces
through the count-specific zero probabilities. Censoring is independent
exponential, with its mean calibrated using 32,768 separate observations to
target 20% or 50% censoring **among susceptible subjects**, respectively.
Realized censoring fractions are recorded separately.

Fitted models are Poisson, estimated-dispersion NB, and binomial `K=2` GGKM, all
using RBF and the existing PCM penalty/bandwidth grid, inner folds, IBS tuning
objective, and 70/30 split. The full-model fits estimate GG latency. The 20
primary `n=600` scenarios also run the fixed-true-latency diagnostic using the
same datasets; only its latency M-step is disabled. Predictor, E-step, and
dispersion updates use the shared GGKM implementation. This gives 204 fitted
scenario/model/latency cells, or **51,000 final outer-fit attempts** at 250
replications, plus inner tuning.

The independent pilot uses 10 replications and separate seeds; its 2,040 outer
attempts never enter final estimates. Its summary checks the planned 250
replications against target paired MCSEs of 0.003 for IBS deterioration and
0.005 for cure-RMSE deterioration. Allocation uses each contrast's paired
sample SD and observed finite-pair fraction. An incomplete pilot or insufficient
projected precision stops the final dependency chain; the replication count is
never changed automatically. Inspect `pilot/reports/pilot_precision.json` to
adjust the allocation or explicitly rerun the summary with
`--allow-insufficient-precision` for a complete pilot. This option does not
claim that the precision target was achieved.

To change the final allocation without refitting or invalidating the pilot,
export `GGKM_ROBUSTNESS_REPLICATIONS` before launching. Both the precision
check and final array use it. If final results already exist at another
allocation, also select a new final directory, for example:

```bash
export GGKM_ROBUSTNESS_REPLICATIONS=500
export GGKM_ROBUSTNESS_FINAL_DIR=/path/to/data/latent_robustness/final_500
python scripts/run_workflow.py --mode cluster --datasets latent-robustness
```

`GGKM_ROBUSTNESS_PILOT_DIR` optionally relocates the existing pilot directory.
The launcher exports these settings to the dependent jobs. Changing the
allocation never pools pilot observations with final results or mixes final
directories with different study identities.

Study artifacts are separate from `data/results/` and existing paper tables:

- `data/latent_robustness/pilot/` and `data/latent_robustness/final/`: recorded
  designs, source/software fingerprints, calibrated censoring, resumable
  replication checkpoints, and raw result CSVs.
- `samples/case_*/replication_*.npz` within each phase: covariates, true latent
  counts, true/censored times, cure probabilities, and the exact train/test
  indices and scoring grids. Latent truth is used only for evaluation.
- `final/reports/`: absolute performance, paired raw differences and summaries,
  fixed-versus-estimated latency contrasts, two LaTeX table files, and JSON
  interpretation metadata. Every planned attempt and failure is retained.
- `figures/review/latent_robustness/`: separate IBS, cure RMSE, survival RMSE,
  and Uno C-index heatmaps for estimated latency, fixed latency, and the mixture
  stress experiment, each as PDF, PNG, and SVG.

All error contrasts are fitted minus the correct count-family reference;
C-index contrasts are reference minus fitted, so positive is worse throughout.
NB under Poisson truth is labeled as a limiting case. Mixture scenarios have
no correct fitted reference and are evaluated against known truth in the
absolute summaries. Survival RMSE uses a common training-supported horizon;
the additional `tail_survival_rmse` evaluates a fixed grid up to the 99th
percentile of the activation distribution, including extrapolation when needed.

Practical robustness margins default to unset. Table generation can take
`--ibs-margin` and `--cure-rmse-margin` after these are scientifically specified;
otherwise it records `not_assessed`. Monte Carlo intervals quantify simulation
uncertainty, and similarity does not identify the true latent count law.

To run a small reproducibility check without submitting jobs:

```bash
python -m scripts.run_latent_robustness_experiment --phase pilot --task-id 0 \
  --replications 2 --workers 1 --output-dir /tmp/ggkm-robustness-check
```

For only the pilot, use the workflow's `--stage pilot`. For reports/figures from
complete final CSVs, use `--stage postprocess`. Use the same environment and
source version when resuming; changed grids, sources, seeds, or software require
a separate output directory. Fit-attempt failures are recorded scientific
outcomes, whereas missing/corrupted output files prevent postprocessing.

Robustness arrays request four CPUs, `16gb`, and 336 hours for final estimation
or 72 hours for the pilot; replication workers default to the allocated CPU
count. Figure and table jobs request one CPU and `8gb`; the pilot summary uses
one CPU and `4gb`. They use the same logging and environment setup as the other
jobs. No existing paper results or figures are overwritten by this study.

Each job uses one BLAS/OpenMP thread per Python process and the `Agg` plotting
backend. Standard output, standard error, and the final exit status/timing are
recorded in `logs/pbs/`, with the job key, PBS job ID, and array index in the
filenames. A failed command returns a nonzero job exit status.

## Submitting individual jobs

From the repository root, the original experiment commands still work:

```bash
qsub pbs_jobs/run_experiment.pbs
qsub pbs_jobs/run_melanoma_experiment.pbs
qsub pbs_jobs/run_breast_cancer_experiment.pbs
qsub pbs_jobs/run_fosp_melanoma_experiment.pbs
```

To submit the pilot and its dependent summary directly:

```bash
pilot_job_id=$(qsub pbs_jobs/run_simulation_pilot.pbs)
qsub -W "depend=afterok:${pilot_job_id}" pbs_jobs/summarize_simulation_pilot.pbs
```

Figure and table jobs can be submitted directly when the corresponding raw
CSV/NPZ results already exist. For example:

```bash
qsub pbs_jobs/create_melanoma_figures.pbs
qsub pbs_jobs/create_melanoma_table.pbs
```

For direct `qsub` calls, custom environment settings must also be passed to
PBS, for example:

```bash
qsub -v GGKM_PROJECT_ROOT,GGKM_VENV,GGKM_R_HOME pbs_jobs/create_melanoma_figures.pbs
```

The workflow launcher handles its supported settings automatically. Direct
submissions do not create dependencies; use the launcher to submit the complete
workflow, including the pilot and its precision check before final simulation.
