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
| Breast cancer | `run_breast_cancer_experiment.pbs` (0–35) | `create_breast_cancer_figures.pbs` | `create_breast_cancer_table.pbs` |

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
any combination of `simulation`, `melanoma`, and `breast-cancer`.

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
| `GGKM_WORKERS` | Breast cancer: 2; other experiment jobs: allocated `PBS_NCPUS`, or 4; figure/table jobs: 1 | Number of concurrent outer holdouts used by the clinical runners, passed through `PBS_NCPUS`. |

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

The pilot array requests four CPUs, `16gb`, and 72 hours per task; its summary
requests one CPU, `4gb`, and one hour.

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
