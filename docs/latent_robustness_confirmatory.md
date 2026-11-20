# Confirmatory latent-count misspecification study

This is a separate workflow. The previous study, paper experiments, application
jobs, and existing reports under `tmp/` are preserved. New results default to
`data/latent_robustness/confirmatory/`.

The scientific question is whether **mean predictive deterioration** from
changing the fitted count family remains below scientifically acceptable
limits in the declared simulation scenarios. This does not test recovery of
the true latent count law or prove robustness for every possible count law.

## Compact storage for restricted cluster quotas

The default protocol uses `storage.format: compact-v1` and an **80 GiB output
budget**. The original full format duplicated optimizer traces across all starts,
profiles, and inner folds; with a large pilot-based allocation this can exceed
100 GB. Compact storage changes persistence and reporting, not generated data,
models, tuning, margins, seeds, convergence rules, or replication allocation.

- Audit samples and complete audit traces are retained; diagnostics use gzip.
- Pilot/final samples are represented by deterministic seeds and an array hash.
  Resuming/reporting regenerates them and verifies the hash and pairing fields.
- Pilot/final checkpoints and diagnostics are compressed. Outer-start statuses,
  objectives, fitted parameters, boundary flags, and solver-failure counts are
  retained. Iteration histories become summaries; inner-fold journals retain
  start/convergence/failure counts and the selected start. Full iteration traces
  are available for the audit, rather than duplicated for every later fit.
- `paired_replications.csv.gz` preserves every paired value. Reports process
  one scenario at a time rather than collecting millions of pairs in memory.

Use a **new directory** after updating the code, because source hashes and the
storage policy are frozen. Existing full-format studies are not silently
converted or resumed with different source hashes. No old files are deleted.

```bash
python scripts/run_confirmatory_workflow.py --mode cluster --stage pilot \
  --output-dir data/latent_robustness/confirmatory_compact
cat data/latent_robustness/confirmatory_compact/pilot/storage_plan.json
cat data/latent_robustness/confirmatory_compact/pilot/allocation.json
# When allocation.json reports ready=true:
python scripts/run_confirmatory_workflow.py --mode cluster --stage final \
  --output-dir data/latent_robustness/confirmatory_compact
```

The storage projection uses the largest observed per-replication allocation in
each pilot case, filesystem-block/inode allowances, reporting space, and a
twofold safety factor. If it exceeds the frozen budget, the pilot summary exits
nonzero and blocks final jobs. It never reduces R or relaxes margins to fit.
This is a conservative estimate, not a guarantee of the filesystem's quota;
other files in the account and other runs also consume space. 80 GiB is about
86 GB, so a 100 GB account must leave the remaining space available. Check the
new pilot projection before committing to the large final run.

## Start with the optimization audit

From the repository root, with the experiment environment active:

```bash
python scripts/run_confirmatory_workflow.py --mode cluster --stage audit --dry-run
python scripts/run_confirmatory_workflow.py --mode cluster --stage audit
```

If you already ran the original audit, keep its results and use a new directory
for the revised profile-restart fitting procedure:

```bash
python scripts/run_confirmatory_workflow.py --mode cluster --stage audit \
  --output-dir data/latent_robustness/confirmatory_v2
```

Use that same `--output-dir data/latent_robustness/confirmatory_v2` for every
subsequent pilot, final, or report command below. Updated source hashes prevent
mixing revised fits with the previous audit. There is no need to delete data.

Use `--mode local --workers 1` for local execution. Cluster execution uses the
same module/virtual-environment settings as the other PBS jobs; `GGKM_PYTHON`,
`GGKM_VENV`, `GGKM_MODULES`, and `GGKM_PROJECT_ROOT` retain their usual meanings.
Use `--protocol /absolute/path/protocol.json` and `--output-dir /absolute/path/new-study`
to choose a different protocol or destination. The launcher exports both paths
to every PBS job. `GGKM_WORKERS` or `--workers` can limit the outer replication
processes to at most four on the cluster.

The audit uses five independent replications in six scenarios: Poisson and
NB(phi=1) under n=600/quadratic/long follow-up, and all four n=600/NB(phi=2)
surface/follow-up combinations. All three count families are fitted. It also
includes fixed-latency fits and true-fixed-dispersion NB controls where applicable.

At every kernel/penalty candidate and inner fold, the new wrapper runs the
declared initializations and selects the highest finite penalized likelihood.
The default latency starts are (1,1,1) and (1.6,1.3,1.1); the estimated NB
dispersion starts are 0.0001, 0.25, 1, and 2. A nonconverged or failed start is
retried once from the same initialization at twice its original EM budget.
Objectives are compared only within the same kernel and penalty. Inner IBS
still selects the tuning configuration. Test data never select starts or tuning.

With `fitting.profile_restart: true`, every free-NB fit also evaluates the
declared fixed-dispersion/Poisson profile on its own training data, including
each inner fold. It then restarts the free-NB ECME updates from the complete
best profile solution: intercept, kernel coefficients, latency, and dispersion.
Dispersion is estimated again; a fixed-profile convergence flag is never
inherited. The Poisson endpoint initializes NB at the existing numerical lower
bound, phi=1e-8, with the initial NB objective recomputed. The ordinary starts
remain candidates, and selection still uses the highest finite objective.
This adds training fits and therefore increases runtime.

Continuation rejects objective decreases larger than 1e-8 and invalid training
survival/cure predictions. Failed continuations are recorded and cannot replace
the free fit with a fixed-profile solution. Diagnostics include restart sources,
the initial/final objectives, a hash of the transferred predictor coefficients,
solver histories, and flags for latency/dispersion at their existing lower
bounds. A boundary flag is descriptive, not by itself an audit failure. No upper
latency bound or change to the audit tolerance is introduced.

For each selected audit NB configuration, the profile refits the other
parameters with phi fixed at 0.0001, 0.25, 0.5, 1, 2, and 4, and fits an exact
Poisson model for the phi=0 point. Both estimated- and fixed-latency profiles
are available. The selected fit's training profile is reused when already
computed. Profile objectives share the same selected kernel and penalty.

**NB approaching Poisson as phi approaches zero is expected. A small fitted
dispersion does not itself fail the audit.** The audit stops progression if a
selected fit/profile does not converge, a fit fails, or a profile improves the
free-NB objective by more than the declared numerical tolerance (0.01).
Inspect `audit/report.json`, `audit/dispersion_profiles.csv`,
`audit/performance_summary.csv`, `audit/dispersion_profile_case_*.pdf`, and the
full `audit/diagnostics/` histories.
A passed audit does not prove a global optimum or identifiability.

If the audit flags a better solution, investigate before launching confirmation;
revise the fitting protocol and rerun the audit in a new output directory. Do
not edit readiness flags to bypass a fitting problem.

## Selected scientific margins

The protocol in `config/latent_robustness_confirmatory.json` now sets:

- `margins.integrated_brier`: **0.002 absolute increase in mean IBS**.
- `margins.cure_rmse`: **0.010 absolute increase in mean cure-probability RMSE**.
- `margin_justification`: the benchmark-based rationale and the selection timing.

The paper's primary quadratic-surface simulations report RBF-versus-linear
improvements of approximately 0.007-0.012 in IBS and 0.04-0.07 in cure RMSE
([discussion, principal simulation results](../sections/06_discussion.tex)).
The selected margins allow losing approximately 17-29% of the IBS improvement
and 14-25% of the cure-RMSE improvement. Thus they preserve at least approximately
71% and 75%, respectively, of those reported benchmark gains. This benchmark
motivates common absolute tolerances; it does not imply equal benchmark gains
in every robustness cell.

These are study-specific research judgments, not clinically validated or
universal cutoffs. They were selected after inspecting the paper and the
exploratory misspecification results, before the new independent pilot/final
outcomes. Disclose that timing when reporting confirmation. They were not
selected to make every historical comparison pass. The same limits apply to
all primary scenarios and both analysis sets, and remain fixed even if a
comparison fails or is inconclusive.
The old MCSE targets (0.003 and 0.005) are precision goals, not noninferiority margins.

For exploratory sensitivity around the selected margins, generate a planning
review from the existing summaries:

```bash
python -m scripts.confirmatory_robustness --stage margin-review \
  --exploratory-summary tmp/paired_summary.csv \
  --output-dir tmp/confirmatory-margin-review
```

Read `tmp/confirmatory-margin-review/margin_review/margin_review.md`. Cure RMSE
is expressed on the probability scale: an increase from 0.08 to 0.09 is a
one-percentage-point increase in RMSE, without implying that every individual's
prediction changes by one point. IBS is an average IPCW squared error and needs
a separate anchor. One possible justification is preserving a prespecified
fraction of the model's useful gain over a simpler benchmark, established
independently of these misspecification losses. The generated sensitivity grid
shows consequences of illustrative margins without changing the selected
primary margins.
If no practical margin can be justified, report the bounds without a binary
practical-robustness claim.

Margins can be supplied after an audit without invalidating it. Before the
pilot begins, the full protocol, sources, software versions, and tuning grids
are frozen. Subsequent changes require a new output directory.

Then launch the full chain (the audit is reused from verified checkpoints):

```bash
python scripts/run_confirmatory_workflow.py --mode cluster --stage all --dry-run
python scripts/run_confirmatory_workflow.py --mode cluster --stage all
```

To run only through the independent pilot and inspect the allocation first:

```bash
python scripts/run_confirmatory_workflow.py --mode cluster --stage pilot
# After the pilot reports ready:
python scripts/run_confirmatory_workflow.py --mode cluster --stage final
```

The PBS dependency chain is:

```text
audit [0-5] -> audit-summary -> pilot [0-47] -> pilot-summary
            -> final [0-47] -> report
```

Audit/pilot failures exit nonzero, preventing dependent final jobs from running.
Runtime gates also check the prerequisite reports, so manually submitting a
final array cannot bypass the scientific protocol requirements. PBS logs are
in `logs/pbs/`; launcher manifests record submitted IDs in `logs/confirmatory_*/`.

## Independent pilot and replication allocation

The pilot uses 30 fresh replications per scenario. Audit, pilot, and final seeds
are 20261120, 20261121, and 20261122, respectively; these differ from the historical
study and calibration seeds. No audit/pilot observations enter final estimates.

There are 40 primary cells: Poisson, NB(phi=0.25,1,2), and binomial(K=2) truths,
n=300/600, quadratic/periodic surfaces, and long/short follow-up. Within a cell,
every fitted count family shares the generated dataset, split, inner-fold seed,
and training-defined evaluation grid. Generating cure surfaces, latency, and
censoring calibration follow the original study. The RBF tuning grid is unchanged:
gamma in {0.25,1} and lambda in {0.3,3}, with two inner folds and a 70/30 split.

The eight additional Poisson-mixture cells remain descriptive stress tests.
No correctly specified mixture model is fitted; agreement among fitted models
therefore cannot establish noninferiority relative to the true mixture family.

For primary inference, each nonreference family is compared with the fit of
the correctly specified count family, whose unknown parameters are estimated.
The NB-under-Poisson comparison is explicitly a limiting-family control.
The design includes fixed-latency diagnostics in all n=600 primary cells and
NB fits with phi fixed at its generating value in the 12 n=600 NB-truth cells.
These are controls; they do not replace the main estimated-parameter reference.

The pilot plans precision for 320 primary contrasts:
40 cells x 2 nonreference fits x 2 primary metrics x 2 analysis sets.
For each contrast, it uses the paired SD and finite-pair fraction to choose
enough replications for the projected Bonferroni-t halfwidth to be no more than
one quarter of the corresponding practical margin. The maximum requirement
across contrasts is used, with a minimum of 356 replications.

`final_replications: null` declares this pilot-based allocation rule in advance:
the pilot writes the chosen R to `pilot/allocation.json`, and the final stage
uses exactly that allocation. An explicit `final_replications` is also allowed,
but the pilot blocks progression if it is below the precision requirement.
R is never updated using final outcomes. This is precision planning, not a
guarantee that noninferiority will be established when the true loss is near a margin.

## Decision and outputs

For IBS and cure RMSE, define the replication difference as misspecified-fit
error minus correct-family-fit error. Positive values mean deterioration.
The primary analyses are:

1. Every pair with finite predictions from successfully completed fits, including
   completed nonconverged fits. A missing primary pair blocks a full-design verdict.
2. A sensitivity analysis retaining pairs where both selected fits converged.
   Its retained counts and fractions are reported; this subset can be selected
   by convergence and should be interpreted together with the first analysis.

For both analyses, calculate one-sided upper Monte Carlo bounds using
`mean + t(1-alpha/320, valid_pairs-1) * paired_SD/sqrt(valid_pairs)`.
The Bonferroni adjustment covers every declared primary contrast and both
analysis sets. These Student-t bounds are finite-simulation approximations.
C-index, survival RMSE, fixed-parameter diagnostics, and mixture comparisons
are secondary and do not acquire a formal primary robustness verdict.

Final decisions in `final/report.json`:

- `within_prespecified_margins`: every primary upper bound is below its margin
  in both analyses.
- `exceeds_margin`: a primary lower bound is above its margin. Lower bounds
  use a separate simultaneous one-sided construction for diagnosing deterioration.
- `inconclusive`: complete comparisons remain too uncertain or the two analyses disagree.
- `incomplete`: primary comparisons or finite primary pairs are missing.

An inconclusive noninferiority test is not evidence of unacceptable deterioration.
The verdict concerns mean performance in this design; individual-replication
losses can be larger. Favorable differences cannot offset a failed scenario.

Final artifacts include:

- `performance_summary.csv`: absolute performance, attempted/completed/converged counts.
- `paired_replications.csv.gz` in compact mode (`.csv` in full mode): every
  replication-level difference with analysis-set labels; pandas reads either directly.
- `paired_summary.csv`: margins, pointwise descriptive intervals, simultaneous
  bounds, retained counts, and individual assessments.
- `margin_sensitivity.csv`: consequences of an illustrative margin grid,
  without changing the prespecified primary margins.
- `fixed_dispersion_controls.csv` and `latency_compensation.csv`: compensation diagnostics.
- `confirmatory-contrasts.tex` and `conclusion.md`: reviewable tables and the verdict.
- Separate `confirmatory_integrated_brier.{pdf,png}` and
  `confirmatory_cure_rmse.{pdf,png}` figures with shared axes between analyses.
- `samples/` for the audit, compressed `checkpoints/` and `diagnostics/`: exact
  pairing, sample hashes, starting values, retries, and optimizer diagnostics,
  with the trace-retention policy described above.

Before using a strong manuscript claim, review the audit and convergence
diagnostics alongside the formal verdict. Parameter identification is a
different question from predictive stability under count-law misspecification.

Methodological references: [Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086)
for simulation design and Monte Carlo uncertainty; [Lakens (2017)](https://doi.org/10.1177/1948550617697177)
for meaningful bounds and prespecification. The one-sided noninferiority rule
above is the declared decision rule for this particular experiment.
