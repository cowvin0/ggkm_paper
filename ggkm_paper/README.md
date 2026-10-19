# GGKM paper experiments

The spline comparison models and their linear primal estimators belong to the
paper code. Import them from the repository checkout:

```python
from ggkm_paper.models import (
    GGNBPrimal,
    GGPoissonPrimal,
    SplineNegativeBinomialModel,
    SplinePoissonModel,
)
```

They retain the scikit-survival estimator API: `fit(X, y)` takes a structured
survival target, and prediction methods include `predict_survival` and
`predict_cure_probability`. The paper's `fit_cure_model` adapter continues to
accept separate time and event arrays for these estimators.

Install `skcure` and the experiment dependencies in your environment, then run
experiments from the repository root as modules, for example:

```bash
python -m pip install -e .
python -m pip install pandas optuna tqdm
python -m ggkm_paper.scripts.run_pcm_experiment
python -m ggkm_paper.scripts.run_melanoma_experiment
python -m ggkm_paper.scripts.run_breast_cancer_experiment
```

The scripts retain their existing experiment grids, task selection through
`PBS_ARRAY_INDEX`, input data paths, and output directories. They run the full
paper experiments; adjust their constants before a small exploratory run.
The clinical and default experiments require their input datasets at the paths
used by each script.

`ggkm_paper` is research code in the checkout and is excluded from the published
`skcure` package. The library's mixture cure, promotion time, kernel, and boosting
models continue to be imported from `skcure.models`.

Run the library and paper regression tests together from the repository root:

```bash
python -m pytest tests ggkm_paper/tests
```
