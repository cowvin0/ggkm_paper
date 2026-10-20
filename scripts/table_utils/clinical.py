"""Read paired clinical split CSVs and render the paper's performance tables.

Only NumPy, pandas, and the standard library are needed. Saved predictions and
the original patient data are unnecessary, and no estimators are imported.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClinicalMethod:
    model: str
    kernel: str = "parametric"

    @property
    def predictor(self) -> str:
        if self.model == "mixture_cure":
            return "Mixture cure"
        if self.model == "promotion_time":
            return "Promotion time"
        if self.model.startswith("spline_"):
            return "Spline"
        label = {"rbf": "RBF", "laplacian": "Laplacian", "cauchy": "Cauchy"}.get(
            self.kernel, self.kernel,
        )
        return f"GG-KM ({label})"

    @property
    def count_law(self) -> str:
        if self.model == "mixture_cure":
            return "Mixture cure"
        return "Negative binomial" if self.model in ("bn", "spline_bn") else "Poisson"


KERNELS = ("linear", "rbf", "laplacian", "exponential", "cauchy", "polynomial", "wavelet")
METRICS = ("integrated_brier", "cure_auc", "mean_predicted_cure")
_REQUIRED_COLUMNS = frozenset((
    "split", "status", "converged", "runtime_seconds", "train_index_hash",
    "test_index_hash", *METRICS,
))
_READ_COLUMNS = _REQUIRED_COLUMNS | {"model_name", "kernel", "inner_seed", "training_size", "test_size"}
_FILE_PATTERN = re.compile(
    r"^splits_(?P<dataset>melanoma|breast_cancer)_task\d+_"
    r"(?P<model>mixture_cure|promotion_time|spline_poisson|spline_bn|poisson|bn)"
    r"(?:_kernel_(?P<kernel>[^/]+))?\.csv$"
)


def methods(*, include_sigmoid: bool = False) -> tuple[ClinicalMethod, ...]:
    """Match the active row order of the supplied clinical LaTeX tables."""
    kernels = KERNELS
    if include_sigmoid:
        kernels = (*KERNELS[:5], "sigmoid", *KERNELS[5:])
    return (
        ClinicalMethod("mixture_cure"), ClinicalMethod("promotion_time"),
        ClinicalMethod("spline_poisson"), ClinicalMethod("spline_bn"),
        *(ClinicalMethod(model, kernel) for kernel in kernels for model in ("poisson", "bn")),
    )


def _numeric(frame: pd.DataFrame, column: str, path: Path) -> pd.Series:
    try:
        return pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: column {column!r} contains nonnumeric values.") from error


def _convergence(value, path: Path):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "1.0"):
            return True
        if normalized in ("false", "0", "0.0"):
            return False
    raise ValueError(f"{path}: invalid convergence value {value!r}; expected true/false or 1/0.")


class ClinicalResults:
    """Validate complete paired holdouts and summarize successful finite values."""

    def __init__(
        self, results_dir: str | Path, dataset: str, *, include_sigmoid: bool = False,
    ):
        if dataset not in ("melanoma", "breast-cancer"):
            raise ValueError("dataset must be 'melanoma' or 'breast-cancer'.")
        if include_sigmoid and dataset != "breast-cancer":
            raise ValueError("The optional sigmoid rows apply only to breast cancer.")
        self.dataset = dataset
        self.results_dir = Path(results_dir).expanduser().resolve()
        self.methods = methods(include_sigmoid=include_sigmoid)
        self._paths: dict[ClinicalMethod, Path] = {}
        self._frames: dict[ClinicalMethod, pd.DataFrame] = {}
        if not self.results_dir.is_dir():
            raise FileNotFoundError(
                f"{dataset} results directory does not exist: {self.results_dir}. "
                "Run on the cluster or supply the corresponding --*-results-dir option."
            )
        for path in sorted(self.results_dir.glob("splits_*.csv")):
            match = _FILE_PATTERN.fullmatch(path.name)
            if match is None or match["dataset"] != dataset.replace("-", "_"):
                continue
            method = ClinicalMethod(match["model"], match["kernel"] or "parametric")
            if method not in self.methods:
                continue
            if method in self._paths:
                raise ValueError(
                    f"Duplicate {dataset} split CSVs for {method.model}/{method.kernel}:\n"
                    f"{self._paths[method]}\n{path}"
                )
            self._paths[method] = path

    def path(self, method: ClinicalMethod) -> Path:
        try:
            return self._paths[method]
        except KeyError as error:
            raise FileNotFoundError(
                f"Missing {self.dataset} split CSV for {method.model}/{method.kernel} "
                f"in {self.results_dir}. Use splits_*.csv files saved by the experiment runner."
            ) from error

    def frame(self, method: ClinicalMethod) -> pd.DataFrame:
        if method in self._frames:
            return self._frames[method]
        path = self.path(method)
        try:
            with path.open(newline="", encoding="utf-8-sig") as source:
                header = next(csv.reader(source), [])
            if len(header) != len(set(header)):
                raise ValueError(f"{path}: duplicate CSV column names.")
            frame = pd.read_csv(
                path, usecols=lambda column: column in _READ_COLUMNS,
                dtype={"train_index_hash": "string", "test_index_hash": "string"},
            )
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            raise ValueError(f"Cannot read clinical split CSV {path}: {error}") from error
        missing = _REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}.")
        if frame.empty:
            raise ValueError(f"{path}: split CSV has no rows.")
        split = _numeric(frame, "split", path)
        if not np.isfinite(split).all() or (split < 0).any() or (split % 1 != 0).any():
            raise ValueError(f"{path}: split IDs must be nonnegative integers without missing values.")
        frame["split"] = split.astype(np.int64)
        if frame["split"].duplicated().any():
            raise ValueError(f"{path}: duplicate split IDs.")
        status = frame["status"].astype("string").str.strip().str.lower()
        if not status.isin(("ok", "failed")).all():
            raise ValueError(f"{path}: status must be explicitly 'ok' or 'failed' on every row.")
        frame["status"] = status
        for column in (*METRICS, "runtime_seconds"):
            frame[column] = _numeric(frame, column, path)
        frame["converged"] = pd.array(
            [_convergence(value, path) for value in frame["converged"]], dtype="boolean",
        )
        successful = frame["status"].eq("ok")
        if frame.loc[successful, "converged"].isna().any():
            raise ValueError(f"{path}: missing convergence status for a successful fit.")
        runtime = frame.loc[successful, "runtime_seconds"]
        if (runtime.loc[np.isfinite(runtime)] < 0).any():
            raise ValueError(f"{path}: successful runtime_seconds values must be nonnegative.")
        for column, expected in (("model_name", method.model), ("kernel", method.kernel)):
            if column in frame:
                if not frame[column].astype("string").eq(expected).fillna(False).all():
                    raise ValueError(f"{path}: {column!r} metadata does not match the filename ({expected!r}).")
        for column in ("train_index_hash", "test_index_hash"):
            if frame[column].isna().any() or frame[column].str.strip().eq("").any():
                raise ValueError(f"{path}: missing {column}; paired holdouts cannot be verified.")
        frame = frame.sort_values("split").reset_index(drop=True)
        self._frames[method] = frame
        return frame

    def validate(self, planned_splits: int | None = None) -> int:
        """Require complete 0-based split IDs and matching train/test hashes."""
        if planned_splits is not None and planned_splits < 1:
            raise ValueError("--planned-splits must be a positive integer.")
        frames = [self.frame(method) for method in self.methods]
        counts = {len(frame) for frame in frames}
        if len(counts) != 1:
            raise ValueError(
                f"{self.dataset}: inconsistent numbers of attempted holdouts "
                f"({', '.join(str(count) for count in sorted(counts))}); use completed runs."
            )
        count = counts.pop()
        if planned_splits is not None and count != planned_splits:
            raise ValueError(f"{self.dataset}: expected {planned_splits} planned splits, but CSV files contain {count}.")
        expected_ids = np.arange(count, dtype=np.int64)
        reference = frames[0]
        for method, frame in zip(self.methods, frames):
            path = self.path(method)
            if not np.array_equal(frame["split"].to_numpy(), expected_ids):
                raise ValueError(f"{path}: expected split IDs 0 through {count - 1}; use complete paired holdouts.")
            for column in ("train_index_hash", "test_index_hash"):
                if not frame[column].eq(reference[column]).all():
                    raise ValueError(f"{path}: {column} differs between methods for the same split; holdouts are not paired.")
            for column in ("inner_seed", "training_size", "test_size"):
                if column in frame and column in reference:
                    values = _numeric(frame, column, path)
                    expected = _numeric(reference, column, self.path(self.methods[0]))
                    if values.isna().any() or not values.eq(expected).all():
                        raise ValueError(f"{path}: {column} differs between methods for the same split.")
        return count

    def summary(self, method: ClinicalMethod, metric: str) -> tuple[float, float]:
        """Return a successful-fit mean and sample SD, excluding nonfinite values."""
        if metric not in METRICS:
            raise ValueError(f"Unknown clinical table metric: {metric!r}.")
        frame = self.frame(method)
        values = frame.loc[frame["status"].eq("ok") & np.isfinite(frame[metric]), metric]
        mean = float(values.mean()) if len(values) else float("nan")
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        return mean, sd

    def convergence(self, method: ClinicalMethod) -> tuple[int, int]:
        frame = self.frame(method)
        successful = frame.loc[frame["status"].eq("ok")]
        return int(successful["converged"].sum()), len(successful)

    def runtime(self, method: ClinicalMethod) -> tuple[float, float]:
        frame = self.frame(method)
        values = frame.loc[
            frame["status"].eq("ok") & np.isfinite(frame["runtime_seconds"]), "runtime_seconds",
        ]
        if values.empty:
            return float("nan"), float("nan")
        return float(values.mean()), float(values.max())


def _format_summary(values: tuple[float, float], digits: int) -> str:
    mean, spread = values
    if not np.isfinite(mean):
        return "--"
    detail = f"{spread:.{digits}f}" if np.isfinite(spread) else "--"
    return f"{mean:.{digits}f} ({detail})"


def render_performance_table(results: ClinicalResults, *, planned_splits: int) -> str:
    """Preserve the supplied table's labels, seven columns, and row ordering."""
    cohort = (
        "melanoma cohort." if results.dataset == "melanoma"
        else "breast cancer cohort\nfrom Paraíba State, Brazil."
    )
    lines = [
        r"\begin{table*}[tbp]",
        f"\\caption{{Predictive performance in {planned_splits} paired repeated holdouts of the {cohort}",
        r"Values for the IPCW integrated Brier score (IBS), cure-discrimination area under the curve",
        r"($\mathrm{AUC}_{\mathrm{cure}}$), and mean predicted cure fraction are means across successful",
        r"splits, with sample split standard deviations in parentheses. Convergence is reported as the number",
        r"of converged fits among successful fits. Runtime is the mean across successful fits with the maximum",
        r"in parentheses. Nonfinite measurements are excluded from each summary; a dash indicates an undefined value.",
        "}", f"\\label{{tab:{results.dataset}-performance}}", r"\centering",
        r"\begin{tabular}{llccccc}", r"\hline",
        r"Predictor & Count law & Converged & IPCW IBS &",
        r"$\mathrm{AUC}_{\mathrm{cure}}$ & Mean cure prediction & Runtime (s) \\",
        r"\hline", "",
    ]
    for index, method in enumerate(results.methods):
        if index in (2, 4):
            lines.extend((r"\hline", ""))
        converged, successful = results.convergence(method)
        lines.extend((
            method.predictor, f"& {method.count_law}", f"& {converged}/{successful}",
            f"& {_format_summary(results.summary(method, 'integrated_brier'), 4)}",
            f"& {_format_summary(results.summary(method, 'cure_auc'), 4)}",
            f"& {_format_summary(results.summary(method, 'mean_predicted_cure'), 3)}",
            f"& {_format_summary(results.runtime(method), 2)} " + r"\\", "",
        ))
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table*}", ""))
    return "\n".join(lines)
