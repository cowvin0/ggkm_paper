"""Read PCM replication files and compute the summaries used by paper tables.

This module deliberately does not import the experiment runner: generating tables
only requires NumPy and pandas, without loading or fitting any estimators.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Method:
    model: str
    kernel: str


@dataclass(frozen=True)
class Scenario:
    method: int
    generating: str
    n: int


KERNELS = (
    "linear", "rbf", "laplacian", "exponential", "cauchy", "polynomial", "wavelet",
)
METRICS = ("test_ibs", "cure_rmse", "test_auc")
POISSON_METHODS = tuple(Method("poisson", kernel) for kernel in KERNELS) + (
    Method("spline_poisson", "parametric"),
    Method("promotion_time", "parametric"),
    Method("mixture_cure", "parametric"),
)
NB_METHODS = tuple(Method("bn", kernel) for kernel in KERNELS) + (
    Method("spline_bn", "parametric"),
)
ALL_METHODS = POISSON_METHODS + NB_METHODS
PRIMARY_SCENARIOS = tuple(
    Scenario(method, generating, n)
    for method in (1, 2, 3)
    for n in (300, 600)
    for generating in ("poisson", "nb025", "nb100")
)
HIGHDIM_SCENARIOS = tuple(
    Scenario(method, generating, n)
    for method in (4, 5)
    for n in (300, 600)
    for generating in ("poisson", "nb025", "nb100")
)

_FILE_PATTERN = re.compile(
    r"^replications_task\d+_(?P<model>.+?)_n(?P<n>\d+)_"
    r"method(?P<method>\d+)_gen(?P<generating>poisson|nb025|nb100)_"
    r"kernel(?P<kernel>[^/]+)\.csv$"
)
_REQUIRED_COLUMNS = frozenset(("replication", "status", *METRICS))
_READ_COLUMNS = _REQUIRED_COLUMNS | {
    "converged", "n", "method", "model", "kernel",
    "generating_distribution", "generating_phi",
    "generation_seed", "split_seed", "inner_seed",
    "runtime_seconds", "fitted_a", "fitted_d", "fitted_p", "fitted_phi",
}
_GENERATING_PARAMETERS = {
    "poisson": ("poisson", None),
    "nb025": ("negative_binomial", 0.25),
    "nb100": ("negative_binomial", 1.0),
}


def _describe(scenario: Scenario, method: Method) -> str:
    return (
        f"method={scenario.method}, generating={scenario.generating}, "
        f"n={scenario.n}, model={method.model}, kernel={method.kernel}"
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
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "1.0"):
            return True
        if normalized in ("false", "0", "0.0"):
            return False
    raise ValueError(f"{path}: invalid convergence value {value!r}; expected true/false or 1/0.")


class Results:
    """Index PCM output files and lazily validate each requested configuration."""

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir).expanduser().resolve()
        if not self.results_dir.is_dir():
            raise FileNotFoundError(
                f"PCM results directory does not exist: {self.results_dir}. "
                "Run this command on the cluster, or supply --results-dir."
            )
        self._paths: dict[tuple[Scenario, Method], Path] = {}
        self._frames: dict[tuple[Scenario, Method], pd.DataFrame] = {}
        for path in sorted(self.results_dir.glob("replications_*.csv")):
            match = _FILE_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            fields = match.groupdict()
            method = Method(fields["model"], fields["kernel"])
            if method not in ALL_METHODS:
                continue
            scenario = Scenario(int(fields["method"]), fields["generating"], int(fields["n"]))
            key = (scenario, method)
            if key in self._paths:
                raise ValueError(
                    f"Duplicate PCM replication files for {_describe(*key)}:\n"
                    f"{self._paths[key]}\n{path}"
                )
            self._paths[key] = path
        if not self._paths:
            raise FileNotFoundError(
                f"No matching PCM replications_*.csv files found in {self.results_dir}. "
                "Use the replication CSVs saved by run_pcm_experiment.py."
            )

    def path(self, scenario: Scenario, method: Method) -> Path:
        try:
            return self._paths[(scenario, method)]
        except KeyError as error:
            raise FileNotFoundError(
                f"Missing PCM replication file for {_describe(scenario, method)} "
                f"in {self.results_dir}."
            ) from error

    def validate(self, scenarios: Iterable[Scenario], methods: Iterable[Method]) -> None:
        """Require and read every selected configuration before writing tables."""
        methods = tuple(methods)
        for scenario in scenarios:
            for method in methods:
                self.frame(scenario, method)

    def frame(self, scenario: Scenario, method: Method) -> pd.DataFrame:
        key = (scenario, method)
        if key not in self._frames:
            path = self.path(scenario, method)
            try:
                with path.open(newline="", encoding="utf-8-sig") as source:
                    header = next(csv.reader(source), [])
                if len(header) != len(set(header)):
                    raise ValueError(f"{path}: duplicate CSV column names.")
                # ROC arrays and parameter/error JSON can dominate the raw CSV;
                # retain only columns that tables or paired comparisons use.
                frame = pd.read_csv(path, usecols=lambda column: column in _READ_COLUMNS)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
                raise ValueError(f"Cannot read PCM replication CSV {path}: {error}") from error
            missing = _REQUIRED_COLUMNS.difference(frame.columns)
            if missing:
                raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}.")
            if frame.empty:
                raise ValueError(f"{path}: replication CSV has no rows.")

            replication = _numeric(frame, "replication", path)
            if not np.isfinite(replication).all() or (replication < 0).any() or (replication % 1 != 0).any():
                raise ValueError(f"{path}: replication IDs must be nonnegative integers without missing values.")
            frame["replication"] = replication.astype(np.int64)
            if frame["replication"].duplicated().any():
                raise ValueError(f"{path}: duplicate replication IDs.")
            status = frame["status"].astype("string").str.strip().str.lower()
            if not status.isin(("ok", "failed")).all():
                raise ValueError(f"{path}: status must be explicitly 'ok' or 'failed' on every row.")
            frame["status"] = status
            for column in METRICS:
                frame[column] = _numeric(frame, column, path)
            if "converged" in frame:
                frame["converged"] = pd.array(
                    [_convergence(value, path) for value in frame["converged"]], dtype="boolean",
                )
            self._validate_metadata(frame, scenario, method, path)
            self._frames[key] = frame
        return self._frames[key]

    @staticmethod
    def _validate_metadata(frame: pd.DataFrame, scenario: Scenario, method: Method, path: Path) -> None:
        for column, expected in (("n", scenario.n), ("method", scenario.method)):
            if column in frame:
                if not _numeric(frame, column, path).eq(expected).all():
                    raise ValueError(f"{path}: {column!r} metadata does not match the filename ({expected}).")
        distribution, phi = _GENERATING_PARAMETERS[scenario.generating]
        for column, expected in (("model", method.model), ("generating_distribution", distribution)):
            if column in frame and not frame[column].astype("string").eq(expected).fillna(False).all():
                raise ValueError(f"{path}: {column!r} metadata does not match the filename ({expected!r}).")
        if "kernel" in frame:
            kernel = frame["kernel"].astype("string").fillna("parametric")
            if not kernel.eq(method.kernel).all():
                raise ValueError(f"{path}: 'kernel' metadata does not match the filename ({method.kernel!r}).")
        if "generating_phi" in frame:
            values = _numeric(frame, "generating_phi", path)
            matches = values.isna() if phi is None else np.isclose(values, phi, rtol=1e-10, atol=0)
            if not np.all(matches):
                raise ValueError(f"{path}: 'generating_phi' metadata does not match the filename ({phi!r}).")

    def metric(self, scenario: Scenario, method: Method, metric: str) -> pd.Series:
        """Return successful finite measurements, indexed by replication for pairing."""
        if metric not in METRICS:
            raise ValueError(f"Unknown table metric {metric!r}; expected one of {METRICS}.")
        frame = self.frame(scenario, method)
        keep = frame["status"].eq("ok") & np.isfinite(frame[metric])
        return frame.loc[keep].set_index("replication")[metric].astype(float).sort_index()

    def summary(self, scenario: Scenario, method: Method, metric: str) -> tuple[float, float, int]:
        """Compute a Monte Carlo mean and its standard error using finite successes."""
        values = self.metric(scenario, method, metric)
        count = len(values)
        mean = float(values.mean()) if count else float("nan")
        mcse = float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else float("nan")
        return mean, mcse, count

    def diagnostics(
        self,
        scenarios: Iterable[Scenario],
        method: Method,
        *,
        latency_lower_bound: float = 1e-5,
        phi_lower_bound: float = 1e-8,
    ) -> dict[str, int | float | None]:
        """Aggregate fit status, lower-bound contacts, and runtimes over scenarios.

        Missing successful-fit diagnostics raise an error instead of being counted
        as nonconvergence or as absence of bound contact. Runtime includes failures.
        """
        if not np.isfinite(latency_lower_bound) or latency_lower_bound <= 0:
            raise ValueError("latency_lower_bound must be finite and positive.")
        if not np.isfinite(phi_lower_bound) or phi_lower_bound <= 0:
            raise ValueError("phi_lower_bound must be finite and positive.")
        counts = {"converged": 0, "nonconverged": 0, "failed": 0, "total": 0, "latency_bound": 0}
        negative_binomial = method.model in ("bn", "spline_bn")
        phi_contacts = 0
        runtimes = []
        for scenario in scenarios:
            frame = self.frame(scenario, method)
            path = self.path(scenario, method)
            columns = {"converged", "runtime_seconds", "fitted_a", "fitted_d", "fitted_p"}
            if negative_binomial:
                columns.add("fitted_phi")
            missing = columns.difference(frame.columns)
            if missing:
                raise ValueError(f"{path}: missing diagnostic columns: {', '.join(sorted(missing))}.")
            successful = frame.loc[frame["status"].eq("ok")]
            missing_convergence = int(successful["converged"].isna().sum())
            if missing_convergence:
                raise ValueError(f"{path}: {missing_convergence} successful fits have missing convergence diagnostics.")
            runtime = _numeric(frame, "runtime_seconds", path)
            invalid_runtime = int((~np.isfinite(runtime) | (runtime < 0)).sum())
            if invalid_runtime:
                raise ValueError(f"{path}: {invalid_runtime} fits have missing or invalid runtime_seconds.")
            runtimes.extend(runtime.to_numpy(dtype=float))
            latency_values = np.column_stack([
                _numeric(successful, column, path).to_numpy(dtype=float)
                for column in ("fitted_a", "fitted_d", "fitted_p")
            ])
            missing_latency = int((~np.isfinite(latency_values) | (latency_values <= 0)).any(axis=1).sum())
            if missing_latency:
                raise ValueError(f"{path}: {missing_latency} successful fits have missing or invalid latency parameters.")
            counts["latency_bound"] += int(
                np.isclose(latency_values, latency_lower_bound, rtol=1e-6, atol=0).any(axis=1).sum()
            )
            if negative_binomial:
                phi = _numeric(successful, "fitted_phi", path).to_numpy(dtype=float)
                missing_phi = int((~np.isfinite(phi) | (phi <= 0)).sum())
                if missing_phi:
                    raise ValueError(f"{path}: {missing_phi} successful fits have missing or invalid fitted_phi.")
                phi_contacts += int(np.isclose(phi, phi_lower_bound, rtol=1e-6, atol=0).sum())
            converged = int(successful["converged"].sum())
            counts["converged"] += converged
            counts["nonconverged"] += len(successful) - converged
            counts["failed"] += int(frame["status"].eq("failed").sum())
            counts["total"] += len(frame)
        if not counts["total"]:
            raise ValueError("Cannot summarize diagnostics without any scenarios.")
        return {
            **counts,
            "phi_bound": phi_contacts if negative_binomial else None,
            "runtime_mean": float(np.mean(runtimes)),
            "runtime_max": float(np.max(runtimes)),
        }
