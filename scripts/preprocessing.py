from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List, Optional, Union
from sklearn.preprocessing import (
    PowerTransformer,
    TargetEncoder,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from ggkm_paper.evaluate.metrics import uno_c_index_rmst, harrell_c_index
from skcure.utils.estimator import fit_cure_model


class MelanomaSurvivalPreprocessor:

    def __init__(
        self,
        time_col="time",
        status_col="delta",
        sex_col="sex",
        age_col="age",
        year_col="year",
        thickness_col="thickness",
        ulcer_col="ulcer",
    ):
        self.time_col = time_col
        self.status_col = status_col

        self.sex_col = sex_col
        self.age_col = age_col
        self.year_col = year_col
        self.thickness_col = thickness_col
        self.ulcer_col = ulcer_col

        self.feature_cols_ = [
            self.sex_col,
            self.age_col,
            self.year_col,
            self.thickness_col,
            self.ulcer_col,
        ]

        self.scaler_ = None

    def _validate(self, df):
        required = [
            self.time_col,
            self.status_col,
            self.sex_col,
            self.age_col,
            self.year_col,
            self.thickness_col,
            self.ulcer_col,
        ]

        missing_columns = [col for col in required if col not in df.columns]

        if missing_columns:
            raise ValueError(
                "Melanoma data are missing required columns: " f"{missing_columns}"
            )

        if df[required].isna().any().any():
            missing_counts = df[required].isna().sum()

            missing_counts = missing_counts[missing_counts > 0]

            raise ValueError(
                "Missing values were found in the melanoma data:\n" f"{missing_counts}"
            )

        thickness = df[self.thickness_col].to_numpy(dtype=float)

        if np.any(~np.isfinite(thickness)):
            raise ValueError("Tumour thickness contains non-finite values.")

        if np.any(thickness <= 0.0):
            raise ValueError(
                "Tumour thickness must be strictly positive "
                "because log(thickness) is used."
            )

        time = df[self.time_col].to_numpy(dtype=float)

        if np.any(~np.isfinite(time)):
            raise ValueError("Follow-up time contains non-finite values.")

        if np.any(time < 0.0):
            raise ValueError("Follow-up time cannot be negative.")

    def _make_features(self, df):

        self._validate(df)

        sex = df[self.sex_col].to_numpy(dtype=float)
        age = df[self.age_col].to_numpy(dtype=float)
        year = df[self.year_col].to_numpy(dtype=float)
        log_thickness = np.log(df[self.thickness_col].to_numpy(dtype=float))
        ulcer = df[self.ulcer_col].to_numpy(dtype=float)
        X = np.column_stack(
            [
                sex,
                age,
                year,
                log_thickness,
                ulcer,
            ]
        )

        return X

    def _make_outcome(self, df):

        time_days = df[self.time_col].to_numpy(dtype=float)
        delta = df[self.status_col].to_numpy(dtype=float)

        return (
            time_days / 365.25,
            delta,
        )

    def fit(self, df):

        X = self._make_features(df)
        t, delta = self._make_outcome(df)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        return (
            X_scaled,
            t,
            delta,
        )

    def transform(self, df):

        if self.scaler_ is None:
            raise RuntimeError("Call .fit(df) before .transform(df).")

        X = self._make_features(df)
        t, delta = self._make_outcome(df)
        X_scaled = self.scaler_.transform(X)

        return (
            X_scaled,
            t,
            delta,
        )


class FOSPMelanomaSurvivalPreprocessor:

    def __init__(
        self,
        followup="extended",
        include_treatments=True,
        include_extra_treatments=True,
        include_topography=True,
        include_morphology=True,
        min_morphology_frequency=50,
    ):

        if followup not in {
            "extended",
            "article",
        }:
            raise ValueError("followup must be either " "'extended' or 'article'.")

        if (
            not isinstance(
                min_morphology_frequency,
                (int, np.integer),
            )
            or min_morphology_frequency < 1
        ):
            raise ValueError("min_morphology_frequency must " "be a positive integer.")

        self.followup = followup
        self.include_treatments = include_treatments
        self.include_extra_treatments = include_extra_treatments
        self.include_topography = include_topography
        self.include_morphology = include_morphology
        self.min_morphology_frequency = int(min_morphology_frequency)

        if followup == "extended":

            self.time_col = "tempo_ext"
            self.status_col = "status_ext"

        else:

            self.time_col = "tempo"
            self.status_col = "status"

        self.continuous_cols_ = [
            "idade",
            "anodiag",
        ]

        self.binary_cols_ = [
            "female",
        ]

        if self.include_treatments:
            self.binary_cols_.extend(
                [
                    "cirurgia",
                    "radioterapia",
                    "quimioterapia",
                ]
            )

        if self.include_extra_treatments:

            self.binary_cols_.extend(
                [
                    "hormonio",
                    "imuno",
                ]
            )

        self.stage_col_ = "estadio"
        self.topography_col_ = "topo"
        self.morphology_col_ = "descmorfo"

        self.scaler_ = None

        self.stage_encoder_ = None
        self.topography_encoder_ = None
        self.morphology_encoder_ = None

        self.frequent_morphologies_ = None

        self.feature_names_ = None

    def _required_columns(self):

        required = (
            [
                self.time_col,
                self.status_col,
            ]
            + self.continuous_cols_
            + self.binary_cols_
            + [
                self.stage_col_,
            ]
        )

        if self.include_topography:
            required.append(self.topography_col_)

        if self.include_morphology:
            required.append(self.morphology_col_)

        return required

    def _validate(self, df):

        required = self._required_columns()

        missing_columns = [col for col in required if col not in df.columns]

        if missing_columns:

            raise ValueError(
                "FOSP melanoma data are missing " f"required columns: {missing_columns}"
            )

        missing_counts = df[required].isna().sum()

        missing_counts = missing_counts[missing_counts > 0]

        if len(missing_counts) > 0:

            raise ValueError(
                "Missing values were found in "
                "the FOSP melanoma data:\n"
                f"{missing_counts}"
            )

        for col in self.continuous_cols_:

            values = df[col].to_numpy(dtype=float)

            if np.any(~np.isfinite(values)):

                raise ValueError(f"{col} contains " "non-finite values.")

        age = df["idade"].to_numpy(dtype=float)

        if np.any(age <= 0.0):

            raise ValueError("Age must be strictly positive.")

        for col in self.binary_cols_:

            values = df[col].to_numpy()

            if not np.all(
                np.isin(
                    values,
                    [
                        0,
                        1,
                    ],
                )
            ):

                raise ValueError(f"{col} must contain " "only 0 and 1.")

        stage = df[self.stage_col_].astype(str)

        allowed_stages = {
            "I",
            "II",
            "III",
            "IV",
        }

        invalid_stage = ~stage.isin(allowed_stages)

        if invalid_stage.any():

            invalid_values = sorted(stage[invalid_stage].unique())

            raise ValueError(
                "Invalid melanoma stages found: "
                f"{invalid_values}. "
                "Expected I, II, III, or IV."
            )

        if self.include_topography:

            topography = df[self.topography_col_].astype(str)

            if topography.str.strip().eq("").any():

                raise ValueError("Topography contains " "empty categories.")

        if self.include_morphology:

            morphology = df[self.morphology_col_].astype(str)

            if morphology.str.strip().eq("").any():

                raise ValueError("Morphology contains " "empty categories.")

        time = df[self.time_col].to_numpy(dtype=float)

        if np.any(~np.isfinite(time)):

            raise ValueError("Follow-up time contains " "non-finite values.")

        if np.any(time <= 0.0):

            raise ValueError("Follow-up time must be " "strictly positive.")

        delta = df[self.status_col].to_numpy(dtype=float)

        if np.any(~np.isfinite(delta)):

            raise ValueError("Survival status contains " "non-finite values.")

        if not np.all(
            np.isin(
                delta,
                [
                    0.0,
                    1.0,
                ],
            )
        ):

            raise ValueError("Survival status must contain " "only 0 and 1.")

    def _continuous_features(
        self,
        df,
    ):

        return df[self.continuous_cols_].to_numpy(dtype=float)

    def _binary_features(
        self,
        df,
    ):

        return df[self.binary_cols_].to_numpy(dtype=float)

    def _stage_features(
        self,
        df,
    ):

        return df[
            [
                self.stage_col_,
            ]
        ].astype(str)

    def _topography_features(
        self,
        df,
    ):

        return df[
            [
                self.topography_col_,
            ]
        ].astype(str)

    def _fit_morphology_groups(
        self,
        df,
    ):
        """
        Determine frequent morphology categories using
        TRAINING DATA ONLY.

        Categories with frequency below
        min_morphology_frequency are mapped to "OTHER".
        """

        morphology = df[self.morphology_col_].astype(str)

        counts = morphology.value_counts()

        frequent = counts[counts >= self.min_morphology_frequency].index.tolist()

        self.frequent_morphologies_ = set(frequent)

    def _group_morphology(
        self,
        df,
    ):

        if self.frequent_morphologies_ is None:

            raise RuntimeError(
                "Morphology groups have not " "been fitted. Call .fit(df) first."
            )

        morphology = df[self.morphology_col_].astype(str).copy()

        morphology = morphology.where(
            morphology.isin(self.frequent_morphologies_),
            "OTHER",
        )

        return morphology.to_frame(name=self.morphology_col_)

    def _make_outcome(
        self,
        df,
    ):

        self._validate(df)

        # Time is already measured in YEARS.
        time = df[self.time_col].to_numpy(dtype=float)

        delta = df[self.status_col].to_numpy(dtype=float)

        return (
            time,
            delta,
        )

    def fit(
        self,
        df,
    ):

        self._validate(df)

        X_continuous = self._continuous_features(df)

        self.scaler_ = StandardScaler()

        X_continuous_scaled = self.scaler_.fit_transform(X_continuous)

        X_binary = self._binary_features(df)

        X_stage = self._stage_features(df)

        self.stage_encoder_ = OneHotEncoder(
            categories=[
                [
                    "I",
                    "II",
                    "III",
                    "IV",
                ]
            ],
            sparse_output=False,
            handle_unknown="error",
            dtype=float,
        )

        X_stage_encoded = self.stage_encoder_.fit_transform(X_stage)

        feature_blocks = [
            X_continuous_scaled,
            X_binary,
            X_stage_encoded,
        ]

        feature_names = (
            self.continuous_cols_
            + self.binary_cols_
            + [
                "estadio_I",
                "estadio_II",
                "estadio_III",
                "estadio_IV",
            ]
        )

        if self.include_topography:

            X_topography = self._topography_features(df)

            self.topography_encoder_ = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=float,
            )

            X_topography_encoded = self.topography_encoder_.fit_transform(X_topography)

            feature_blocks.append(X_topography_encoded)

            topography_names = self.topography_encoder_.get_feature_names_out(
                [
                    self.topography_col_,
                ]
            ).tolist()

            feature_names.extend(topography_names)

        if self.include_morphology:

            self._fit_morphology_groups(df)

            X_morphology = self._group_morphology(df)

            self.morphology_encoder_ = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=float,
            )

            X_morphology_encoded = self.morphology_encoder_.fit_transform(X_morphology)

            feature_blocks.append(X_morphology_encoded)

            morphology_names = self.morphology_encoder_.get_feature_names_out(
                [
                    self.morphology_col_,
                ]
            ).tolist()

            feature_names.extend(morphology_names)

        X = np.column_stack(feature_blocks)

        self.feature_names_ = feature_names

        t, delta = self._make_outcome(df)

        return (
            X,
            t,
            delta,
        )

    def transform(
        self,
        df,
    ):

        if self.scaler_ is None or self.stage_encoder_ is None:
            raise RuntimeError("Call .fit(df) before " ".transform(df).")

        if self.include_topography and self.topography_encoder_ is None:
            raise RuntimeError("Topography encoder has not " "been fitted.")

        if self.include_morphology and (
            self.morphology_encoder_ is None or self.frequent_morphologies_ is None
        ):
            raise RuntimeError("Morphology preprocessing has " "not been fitted.")

        self._validate(df)

        X_continuous = self._continuous_features(df)

        X_continuous_scaled = self.scaler_.transform(X_continuous)

        X_binary = self._binary_features(df)

        X_stage = self._stage_features(df)

        X_stage_encoded = self.stage_encoder_.transform(X_stage)

        feature_blocks = [
            X_continuous_scaled,
            X_binary,
            X_stage_encoded,
        ]

        if self.include_topography:

            X_topography = self._topography_features(df)

            X_topography_encoded = self.topography_encoder_.transform(X_topography)

            feature_blocks.append(X_topography_encoded)

        if self.include_morphology:

            X_morphology = self._group_morphology(df)

            X_morphology_encoded = self.morphology_encoder_.transform(X_morphology)

            feature_blocks.append(X_morphology_encoded)

        X = np.column_stack(feature_blocks)

        t, delta = self._make_outcome(df)

        return (
            X,
            t,
            delta,
        )

    def get_feature_names_out(
        self,
    ):

        if self.feature_names_ is None:

            raise RuntimeError("Call .fit(df) before " ".get_feature_names_out().")

        return np.asarray(
            self.feature_names_,
            dtype=object,
        )


class PassthroughPreprocessor:

    def __init__(self, time_col="time", status_col="status", feature_cols=None):
        self.time_col = time_col
        self.status_col = status_col
        self.feature_cols = feature_cols
        self.feature_cols_ = None

    def _split(self, df):
        X = df[self.feature_cols_].to_numpy()
        t = df[self.time_col].to_numpy().astype(float)
        delta = df[self.status_col].to_numpy().astype(float)
        return X, t, delta

    def fit(self, df):
        if self.feature_cols is not None:
            self.feature_cols_ = list(self.feature_cols)
        else:
            y_cols = [self.time_col, self.status_col]
            self.feature_cols_ = df.drop(columns=y_cols).columns.tolist()

        return self._split(df)

    def transform(self, df):
        if self.feature_cols_ is None:
            raise RuntimeError("Call .fit(df) before .transform(df).")

        return self._split(df)


class MissingIndicatorNumericEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: List[str],
    ):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):

        self.medians_ = {}

        for col in self.columns:
            if col not in X.columns:
                continue
            self.medians_[col] = X[col].median()

        return self

    def transform(self, X: pd.DataFrame):

        X = X.copy()

        for col in self.columns:
            if col not in X.columns:
                continue
            missing_name = f"{col}_missing"
            X[missing_name] = (X[col].isna()).astype(int)
            X[col] = X[col].fillna(self.medians_[col])

        return X


class StageTreatmentInteractionEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        stage_columns: List[str],
        treatment_columns: List[str],
    ):
        self.stage_columns = stage_columns
        self.treatment_columns = treatment_columns

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.copy()

        for stage in self.stage_columns:
            if stage not in X.columns:
                continue
            for treatment in self.treatment_columns:
                if treatment not in X.columns:
                    continue
                interaction_name = f"{stage}_{treatment}"
                X[interaction_name] = X[stage].astype(int) * X[treatment].astype(int)

        return X


class ContinuousFlagInteractionEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, continuous_column: str, flag_columns: List[str]):
        self.continuous_column = continuous_column
        self.flag_columns = flag_columns

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.continuous_column not in X.columns:
            return X
        X = X.copy()
        for flag in self.flag_columns:
            if flag not in X.columns:
                continue
            X[f"{self.continuous_column}_x_{flag}"] = X[self.continuous_column] * X[
                flag
            ].astype(int)
        return X


class OrdinalMapEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, mapping: Dict[str, float]):
        self.column = column
        self.mapping = mapping

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column not in X.columns:
            return X
        X = X.copy()
        X[self.column] = X[self.column].map(self.mapping)
        return X


class MissingFlagOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column: str,
        mapping: Dict[str, float],
        missing_category: str,
        missing_flag_name: str,
    ):
        self.column = column
        self.mapping = mapping
        self.missing_category = missing_category
        self.missing_flag_name = missing_flag_name

    def fit(self, X: pd.DataFrame, y=None):
        if self.column not in X.columns:
            self.median_ = None
            return self
        mapped = X[self.column].map(self.mapping)
        self.median_ = mapped.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column not in X.columns:
            return X
        X = X.copy()
        X[self.missing_flag_name] = (X[self.column] == self.missing_category).astype(
            int
        )
        X[self.column] = X[self.column].map(self.mapping)
        X[self.column] = X[self.column].fillna(self.median_)
        return X


class CategoryFlagEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, flags: Dict[str, str], drop_original: bool = True):
        self.column = column
        self.flags = flags
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column not in X.columns:
            return X
        X = X.copy()
        for new_col, category in self.flags.items():
            X[new_col] = (X[self.column] == category).astype(int)
        if self.drop_original:
            X = X.drop(columns=[self.column])
        return X


class CategoryTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column: str,
        output_column: Optional[str] = None,
        drop_original: bool = True,
        target_type: str = "auto",
        smooth: Union[str, float] = "auto",
        cv: int = 5,
        random_state: Optional[int] = None,
    ):
        self.column = column
        self.output_column = output_column
        self.drop_original = drop_original
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state

    def _out_col(self) -> str:
        return self.output_column or f"{self.column}_te"

    def fit(self, X: pd.DataFrame, y=None):
        if self.column not in X.columns:
            self.encoder_ = None
            return self
        if y is None:
            raise ValueError(
                "CategoryTargetEncoder requires y (the target) to fit. "
                "Pass it through pipeline.fit_transform(X, y)."
            )
        self.encoder_ = TargetEncoder(
            target_type=self.target_type,
            smooth=self.smooth,
            cv=self.cv,
            random_state=self.random_state,
        )
        self.encoder_.fit(X[[self.column]], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.column not in X.columns or getattr(self, "encoder_", None) is None:
            return X
        X = X.copy()
        encoded = self.encoder_.transform(X[[self.column]])
        X[self._out_col()] = encoded.ravel()
        if self.drop_original and self.column != self._out_col():
            X = X.drop(columns=[self.column])
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.column not in X.columns:
            self.encoder_ = None
            return X
        if y is None:
            raise ValueError(
                "CategoryTargetEncoder requires y (the target) to fit_transform. "
                "Pass it through pipeline.fit_transform(X, y)."
            )
        self.encoder_ = TargetEncoder(
            target_type=self.target_type,
            smooth=self.smooth,
            cv=self.cv,
            random_state=self.random_state,
        )
        encoded = self.encoder_.fit_transform(X[[self.column]], y)
        X = X.copy()
        X[self._out_col()] = encoded.ravel()
        if self.drop_original and self.column != self._out_col():
            X = X.drop(columns=[self.column])
        return X


class SurvivalTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column: str,
        time_column: str,
        event_column: str,
        output_column: Optional[str] = None,
        drop_original: bool = True,
        tau: Optional[float] = None,
        smooth: float = 10.0,
        cv: int = 5,
        random_state: Optional[int] = None,
    ):
        self.column = column
        self.time_column = time_column
        self.event_column = event_column
        self.output_column = output_column
        self.drop_original = drop_original
        self.tau = tau
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state

    def _out_col(self) -> str:
        return self.output_column or f"{self.column}_rmst"

    @staticmethod
    def _kaplan_meier(times: np.ndarray, events: np.ndarray):
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=float)

        unique_event_times = np.unique(times[events == 1])
        if unique_event_times.size == 0:
            return np.array([]), np.array([])

        survival = []
        s = 1.0
        for t in unique_event_times:
            n_at_risk = np.sum(times >= t)
            n_events = np.sum((times == t) & (events == 1))
            if n_at_risk > 0:
                s *= 1.0 - (n_events / n_at_risk)
            survival.append(s)
        return unique_event_times, np.array(survival)

    @staticmethod
    def _rmst(
        unique_event_times: np.ndarray, survival: np.ndarray, tau: float
    ) -> float:
        if tau <= 0:
            return 0.0
        if unique_event_times.size == 0:
            return float(tau)

        t_grid = np.concatenate(([0.0], unique_event_times))
        s_grid = np.concatenate(([1.0], survival))

        mask = t_grid <= tau
        t_grid = t_grid[mask]
        s_grid = s_grid[mask]

        if t_grid[-1] < tau:
            t_grid = np.concatenate([t_grid, [tau]])
            s_grid = np.concatenate([s_grid, [s_grid[-1]]])

        area = 0.0
        for i in range(len(t_grid) - 1):
            area += s_grid[i] * (t_grid[i + 1] - t_grid[i])
        return float(area)

    def _category_stats(self, df: pd.DataFrame, tau: float) -> Dict:
        stats = {}
        for category, group in df.groupby(self.column, dropna=False):
            u_times, surv = self._kaplan_meier(
                group[self.time_column].to_numpy(),
                group[self.event_column].to_numpy(),
            )
            stats[category] = {
                "rmst": self._rmst(u_times, surv, tau),
                "count": len(group),
            }
        return stats

    def _global_rmst(self, df: pd.DataFrame, tau: float) -> float:
        u_times, surv = self._kaplan_meier(
            df[self.time_column].to_numpy(), df[self.event_column].to_numpy()
        )
        return self._rmst(u_times, surv, tau)

    def _encode_from_stats(self, category, stats: Dict, global_rmst: float) -> float:
        cat_stats = stats.get(category)
        if cat_stats is None:
            return global_rmst
        n = cat_stats["count"]
        weight = n / (n + self.smooth)
        return weight * cat_stats["rmst"] + (1 - weight) * global_rmst

    def fit(self, X: pd.DataFrame, y=None):
        if self.column not in X.columns:
            self.category_stats_ = None
            return self
        tau = self.tau if self.tau is not None else X[self.time_column].max()
        self.tau_ = tau
        self.global_rmst_ = self._global_rmst(X, tau)
        self.category_stats_ = self._category_stats(X, tau)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if (
            self.column not in X.columns
            or getattr(self, "category_stats_", None) is None
        ):
            return X
        X = X.copy()
        X[self._out_col()] = X[self.column].map(
            lambda c: self._encode_from_stats(
                c, self.category_stats_, self.global_rmst_
            )
        )
        if self.drop_original and self.column != self._out_col():
            X = X.drop(columns=[self.column])
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.column not in X.columns:
            self.category_stats_ = None
            return X

        tau = self.tau if self.tau is not None else X[self.time_column].max()
        self.tau_ = tau

        self.global_rmst_ = self._global_rmst(X, tau)
        self.category_stats_ = self._category_stats(X, tau)

        encoded = np.empty(len(X), dtype=float)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for train_idx, holdout_idx in kf.split(np.arange(len(X))):
            fold_df = X.iloc[train_idx]
            fold_global = self._global_rmst(fold_df, tau)
            fold_stats = self._category_stats(fold_df, tau)

            holdout_categories = X.iloc[holdout_idx][self.column].to_numpy()
            encoded[holdout_idx] = [
                self._encode_from_stats(c, fold_stats, fold_global)
                for c in holdout_categories
            ]

        X = X.copy()
        X[self._out_col()] = encoded
        if self.drop_original and self.column != self._out_col():
            X = X.drop(columns=[self.column])
        return X


class CIndexForwardSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        candidate_columns: List[str],
        time_column: str,
        event_column: str,
        max_features: Optional[int] = None,
        min_improvement: float = 0.0,
        cv: int = 5,
        random_state: Optional[int] = None,
        log_path: Optional[str] = None,
    ):
        self.candidate_columns = candidate_columns
        self.time_column = time_column
        self.event_column = event_column
        self.max_features = max_features
        self.min_improvement = min_improvement
        self.cv = cv
        self.random_state = random_state
        self.log_path = log_path

    @staticmethod
    def _numeric_proxy(df: pd.DataFrame, col: str, event_col: str) -> np.ndarray:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(series.median()).to_numpy(dtype=float)
        means = df.groupby(col)[event_col].mean()
        global_mean = df[event_col].mean()
        return series.map(means).fillna(global_mean).to_numpy(dtype=float)

    def _oof_combined_score(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        n = len(df)
        combined = np.zeros(n, dtype=float)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        idx = np.arange(n)

        for col in columns:
            col_scores = np.empty(n, dtype=float)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            for train_idx, holdout_idx in kf.split(idx):
                train_df = df.iloc[train_idx]
                proxy_train = self._numeric_proxy(train_df, col, self.event_column)

                c_train = harrell_c_index(
                    proxy_train,
                    train_df[self.time_column].to_numpy(),
                    train_df[self.event_column].to_numpy(),
                )
                flip = (not np.isnan(c_train)) and c_train < 0.5

                mean_train = np.nanmean(proxy_train)
                std_train = np.nanstd(proxy_train)
                std_train = std_train if std_train > 0 else 1.0

                if is_numeric:
                    holdout_raw = (
                        df.iloc[holdout_idx][col]
                        .fillna(train_df[col].median())
                        .to_numpy(dtype=float)
                    )
                else:
                    means = train_df.groupby(col)[self.event_column].mean()
                    global_mean = train_df[self.event_column].mean()
                    holdout_raw = (
                        df.iloc[holdout_idx][col]
                        .map(means)
                        .fillna(global_mean)
                        .to_numpy(dtype=float)
                    )

                z = (holdout_raw - mean_train) / std_train
                if flip:
                    z = -z
                col_scores[holdout_idx] = z

            combined += col_scores

        return combined

    def fit(self, X: pd.DataFrame, y=None):
        df = X
        remaining = list(self.candidate_columns)
        selected: List[str] = []
        history = []

        max_features = self.max_features or len(remaining)
        time = df[self.time_column].to_numpy()
        event = df[self.event_column].to_numpy()

        best_score = -np.inf
        step = 0

        while remaining and len(selected) < max_features:
            step += 1
            step_results = []

            for col in remaining:
                trial_columns = selected + [col]
                combined = self._oof_combined_score(df, trial_columns)
                c_index = harrell_c_index(combined, time, event)
                step_results.append((col, c_index))

            step_results.sort(
                key=lambda item: (-np.inf if np.isnan(item[1]) else item[1]),
                reverse=True,
            )
            best_col, best_col_cindex = step_results[0]
            improvement = (
                best_col_cindex - best_score
                if not np.isnan(best_col_cindex)
                else -np.inf
            )
            accepted = improvement >= self.min_improvement

            history.append(
                {
                    "step": step,
                    "candidates_evaluated": [c for c, _ in step_results],
                    "candidates_c_index": [c_ for _, c_ in step_results],
                    "selected_feature": best_col,
                    "cumulative_features": selected + [best_col],
                    "cv_c_index": best_col_cindex,
                    "improvement": improvement,
                    "accepted": accepted,
                }
            )

            if self.log_path is not None:
                pd.DataFrame(history).to_csv(self.log_path, index=False)

            if not accepted:
                break

            selected.append(best_col)
            remaining.remove(best_col)
            best_score = best_col_cindex

        self.history_ = pd.DataFrame(history)
        self.selected_features_ = selected
        self.best_cv_c_index_ = best_score if selected else np.nan

        if self.log_path is not None:
            self.history_.to_csv(self.log_path, index=False)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def get_selected_features(self) -> List[str]:
        return list(self.selected_features_)


class BreastCancerSurvivalPreprocessor:
    TIME_COLUMN = "Tempo"
    STATUS_COLUMN = "delta"
    EVENT_LABEL = 1

    RAW_FEATURES: List[str] = [
        "Município",
        "RhcRacaCor",
        "RhcInstrucao",
        # "SimCausaBasicaCategoria",
        "RhcLOCTUPRI",
        "CosemsMacrorregiaoSaude",
        "Origem",
        "RhcOrigemEncamiamento",
        "RhcFonte",
        "RhcTipoHistológico",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumorIntevalo",
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumorIntevalo",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1ConsultaIntevalo",
        "PndrRenda",
        "FaixaETCAT",
        "RhcEstadiamentoClinico",
        "RhcIdadeNo1Consulta",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
        "RhcHistoricoTabaco",
        "RhcHistoricoFamiliarCancer",
        "RhcHistoricoAlcool",
        "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
        "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
        "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
    ]

    SCALED_FEATURES: List[str] = [
        "RhcIdadeNo1Consulta",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor_x_I_II",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor_x_III_IV",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor_x_RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor_x_RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
    ]

    TARGET_ENCODED_FEATURES: List[str] = [
        # "RhcTipoHistológico_te",
        "RhcTipoHistológico_rmst",
        "RhcLOCTUPRI_rmst",
        # "SimCausaBasicaCategoria_rmst",
    ]

    DUMMY_FEATURES: List[str] = [
        "nenhuma_instrucao",
        "fundamental_ou_medio",
        "nivel_superior",
        "instrucao_sem_informacao",
        "branca",
        "nao_branca",
        "ignorado_raca",
        "1_macro",
        "2_macro",
        "3_macro",
        "macro_regiao_sem_informacao",
        "origem_sus",
        "origem_nao_sus",
        "origem_nenhum",
        "sus",
        "origem_sem_informacao",
        "nao_sus",
        "veio_por_conta",
        "nao_se_aplica",
        "laureano",
        "hu_cg",
        "fap",
        "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor_missing",
        "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta_missing",
        "1tratamento_diagnostico_tumor_ate_90dias",
        "1tratamento_diagnostico_tumor_acima_90dias",
        "1tratamento_consulta_ate_90dias",
        "1tratamento_consulta_acima_90dias",
        "1consulta_diagnostico_tumor_ate_90dias",
        "1consulta_diagnostico_tumor_acima_90dias",
        "baixa_renda",
        "media_renda",
        "alta_renda",
        "renda_sem_informacao",
        "20_39",
        "40_49",
        "50_59",
        "60_69",
        "70_99",
        "I_II",
        "III_IV",
        "estadiamento_seminformacao",
        "Tabaco_Sim",
        "Tabaco_ExConsumidor",
        "Tabaco_SemInformacao",
        "HistFamCancer_Sim",
        "HistFamCancer_SemInformacao",
        "Alcool_Sim",
        "Alcool_ExConsumidor",
        "Alcool_SemInformacao",
        "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
        "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
        "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
        "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
        # Stage x treatment interactions
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
        "I_II_RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
        "III_IV_RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
        # NEW: Stage x hospital source interactions
        "I_II_laureano",
        "I_II_fap",
        "I_II_hu_cg",
        "III_IV_laureano",
        "III_IV_fap",
        "III_IV_hu_cg",
        # NEW: Stage x education level interactions
        "I_II_nenhuma_instrucao",
        "I_II_fundamental_ou_medio",
        "I_II_nivel_superior",
        "I_II_instrucao_sem_informacao",
        "III_IV_nenhuma_instrucao",
        "III_IV_fundamental_ou_medio",
        "III_IV_nivel_superior",
        "III_IV_instrucao_sem_informacao",
    ]

    def __init__(self, selected_raw_features: Optional[List[str]] = None):
        self.selected_raw_features = selected_raw_features
        self.feature_pipeline_ = Pipeline(
            steps=[
                (
                    "continuous_delay_missing",
                    MissingIndicatorNumericEncoder(
                        columns=[
                            "DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumor",
                            "DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1Consulta",
                        ]
                    ),
                ),
                (
                    "municipio",
                    CategoryFlagEncoder(
                        column="Município",
                        flags={
                            "joao_pessoa": "João Pessoa",
                            "campina_grande": "Campina Grande",
                            "outros_municipios": "Outros",
                        },
                    ),
                ),
                (
                    "raca",
                    CategoryFlagEncoder(
                        column="RhcRacaCor",
                        flags={
                            "branca": "Branca",
                            "nao_branca": "Não Branca",
                            "ignorado_raca": "Ignorado",
                        },
                    ),
                ),
                (
                    "instrucao",
                    CategoryFlagEncoder(
                        column="RhcInstrucao",
                        flags={
                            "nenhuma_instrucao": "Nenhuma",
                            "fundamental_ou_medio": "Fundamental ou Médio",
                            "nivel_superior": "Nível Superior",
                            "instrucao_sem_informacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "macro_regiao",
                    CategoryFlagEncoder(
                        column="CosemsMacrorregiaoSaude",
                        flags={
                            "1_macro": "1ª Macro",
                            "2_macro": "2ª Macro",
                            "3_macro": "3ª Macro",
                            "macro_regiao_sem_informacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "origem",
                    CategoryFlagEncoder(
                        column="Origem",
                        flags={
                            "origem_sus": "SUS",
                            "origem_nao_sus": "Não SUS",
                            "origem_nenhum": np.nan,
                        },
                    ),
                ),
                (
                    "origem_encaminhamento",
                    CategoryFlagEncoder(
                        column="RhcOrigemEncamiamento",
                        flags={
                            "sus": "SUS",
                            "origem_sem_informacao": "Sem Informação",
                            "nao_sus": "Não SUS",
                            "veio_por_conta": "Veio Por Conta Própria",
                            "nao_se_aplica": "Não se Aplica",
                        },
                    ),
                ),
                (
                    "fonte",
                    CategoryFlagEncoder(
                        column="RhcFonte",
                        flags={"laureano": "Laureano", "fap": "FAP", "hu_cg": "HU CG"},
                    ),
                ),
                # (
                #     "tipo_histologico",
                #     CategoryTargetEncoder(
                #         column="RhcTipoHistológico",
                #         output_column="RhcTipoHistológico_te",
                #         target_type="binary",
                #         smooth="auto",
                #         cv=5,
                #         random_state=42,
                #     ),
                # ),
                (
                    "classificacao_cancer",
                    SurvivalTargetEncoder(
                        column="RhcLOCTUPRI",
                        time_column="Tempo",
                        event_column="delta",
                        output_column="RhcLOCTUPRI_rmst",
                        smooth=10.0,
                        cv=5,
                        random_state=42,
                    ),
                ),
                (
                    "tipo_histologico",
                    SurvivalTargetEncoder(
                        column="RhcTipoHistológico",
                        time_column="Tempo",
                        event_column="delta",
                        output_column="RhcTipoHistológico_rmst",
                        smooth=10.0,
                        cv=5,
                        random_state=42,
                    ),
                ),
                (
                    "1consulta_diagnostico_tumor",
                    CategoryFlagEncoder(
                        column="DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumorIntevalo",
                        flags={
                            "1consulta_diagnostico_tumor_ate_90dias": "1. até 90 dias",
                            "1consulta_diagnostico_tumor_acima_90dias": "2. acima de 90 dias",
                        },
                    ),
                ),
                (
                    "1tratamento_consulta",
                    CategoryFlagEncoder(
                        column="DiffEmDiasEntreRhcDt1TratamentoTumorERhcDt1ConsultaIntevalo",
                        flags={
                            "1tratamento_consulta_ate_90dias": "1. até 90 dias",
                            "1tratamento_consulta_acima_90dias": "2. acima de 90 dias",
                        },
                    ),
                ),
                (
                    "1tratamento_diagnostico_tumor",
                    CategoryFlagEncoder(
                        column="DiffEmDiasEntreRhcDt1TratamentoTumorEDtDiagnosticoTumorIntevalo",
                        flags={
                            "1tratamento_diagnostico_tumor_ate_90dias": "1. até 90 dias",
                            "1tratamento_diagnostico_tumor_acima_90dias": "2. acima de 90 dias",
                        },
                    ),
                ),
                (
                    "renda",
                    CategoryFlagEncoder(
                        column="PndrRenda",
                        flags={
                            "baixa_renda": "Baixa Renda",
                            "media_renda": "Média Renda",
                            "alta_renda": "Alta Renda",
                            "renda_sem_informacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "faixa_etaria",
                    CategoryFlagEncoder(
                        column="FaixaETCAT",
                        flags={
                            "20_39": "20 a 39",
                            "40_49": "40 a 49",
                            "50_59": "50 a 59",
                            "60_69": "60 a 69",
                            "70_99": "70 a 99",
                        },
                    ),
                ),
                (
                    "estadiamento",
                    CategoryFlagEncoder(
                        column="RhcEstadiamentoClinico",
                        flags={
                            "I_II": "I e II",
                            "III_IV": "III e IV",
                            "estadiamento_seminformacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "stage_treatment_interactions",
                    StageTreatmentInteractionEncoder(
                        stage_columns=[
                            "I_II",
                            "III_IV",
                        ],
                        treatment_columns=[
                            "RhcPrimeiroTratamentoRecebidoNoHospitalNenhum",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalCirurgia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalRadioterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalQuimioterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalOutras",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalSemInformacao",
                        ],
                    ),
                ),
                (
                    "stage_fonte_interactions",
                    StageTreatmentInteractionEncoder(
                        stage_columns=["I_II", "III_IV"],
                        treatment_columns=["laureano", "fap", "hu_cg"],
                    ),
                ),
                (
                    "stage_instrucao_interactions",
                    StageTreatmentInteractionEncoder(
                        stage_columns=["I_II", "III_IV"],
                        treatment_columns=[
                            "nenhuma_instrucao",
                            "fundamental_ou_medio",
                            "nivel_superior",
                            "instrucao_sem_informacao",
                        ],
                    ),
                ),
                (
                    "diagnostic_delay_stage_interaction",
                    ContinuousFlagInteractionEncoder(
                        continuous_column="DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
                        flag_columns=["I_II", "III_IV"],
                    ),
                ),
                (
                    "diagnostic_delay_treatment_interaction",
                    ContinuousFlagInteractionEncoder(
                        continuous_column="DiffEmDiasEntreRhcDt1ConsultaEDtDiagnosticoTumor",
                        flag_columns=[
                            "RhcPrimeiroTratamentoRecebidoNoHospitalHormonioterapia",
                            "RhcPrimeiroTratamentoRecebidoNoHospitalImunoterapia",
                        ],
                    ),
                ),
                (
                    "tabaco",
                    CategoryFlagEncoder(
                        column="RhcHistoricoTabaco",
                        flags={
                            "Tabaco_Sim": "Sim",
                            "Tabaco_ExConsumidor": "Ex-consumidor",
                            "Tabaco_SemInformacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "hist_familiar",
                    CategoryFlagEncoder(
                        column="RhcHistoricoFamiliarCancer",
                        flags={
                            "HistFamCancer_Sim": "Sim",
                            "HistFamCancer_SemInformacao": "Sem Informação",
                        },
                    ),
                ),
                (
                    "alcool",
                    CategoryFlagEncoder(
                        column="RhcHistoricoAlcool",
                        flags={
                            "Alcool_Sim": "Sim",
                            "Alcool_ExConsumidor": "Ex-consumidor",
                            "Alcool_SemInformacao": "Sem Informação",
                        },
                    ),
                ),
            ]
        )

        self.df_model_: Optional[pd.DataFrame] = None
        self.column_transformer_: Optional[ColumnTransformer] = None

    def _build_delta(self, df: pd.DataFrame) -> pd.Series:
        return (df[self.STATUS_COLUMN] == self.EVENT_LABEL).astype(int)

    def _active_raw_features(self) -> List[str]:
        if self.selected_raw_features is None:
            return list(self.RAW_FEATURES)
        return [f for f in self.RAW_FEATURES if f in self.selected_raw_features]

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df["delta"] = self._build_delta(df)

        cols = self._active_raw_features() + [self.TIME_COLUMN, "delta"]
        df_model = df[cols].copy()
        y = df_model["delta"]

        df_model = self.feature_pipeline_.fit_transform(df_model, y=y)
        self.df_model_ = df_model

        t = df_model[self.TIME_COLUMN].values
        delta = df_model["delta"].values

        scaled = [c for c in self.SCALED_FEATURES if c in df_model.columns]
        target_encoded = [
            c for c in self.TARGET_ENCODED_FEATURES if c in df_model.columns
        ]
        dummy = [c for c in self.DUMMY_FEATURES if c in df_model.columns]
        self._active_scaled = scaled
        self._active_target_encoded = target_encoded
        self._active_dummy = dummy

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                (
                    "power",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                    scaled,
                ),
                ("target_encoded", "passthrough", target_encoded),
                ("dummy", "passthrough", dummy),
            ]
        )

        X = self.column_transformer_.fit_transform(df_model)
        X = np.asarray(X, dtype=float)

        return X, t, delta

    def transform(self, df: pd.DataFrame):
        if self.column_transformer_ is None:
            raise RuntimeError("Call .fit(df) before .transform(df).")

        df = df.copy()
        df["delta"] = self._build_delta(df)
        delta = df["delta"].values

        cols = self._active_raw_features() + [self.TIME_COLUMN, "delta"]
        df_model = df[cols].copy()
        df_model = self.feature_pipeline_.transform(df_model)

        t = df_model[self.TIME_COLUMN].values
        delta = df_model["delta"].values

        X = self.column_transformer_.transform(df_model)
        X = np.asarray(X, dtype=float)

        return X, t, delta

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df)


class SurvivalModelForwardSelector:
    def __init__(
        self,
        candidate_columns: List[str],
        model_factory: Callable[[], Any],
        preprocessor_cls: type = BreastCancerSurvivalPreprocessor,
        n_times: int = 100,
        validation_size: float = 0.3,
        random_state: Optional[int] = 42,
        max_features: Optional[int] = None,
        min_improvement: float = 0.0,
        log_path: Optional[str] = None,
        secondary_metric: Optional[Callable] = None,
    ):
        self.candidate_columns = candidate_columns
        self.model_factory = model_factory
        self.preprocessor_cls = preprocessor_cls
        self.n_times = n_times
        self.validation_size = validation_size
        self.random_state = random_state
        self.max_features = max_features
        self.min_improvement = min_improvement
        self.log_path = log_path
        self.secondary_metric = secondary_metric

    def _split(self, df: pd.DataFrame):
        delta = (
            df[self.preprocessor_cls.STATUS_COLUMN] == self.preprocessor_cls.EVENT_LABEL
        ).astype(int)
        train_idx, val_idx = train_test_split(
            np.arange(len(df)),
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=delta,
        )
        return train_idx, val_idx

    def _evaluate(self, df: pd.DataFrame, columns: List[str], train_idx, val_idx):
        preprocessor = self.preprocessor_cls(selected_raw_features=columns)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        X_train, t_train, delta_train = preprocessor.fit(df_train)
        X_val, t_val, delta_val = preprocessor.transform(df_val)

        model = self.model_factory()
        fit_cure_model(model, X_train, t_train, delta_train)

        times = np.linspace(np.min(t_train), np.max(t_train), self.n_times)
        S_pred = model.predict_survival(X_val, times)

        c_index = uno_c_index_rmst(
            S_pred, t_val, delta_val, t_train, delta_train, times
        )

        secondary = None
        if self.secondary_metric is not None:
            secondary = self.secondary_metric(
                S_pred, t_val, delta_val, t_train, delta_train, times
            )

        return c_index, secondary

    def fit(self, df: pd.DataFrame):
        train_idx, val_idx = self._split(df)

        remaining = list(self.candidate_columns)
        selected: List[str] = []
        history = []

        max_features = self.max_features or len(remaining)
        best_score = -np.inf
        step = 0

        while remaining and len(selected) < max_features:
            step += 1
            step_results = []

            for col in remaining:
                trial_columns = selected + [col]
                try:
                    c_index, secondary = self._evaluate(
                        df, trial_columns, train_idx, val_idx
                    )
                except Exception as e:
                    c_index, secondary = float("nan"), None
                step_results.append((col, c_index, secondary))

            step_results.sort(
                key=lambda item: (-np.inf if np.isnan(item[1]) else item[1]),
                reverse=True,
            )
            best_col, best_col_cindex, best_col_secondary = step_results[0]
            improvement = (
                best_col_cindex - best_score
                if not np.isnan(best_col_cindex)
                else -np.inf
            )
            accepted = improvement >= self.min_improvement

            history.append(
                {
                    "step": step,
                    "candidates_evaluated": [c for c, _, _ in step_results],
                    "candidates_c_index": [c for _, c, _ in step_results],
                    "candidates_secondary_metric": [s for _, _, s in step_results],
                    "selected_feature": best_col,
                    "cumulative_features": selected + [best_col],
                    "val_c_index": best_col_cindex,
                    "val_secondary_metric": best_col_secondary,
                    "improvement": improvement,
                    "accepted": accepted,
                }
            )

            if self.log_path is not None:
                pd.DataFrame(history).to_csv(self.log_path, index=False)

            if not accepted:
                break

            selected.append(best_col)
            remaining.remove(best_col)
            best_score = best_col_cindex

        self.history_ = pd.DataFrame(history)
        self.selected_features_ = selected
        self.best_val_c_index_ = best_score if selected else float("nan")

        if self.log_path is not None:
            self.history_.to_csv(self.log_path, index=False)

        return self

    def get_selected_features(self) -> List[str]:
        return list(self.selected_features_)


class DefaultSurvivalPreprocessor:
    TIME_COLUMN = "t"
    STATUS_COLUMN = "delta"
    EVENT_LABEL = 1

    RAW_FEATURES: List[str] = [
        "classic_fico",
        "first_time_homebuyer_indicator",
        "mortgage_insurance_percentage(mi%)",
        "original_combined_loan_to_value_(cltv)",
        "original_debt_to_income_(dti)_ratio",
        "original_upb",
        "original_loan_to_value(ltv)",
        "original_interest_rate",
        "channel",
        "property_type",
        "postal_code",
        "original_loan_term",
        "number_of_borrowers",
        "seller_name",
        "harp_indicator",
    ]

    SCALED_FEATURES: List[str] = [
        "classic_fico",
        "mortgage_insurance_percentage(mi%)",
        "original_combined_loan_to_value_(cltv)",
        "original_debt_to_income_(dti)_ratio",
        "original_upb",
        "original_loan_to_value(ltv)",
        "original_interest_rate",
        "original_loan_term",
        "number_of_borrowers",
    ]

    TARGET_ENCODED_FEATURES: List[str] = [
        "postal_code",
    ]

    DUMMY_FEATURES: List[str] = [
        "channel_r",
        "channel_c",
        "channel_b",
    ]

    def __init__(self, selected_raw_features: Optional[List[str]] = None):
        self.selected_raw_features = selected_raw_features
        self.feature_pipeline_ = Pipeline(
            steps=[
                (
                    "channel",
                    CategoryFlagEncoder(
                        column="channel",
                        flags={
                            "channel_r": "R",
                            "channel_c": "C",
                            "channel_b": "B",
                        },
                    ),
                ),
                (
                    "property_type",
                    SurvivalTargetEncoder(
                        column="property_type",
                        time_column="t",
                        event_column="delta",
                        output_column="property_type_rmst",
                        smooth=10.0,
                        cv=5,
                        random_state=42,
                    ),
                ),
                (
                    "seller_name",
                    SurvivalTargetEncoder(
                        column="seller_name",
                        time_column="t",
                        event_column="delta",
                        output_column="seller_name_rmst",
                        smooth=10.0,
                        cv=5,
                        random_state=42,
                    ),
                ),
                (
                    "postal_code",
                    SurvivalTargetEncoder(
                        column="postal_code",
                        time_column="t",
                        event_column="delta",
                        output_column="postal_code_rmst",
                        smooth=10.0,
                        cv=5,
                        random_state=42,
                    ),
                ),
            ]
        )

        self.df_model_: Optional[pd.DataFrame] = None
        self.column_transformer_: Optional[ColumnTransformer] = None

    def _build_delta(self, df: pd.DataFrame) -> pd.Series:
        return (df[self.STATUS_COLUMN] == self.EVENT_LABEL).astype(int)

    def _active_raw_features(self) -> List[str]:
        if self.selected_raw_features is None:
            return list(self.RAW_FEATURES)
        return [f for f in self.RAW_FEATURES if f in self.selected_raw_features]

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df["delta"] = self._build_delta(df)

        cols = self._active_raw_features() + [self.TIME_COLUMN, "delta"]
        df_model = df[cols].copy()
        y = df_model["delta"]

        df_model = self.feature_pipeline_.fit_transform(df_model, y=y)
        self.df_model_ = df_model

        t = df_model[self.TIME_COLUMN].values
        delta = df_model["delta"].values

        scaled = [c for c in self.SCALED_FEATURES if c in df_model.columns]
        target_encoded = [
            c for c in self.TARGET_ENCODED_FEATURES if c in df_model.columns
        ]
        dummy = [c for c in self.DUMMY_FEATURES if c in df_model.columns]
        self._active_scaled = scaled
        self._active_target_encoded = target_encoded
        self._active_dummy = dummy

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                (
                    "power",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                    scaled,
                ),
                ("target_encoded", "passthrough", target_encoded),
                ("dummy", "passthrough", dummy),
            ]
        )

        X = self.column_transformer_.fit_transform(df_model)
        X = np.asarray(X, dtype=float)

        return X, t, delta

    def transform(self, df: pd.DataFrame):
        if self.column_transformer_ is None:
            raise RuntimeError("Call .fit(df) before .transform(df).")

        df = df.copy()
        df["delta"] = self._build_delta(df)
        delta = df["delta"].values

        cols = self._active_raw_features() + [self.TIME_COLUMN, "delta"]
        df_model = df[cols].copy()
        df_model = self.feature_pipeline_.transform(df_model)

        t = df_model[self.TIME_COLUMN].values
        delta = df_model["delta"].values

        X = self.column_transformer_.transform(df_model)
        X = np.asarray(X, dtype=float)

        return X, t, delta

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df)
