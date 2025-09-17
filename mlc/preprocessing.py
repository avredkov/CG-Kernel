from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class PreprocessConfigCls(BaseModel):
    target_column: str = Field(default="cluster_label")
    numeric_columns: List[str] = Field(default_factory=list)
    log_columns: List[str] = Field(default_factory=list)
    categorical_column: str = Field(default="Regime")
    integer_columns: List[str] = Field(default_factory=list)

    @validator("log_columns", each_item=True)
    def _validate_log_columns(cls, v: str, values: Dict) -> str:  # type: ignore[override]
        if "numeric_columns" in values and v not in values["numeric_columns"]:
            raise ValueError(f"Log column '{v}' must be included in numeric_columns")
        return v


def safe_log10(x: pd.Series) -> pd.Series:
    if (x <= 0).any():
        bad = int((x <= 0).sum())
        raise ValueError(
            f"Log-transform requires positive values. Found {bad} non-positive entries in '{x.name}'."
        )
    return np.log10(x)


class DataPreprocessorCls:
    def __init__(self, config: Optional[PreprocessConfigCls] = None) -> None:
        self.config = config or PreprocessConfigCls()
        self.scaler: Optional[StandardScaler] = None
        self.regime_levels_: Optional[List[int]] = None
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[List[int]] = None
        self.label_to_index_: Optional[Dict[int, int]] = None

    def _build_continuous_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        cols = cfg.numeric_columns + cfg.integer_columns
        cont = df[cols].copy()
        for col in cfg.log_columns:
            cont[col] = safe_log10(cont[col])
        return cont

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        required = set([cfg.categorical_column] + cfg.numeric_columns + cfg.integer_columns)
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")
        df = df.copy()
        for col in cfg.numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in cfg.integer_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype(float)
        df[cfg.categorical_column] = pd.to_numeric(df[cfg.categorical_column], errors="coerce").astype(int)
        df = df.dropna(subset=list(required))
        return df

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, List[str]]:
        if self.scaler is None or self.regime_levels_ is None or self.feature_names_ is None:
            raise RuntimeError("DataPreprocessorCls must be fitted before calling transform().")
        df = self._validate_columns(df)
        cont_df = self._build_continuous_frame(df)
        cont_arr = self.scaler.transform(cont_df.values)
        regimes = df[self.config.categorical_column].astype(int)
        cat_arr = np.stack([(regimes == lvl).astype(float).values for lvl in self.regime_levels_], axis=1)
        X = np.concatenate([cont_arr, cat_arr], axis=1).astype(np.float32)
        y = np.zeros(len(df), dtype=np.int64)
        return X, y, regimes.reset_index(drop=True), (self.feature_names_ or [])


