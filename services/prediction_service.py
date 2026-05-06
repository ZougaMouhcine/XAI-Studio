"""
XAI Studio — Prediction Service
===============================
Run inference on new data and format prediction outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    predictions: np.ndarray
    probabilities: np.ndarray | None
    confidence: np.ndarray | None


class PredictionService:
    """Prediction engine for single and batch inference."""

    def validate_features(self, feature_names: list[str], df: pd.DataFrame) -> tuple[list[str], list[str]]:
        missing = [f for f in feature_names if f not in df.columns]
        extra = [c for c in df.columns if c not in feature_names]
        return missing, extra

    def prepare_frame(self, df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        return df[feature_names]

    def transform_frame(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        feature_names = metadata.get("feature_names", []) or list(df.columns)
        input_feature_names = metadata.get("input_feature_names", []) or list(df.columns)
        artifacts = metadata.get("preprocessing_artifacts", {}) or {}
        numeric_feature_names = metadata.get("numeric_feature_names", []) or []
        categorical_feature_names = metadata.get("categorical_feature_names", []) or []

        work = df.copy()
        missing = [name for name in input_feature_names if name not in work.columns]
        if missing:
            raise RuntimeError(f"Missing input columns: {', '.join(missing)}")

        work = work.reindex(columns=input_feature_names)

        num_imputer = artifacts.get("num_imputer")
        if num_imputer is not None and numeric_feature_names:
            work[numeric_feature_names] = num_imputer.transform(work[numeric_feature_names])

        cat_imputer = artifacts.get("cat_imputer")
        if cat_imputer is not None and categorical_feature_names:
            work[categorical_feature_names] = cat_imputer.transform(work[categorical_feature_names])

        one_hot_encoder = artifacts.get("one_hot_encoder")
        if one_hot_encoder is not None and categorical_feature_names:
            encoded = one_hot_encoder.transform(work[categorical_feature_names].astype(str))
            encoded_cols = list(one_hot_encoder.get_feature_names_out(categorical_feature_names))
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=work.index)
            work = pd.concat([work.drop(columns=categorical_feature_names), encoded_df], axis=1)

        scaler = artifacts.get("scaler")
        if scaler is not None:
            ordered = work.reindex(columns=feature_names)
            scaled = scaler.transform(ordered)
            return pd.DataFrame(scaled, columns=feature_names, index=work.index)

        return work.reindex(columns=feature_names)

    def transform_row(self, row_values: list[Any], metadata: dict) -> np.ndarray:
        input_names = metadata.get("input_feature_names", []) or metadata.get("feature_names", [])
        if not input_names:
            return np.array(row_values, dtype=object).reshape(1, -1)
        df = pd.DataFrame([row_values], columns=input_names)
        transformed = self.transform_frame(df, metadata)
        return transformed.values

    def predict(self, model, X: np.ndarray, task_type: str) -> PredictionResult:
        preds = model.predict(X)
        proba = None
        conf = None
        if task_type == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                conf = np.max(proba, axis=1)
            except Exception as exc:
                logger.warning("predict_proba failed: %s", exc)
                proba = None
                conf = None
        return PredictionResult(predictions=np.array(preds), probabilities=proba, confidence=conf)

    def predict_row(self, model, row_values: list[Any], task_type: str) -> PredictionResult:
        X = np.array(row_values, dtype=object).reshape(1, -1)
        return self.predict(model, X, task_type)
