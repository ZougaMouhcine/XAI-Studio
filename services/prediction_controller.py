"""
XAI Studio — Prediction Controller
==================================
UI-facing orchestration for prediction workflows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
import pandas as pd

from services.evaluation_service import ModelRegistry
from services.prediction_service import PredictionService, PredictionResult
from services.xai_service import XAIService
from services.pipeline_service import PipelineService
from utils.logger import get_logger

logger = get_logger(__name__)


class PredictionController:
    """Controller for prediction and local explanations."""

    def __init__(self, pipeline: PipelineService | None = None):
        self.pipeline = pipeline or PipelineService()
        self.registry = ModelRegistry()
        self.prediction_service = PredictionService()
        self.xai_service = XAIService()
        self._executor = ThreadPoolExecutor(max_workers=2)

    def refresh_registry(self):
        self.registry.clear()
        self._register_pipeline_models()
        self.registry.register_saved_models()
        return self.registry.list_entries()

    def register_external_model(self, filepath: str):
        return self.registry.register_external_model(filepath)

    def get_entry(self, entry_id: str):
        return self.registry.get(entry_id)

    def ensure_loaded(self, entry_id: str):
        entry = self.registry.get(entry_id)
        if not entry:
            raise RuntimeError("Model not found")
        return self.registry.ensure_loaded(entry)

    def predict_row(self, entry_id: str, row_values: list[Any]) -> PredictionResult:
        entry = self.ensure_loaded(entry_id)
        X = self.prediction_service.transform_row(row_values, entry.metadata or {})
        return self.prediction_service.predict(entry.model, X, entry.task_type)

    def predict_frame(self, entry_id: str, df: pd.DataFrame) -> PredictionResult:
        entry = self.ensure_loaded(entry_id)
        X = self.prediction_service.transform_frame(df, entry.metadata or {}).values
        return self.prediction_service.predict(entry.model, X, entry.task_type)

    def local_explanation(self, entry_id: str, instance: np.ndarray):
        entry = self.ensure_loaded(entry_id)
        feature_names = entry.metadata.get("feature_names", [])
        return self.xai_service.local_explanation(entry.model, instance, feature_names, entry.task_type)

    def shap_waterfall(self, entry_id: str, instance: np.ndarray):
        entry = self.ensure_loaded(entry_id)
        feature_names = entry.metadata.get("feature_names", [])
        X = np.array(instance).reshape(1, -1)
        return self.xai_service.shap_plot(entry.model, X, feature_names, plot_type="waterfall", instance_idx=0)

    def _register_pipeline_models(self):
        pr = self.pipeline.preprocessing_result
        dataset_name = None
        if self.pipeline.filepath:
            dataset_name = self.pipeline.filepath.split("\\")[-1]

        if self.pipeline.trained_models:
            for name, entry in self.pipeline.trained_models.items():
                model = entry.get("model")
                if model is None:
                    continue
                meta = {
                    "model_name": name,
                    "task_type": pr.task_type if pr else "unknown",
                    "feature_names": pr.feature_names if pr else [],
                    "input_feature_names": pr.input_feature_names if pr else [],
                    "numeric_feature_names": pr.numeric_feature_names if pr else [],
                    "categorical_feature_names": pr.categorical_feature_names if pr else [],
                    "feature_schema": pr.feature_schema if pr else [],
                    "target_columns": self.pipeline.target_columns,
                    "preprocessing_artifacts": {
                        "num_imputer": pr.encoders.get("num_imputer") if pr else None,
                        "cat_imputer": pr.encoders.get("cat_imputer") if pr else None,
                        "one_hot_encoder": pr.encoders.get("one_hot_encoder") if pr else None,
                        "scaler": pr.scaler if pr else None,
                    } if pr else {},
                }
                self.registry.register_in_memory(
                    name=name,
                    model=model,
                    metadata=meta,
                    task_type=meta["task_type"],
                    X_test=None,
                    y_test=None,
                    training_time=entry.get("training_time"),
                    dataset_name=dataset_name,
                )

        if self.pipeline.loaded_model is not None:
            meta = self.pipeline.loaded_model_metadata or {}
            task_type = meta.get("task_type", "unknown")
            self.registry.register_loaded_model(
                name="Uploaded Model",
                model=self.pipeline.loaded_model,
                metadata=meta,
                task_type=task_type,
                dataset_name=dataset_name,
            )

    def run_async(self, fn: Callable, on_done: Callable, on_error: Callable | None = None):
        def _callback(fut):
            try:
                result = fut.result()
                on_done(result)
            except Exception as exc:
                logger.error("Async error: %s", exc)
                if on_error:
                    on_error(exc)

        fut = self._executor.submit(fn)
        fut.add_done_callback(_callback)
        return fut
