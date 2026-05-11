"""
XAI Studio — Evaluation Controller
==================================
UI-facing orchestration layer for evaluation & XAI workflows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from services.evaluation_service import EvaluationService, ModelRegistry
from services.xai_service import XAIService
from services.visualization_service import VisualizationService
from services.report_service import ReportService
from services.pipeline_service import PipelineService
from utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationController:
    """Controller that connects UI to evaluation, XAI, and visualization services."""

    def __init__(self, pipeline: PipelineService | None = None):
        self.pipeline = pipeline or PipelineService()
        self.registry = ModelRegistry()
        self.evaluation_service = EvaluationService(self.registry)
        self.visualization_service = VisualizationService()
        self.xai_service = XAIService(self.visualization_service)
        self.report_service = ReportService()
        self._executor = ThreadPoolExecutor(max_workers=2)

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------
    def refresh_registry(self):
        self.registry.clear()
        self._register_pipeline_models()
        self.registry.register_saved_models()
        return self.registry.list_entries()

    def register_external_model(self, filepath: str):
        return self.registry.register_external_model(filepath)

    def attach_artifacts(self, entry_id: str, X_test=None, y_test=None, y_pred=None, y_proba=None):
        self.registry.attach_artifacts(entry_id, X_test=X_test, y_test=y_test, y_pred=y_pred, y_proba=y_proba)

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
                    "target_columns": self.pipeline.target_columns,
                }
                self.registry.register_in_memory(
                    name=name,
                    model=model,
                    metadata=meta,
                    task_type=meta["task_type"],
                    X_test=pr.X_test if pr else None,
                    y_test=pr.y_test if pr else None,
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

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_models(self, entry_ids: list[str]) -> dict[str, dict]:
        return self.evaluation_service.evaluate_models(entry_ids)

    def comparison_table(self, entry_ids: list[str]):
        return self.evaluation_service.comparison_table(entry_ids)

    # ------------------------------------------------------------------
    # XAI
    # ------------------------------------------------------------------
    def feature_importance(self, entry_id: str):
        entry = self.registry.get(entry_id)
        if not entry:
            raise RuntimeError("Model not found")
        entry = self.registry.ensure_loaded(entry)
        feature_names = entry.metadata.get("feature_names", [])
        return self.xai_service.feature_importance(entry.model, entry.X_test, entry.y_test, feature_names)

    def shap_plot(self, entry_id: str, plot_type: str, instance_idx: int = 0):
        entry = self.registry.get(entry_id)
        if not entry:
            raise RuntimeError("Model not found")
        entry = self.registry.ensure_loaded(entry)
        feature_names = entry.metadata.get("feature_names", [])
        X_data = entry.X_test
        if X_data is None:
            raise RuntimeError("X_test not available")
        X_sample = X_data[:200] if len(X_data) > 200 else X_data
        return self.xai_service.shap_plot(entry.model, X_sample, feature_names, plot_type, instance_idx)

    def lime_plot(self, entry_id: str, instance_idx: int = 0, num_features: int = 10):
        entry = self.registry.get(entry_id)
        if not entry:
            raise RuntimeError("Model not found")
        entry = self.registry.ensure_loaded(entry)
        pr = self.pipeline.preprocessing_result
        X_train = pr.X_train if pr else entry.X_test
        if X_train is None or entry.X_test is None:
            raise RuntimeError("Training data not available for LIME")
        feature_names = entry.metadata.get("feature_names", [])
        return self.xai_service.lime_plot(entry.model, X_train, entry.X_test, feature_names, entry.task_type, instance_idx, num_features)

    def pdp_plot(self, entry_id: str, feature_idxs, feature_names=None):
        entry = self.registry.get(entry_id)
        if not entry:
            raise RuntimeError("Model not found")
        entry = self.registry.ensure_loaded(entry)
        X = entry.X_test
        if X is None:
            raise RuntimeError("X_test not available")
        return self.xai_service.pdp_plot(entry.model, X, feature_idxs, feature_names=feature_names)

    def local_explanation(self, entry_id: str, instance):
        entry = self.registry.get(entry_id)
        if not entry:
            raise RuntimeError("Model not found")
        entry = self.registry.ensure_loaded(entry)
        feature_names = entry.metadata.get("feature_names", [])
        return self.xai_service.local_explanation(entry.model, instance, feature_names, entry.task_type)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def generate_report(self, entry_id: str, output_path: str, fmt: str = "html"):
        entry = self.registry.get(entry_id)
        if not entry:
            raise RuntimeError("Model not found")
        entry = self.registry.ensure_loaded(entry)

        metrics = entry.evaluation or self.evaluation_service.evaluate_entry(entry)
        if "error" in metrics:
            raise RuntimeError(metrics["error"])

        summary = {
            "Model": entry.name,
            "Task": entry.task_type,
            "Score": f"{metrics.get('score_global', '—')}",
            "Training Time": entry.training_time or "—",
            "Complexity": metrics.get("complexity_label", "—"),
        }

        sections = []
        if entry.task_type == "classification":
            cm = metrics.get("confusion_matrix")
            roc = metrics.get("roc_curve")
            pr = metrics.get("pr_curve")
            if cm is not None:
                sections.append(self._section("Confusion Matrix", self.visualization_service.render_confusion_matrix(cm).figure))
            if roc:
                sections.append(self._section("ROC Curve", self.visualization_service.render_roc_curve(roc["fpr"], roc["tpr"], label=entry.name).figure))
            if pr:
                sections.append(self._section("Precision-Recall", self.visualization_service.render_precision_recall_curve(pr["recall"], pr["precision"], label=entry.name).figure))
        elif entry.task_type == "regression":
            y_true = entry.y_test
            y_pred = metrics.get("y_pred")
            if y_true is not None and y_pred is not None:
                sections.append(self._section("Predicted vs Actual", self.visualization_service.render_predicted_vs_actual(y_true, y_pred).figure))
                sections.append(self._section("Residuals", self.visualization_service.render_residuals(y_true, y_pred).figure))
                sections.append(self._section("Error Distribution", self.visualization_service.render_error_distribution(y_true, y_pred).figure))
        elif entry.task_type == "clustering":
            labels = metrics.get("labels")
            if entry.X_test is not None and labels is not None:
                sections.append(self._section("Cluster Scatter", self.visualization_service.render_cluster_scatter(entry.X_test, labels).figure))
                sections.append(self._section("Cluster Distribution", self.visualization_service.render_cluster_distribution(labels).figure))

        title = f"Evaluation Report — {entry.name}"
        if fmt == "pdf":
            return self.report_service.generate_pdf(title, summary, sections, output_path)
        return self.report_service.generate_html(title, summary, sections, output_path)

    @staticmethod
    def _section(title, figure):
        from services.report_service import ReportSection

        return ReportSection(title=title, figure=figure)

    # ------------------------------------------------------------------
    # Async helper
    # ------------------------------------------------------------------
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
