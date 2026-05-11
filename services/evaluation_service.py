"""
XAI Studio — Evaluation Service
===============================
Model registry, evaluation metrics, and comparison logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import label_binarize

from core.model_loader import load_model_file, detect_model_info
from core.persistence import list_saved_models
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEntry:
    entry_id: str
    name: str
    task_type: str
    source: str
    model: Any | None = None
    model_path: str | None = None
    metadata: dict = field(default_factory=dict)
    dataset_name: str | None = None
    trained_at: str | None = None
    training_time: float | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_proba: np.ndarray | None = None
    evaluation: dict | None = None


class ModelRegistry:
    """Local registry for trained/saved models and their artifacts."""

    def __init__(self):
        self._entries: dict[str, ModelEntry] = {}

    def list_entries(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def get(self, entry_id: str) -> ModelEntry | None:
        return self._entries.get(entry_id)

    def clear(self):
        self._entries = {}

    def register_in_memory(
        self,
        name: str,
        model: Any,
        metadata: dict,
        task_type: str,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        training_time: float | None = None,
        dataset_name: str | None = None,
    ) -> ModelEntry:
        entry_id = f"mem::{name}"
        meta = detect_model_info(model, metadata)
        entry = ModelEntry(
            entry_id=entry_id,
            name=name,
            task_type=task_type,
            source="memory",
            model=model,
            metadata=meta,
            dataset_name=dataset_name,
            trained_at=datetime.now().isoformat(),
            training_time=training_time,
            X_test=X_test,
            y_test=y_test,
        )
        self._entries[entry_id] = entry
        return entry

    def register_saved_models(self):
        for item in list_saved_models():
            name = item.get("model_name") or item.get("filename", "Saved Model")
            entry_id = f"saved::{item.get('filename', name)}"
            task_type = item.get("task_type", "unknown")
            entry = ModelEntry(
                entry_id=entry_id,
                name=name,
                task_type=task_type,
                source="saved",
                model=None,
                model_path=item.get("filepath"),
                metadata=item,
                dataset_name=item.get("dataset_name"),
                trained_at=item.get("saved_at"),
                training_time=item.get("training_time"),
            )
            self._entries[entry_id] = entry

    def register_external_model(self, filepath: str, name: str | None = None) -> ModelEntry:
        model, meta = load_model_file(filepath)
        meta = detect_model_info(model, meta)
        entry_id = f"external::{name or meta.get('model_class', 'model')}"
        entry = ModelEntry(
            entry_id=entry_id,
            name=name or meta.get("algorithm", "External Model"),
            task_type=meta.get("task_type", "unknown"),
            source="external",
            model=model,
            model_path=filepath,
            metadata=meta,
            trained_at=datetime.now().isoformat(),
        )
        self._entries[entry_id] = entry
        return entry

    def register_loaded_model(
        self,
        name: str,
        model: Any,
        metadata: dict,
        task_type: str,
        dataset_name: str | None = None,
    ) -> ModelEntry:
        entry_id = f"external::{name}"
        meta = detect_model_info(model, metadata)
        entry = ModelEntry(
            entry_id=entry_id,
            name=name,
            task_type=task_type,
            source="external",
            model=model,
            model_path=None,
            metadata=meta,
            dataset_name=dataset_name,
            trained_at=datetime.now().isoformat(),
        )
        self._entries[entry_id] = entry
        return entry

    def ensure_loaded(self, entry: ModelEntry) -> ModelEntry:
        if entry.model is not None:
            return entry
        if entry.model_path:
            model, meta = load_model_file(entry.model_path)
            entry.model = model
            entry.metadata = detect_model_info(model, {**entry.metadata, **meta})
            if entry.task_type == "unknown":
                entry.task_type = entry.metadata.get("task_type", "unknown")
        return entry

    def attach_artifacts(
        self,
        entry_id: str,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        y_pred: np.ndarray | None = None,
        y_proba: np.ndarray | None = None,
    ):
        entry = self._entries.get(entry_id)
        if not entry:
            return
        if X_test is not None:
            entry.X_test = X_test
        if y_test is not None:
            entry.y_test = y_test
        if y_pred is not None:
            entry.y_pred = y_pred
        if y_proba is not None:
            entry.y_proba = y_proba


class EvaluationService:
    """Compute metrics and comparisons using the model registry."""

    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self._metrics_cache: dict[str, dict] = {}

    def evaluate_models(self, entry_ids: list[str]) -> dict[str, dict]:
        results = {}
        for entry_id in entry_ids:
            entry = self.registry.get(entry_id)
            if not entry:
                continue
            results[entry_id] = self.evaluate_entry(entry)
        return results

    def evaluate_entry(self, entry: ModelEntry) -> dict:
        cache_key = self._cache_key(entry)
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        entry = self.registry.ensure_loaded(entry)
        if entry.X_test is None and entry.task_type != "clustering":
            return {"error": "Missing X_test for evaluation."}
        if entry.task_type in ("classification", "regression") and entry.y_test is None:
            return {"error": "Missing y_test for evaluation."}

        if entry.task_type == "classification":
            result = self._evaluate_classification(entry)
        elif entry.task_type == "regression":
            result = self._evaluate_regression(entry)
        elif entry.task_type == "clustering":
            result = self._evaluate_clustering(entry)
        else:
            result = {"error": "Unknown task type"}

        entry.evaluation = result
        self._metrics_cache[cache_key] = result
        return result

    def comparison_table(self, entry_ids: list[str]) -> pd.DataFrame:
        rows = []
        for entry_id in entry_ids:
            entry = self.registry.get(entry_id)
            if not entry:
                continue
            metrics = entry.evaluation or self.evaluate_entry(entry)
            row = {
                "Model": entry.name,
                "Task": entry.task_type,
                "Score": metrics.get("score_global"),
                "Training Time": entry.training_time,
                "Complexity": metrics.get("complexity_label"),
                "Status": "OK" if "error" not in metrics else "Failed",
            }

            for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "rmse", "mae", "r2", "mape", "silhouette"):
                if key in metrics:
                    row[key] = metrics.get(key)
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty and "Score" in df.columns:
            df = df.sort_values("Score", ascending=False, na_position="last").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Evaluation routines
    # ------------------------------------------------------------------
    def _evaluate_classification(self, entry: ModelEntry) -> dict:
        model = entry.model
        X_test = entry.X_test
        y_test = entry.y_test

        y_pred = entry.y_pred
        if y_pred is None:
            y_pred = model.predict(X_test)
            entry.y_pred = y_pred

        y_proba = entry.y_proba
        if y_proba is None and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                entry.y_proba = y_proba
            except Exception:
                y_proba = None

        y_arr = np.array(y_test)
        is_multi = y_arr.ndim > 1 and y_arr.shape[1] > 1

        if is_multi:
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="samples", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="samples", zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average="samples", zero_division=0)),
                "confusion_matrix": [
                    confusion_matrix(y_arr[:, i], np.array(y_pred)[:, i]).tolist()
                    for i in range(y_arr.shape[1])
                ],
                "classification_report": classification_report(y_test, y_pred, zero_division=0),
                "classification_report_dict": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            }
        else:
            avg = "weighted" if len(np.unique(y_test)) > 2 else "binary"
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, zero_division=0),
                "classification_report_dict": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            }

            roc_info = self._roc_info(y_test, y_proba)
            if roc_info:
                metrics.update(roc_info)

            pr_info = self._pr_info(y_test, y_proba)
            if pr_info:
                metrics.update(pr_info)

        metrics.update(self._complexity_info(model))
        metrics["score_global"] = self._global_score(metrics, task_type="classification")

        return metrics

    def _evaluate_regression(self, entry: ModelEntry) -> dict:
        model = entry.model
        X_test = entry.X_test
        y_test = entry.y_test

        y_pred = entry.y_pred
        if y_pred is None:
            y_pred = model.predict(X_test)
            entry.y_pred = y_pred

        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
            "y_pred": np.array(y_pred),
        }

        metrics.update(self._complexity_info(model))
        metrics["score_global"] = self._global_score(metrics, task_type="regression")
        return metrics

    def _evaluate_clustering(self, entry: ModelEntry) -> dict:
        model = entry.model
        X = entry.X_test
        if X is None:
            return {"error": "Missing X_test for clustering evaluation."}

        labels = entry.y_pred
        if labels is None:
            try:
                labels = model.predict(X) if hasattr(model, "predict") else model.fit_predict(X)
            except Exception as exc:
                return {"error": str(exc)}
            entry.y_pred = labels

        metrics = {
            "silhouette": self._safe_metric(silhouette_score, X, labels),
            "davies_bouldin": self._safe_metric(davies_bouldin_score, X, labels),
            "calinski_harabasz": self._safe_metric(calinski_harabasz_score, X, labels),
            "labels": labels,
        }

        metrics.update(self._complexity_info(model))
        metrics["score_global"] = self._global_score(metrics, task_type="clustering")
        return metrics

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_metric(fn, *args):
        try:
            return float(fn(*args))
        except Exception:
            return None

    @staticmethod
    def _roc_info(y_true, y_proba):
        if y_proba is None:
            return {}
        y_true = np.array(y_true)
        if y_proba.ndim == 2 and y_proba.shape[1] > 2:
            classes = np.unique(y_true)
            y_bin = label_binarize(y_true, classes=classes)
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        else:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        return {"roc_auc": float(roc_auc), "roc_curve": {"fpr": fpr, "tpr": tpr}}

    @staticmethod
    def _pr_info(y_true, y_proba):
        if y_proba is None:
            return {}
        y_true = np.array(y_true)
        if y_proba.ndim == 2 and y_proba.shape[1] > 2:
            classes = np.unique(y_true)
            y_bin = label_binarize(y_true, classes=classes)
            precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        return {"pr_curve": {"precision": precision, "recall": recall}}

    @staticmethod
    def _global_score(metrics: dict, task_type: str) -> float | None:
        try:
            if task_type == "classification":
                acc = metrics.get("accuracy") or 0
                f1 = metrics.get("f1_score") or 0
                roc = metrics.get("roc_auc") or acc
                return float(0.4 * acc + 0.35 * f1 + 0.25 * roc)
            if task_type == "regression":
                r2 = metrics.get("r2") or 0
                rmse = metrics.get("rmse") or 0
                mape = metrics.get("mape") or 0
                inv_rmse = 1 / (1 + rmse)
                inv_mape = 1 / (1 + mape)
                return float(0.5 * r2 + 0.25 * inv_rmse + 0.25 * inv_mape)
            if task_type == "clustering":
                sil = metrics.get("silhouette") or 0
                dbi = metrics.get("davies_bouldin")
                dbi_score = 1 / (1 + dbi) if dbi is not None else 0
                return float(0.6 * sil + 0.4 * dbi_score)
        except Exception:
            return None
        return None

    @staticmethod
    def _complexity_info(model) -> dict:
        complexity = None
        if hasattr(model, "n_estimators"):
            val = getattr(model, "n_estimators")
            if val is not None:
                complexity = int(val)
        elif hasattr(model, "estimators_"):
            try:
                complexity = len(model.estimators_)
            except Exception:
                complexity = None
        elif hasattr(model, "tree_"):
            try:
                complexity = int(model.tree_.node_count)
            except Exception:
                complexity = None
        elif hasattr(model, "coef_"):
            try:
                size = np.prod(model.coef_.shape) if model.coef_ is not None else None
                if size is not None:
                    complexity = int(size)
            except Exception:
                complexity = None

        label = "Low"
        if complexity is not None:
            if complexity >= 500:
                label = "High"
            elif complexity >= 100:
                label = "Medium"

        return {"complexity": complexity, "complexity_label": label}

    @staticmethod
    def _cache_key(entry: ModelEntry) -> str:
        parts = [entry.entry_id]
        for arr in (entry.X_test, entry.y_test, entry.y_pred):
            if arr is None:
                parts.append("none")
                continue
            shape = getattr(arr, "shape", None)
            mean = float(np.nanmean(arr)) if hasattr(arr, "__array__") else 0
            parts.append(f"{shape}-{mean:.5f}")
        return "|".join(parts)
