"""
XAI Studio — XAI Service
========================
Compute explainability artifacts (SHAP, LIME, PDP, feature importance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.logger import get_logger
from core.explainability.feature_importance import get_feature_importance, plot_feature_importance
from core.explainability.shap_explainer import compute_shap_values, plot_shap_summary, plot_shap_waterfall, plot_shap_force
from core.explainability.lime_explainer import create_lime_explainer, explain_instance, plot_lime_explanation
from core.explainability.pdp_explainer import compute_pdp_1d, compute_pdp_2d
from services.visualization_service import VisualizationService

logger = get_logger(__name__)


@dataclass
class LocalExplanation:
    prediction: Any
    contributions: list[tuple[str, float]]


class XAIService:
    """Explainability service with caching and model-aware defaults."""

    def __init__(self, viz: VisualizationService | None = None):
        self._viz = viz or VisualizationService()
        self._shap_cache: dict[str, Any] = {}
        self._lime_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def feature_importance(self, model, X_test, y_test, feature_names):
        result = get_feature_importance(model, X_test, y_test, feature_names)
        fig = plot_feature_importance(result["importances"], result["feature_names"], title="Feature Importance")
        return result, fig

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    def shap_plot(self, model, X_data, feature_names, plot_type="bar", instance_idx=0):
        key = self._cache_key(model, X_data)
        shap_values = self._shap_cache.get(key)
        if shap_values is None:
            shap_values = compute_shap_values(model, X_data, feature_names)
            self._shap_cache[key] = shap_values

        if plot_type in ("bar", "beeswarm"):
            fig = plot_shap_summary(shap_values, X_data, feature_names, plot_type=plot_type)
        elif plot_type == "waterfall":
            idx = min(instance_idx, len(X_data) - 1)
            fig = plot_shap_waterfall(shap_values, idx, feature_names)
        elif plot_type == "force":
            idx = min(instance_idx, len(X_data) - 1)
            fig = plot_shap_force(shap_values, idx, feature_names)
        else:
            fig = plot_shap_summary(shap_values, X_data, feature_names, plot_type="bar")

        return fig

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------
    def lime_plot(self, model, X_train, X_test, feature_names, task_type, instance_idx=0, num_features=10):
        mode = "classification" if task_type == "classification" else "regression"
        key = self._cache_key(model, X_train)
        explainer = self._lime_cache.get(key)
        if explainer is None:
            explainer = create_lime_explainer(X_train, feature_names, mode)
            self._lime_cache[key] = explainer

        idx = min(instance_idx, len(X_test) - 1)
        explanation = explain_instance(explainer, model, X_test[idx], num_features)
        fig = plot_lime_explanation(explanation, title=f"LIME — Instance {idx}")
        return fig

    # ------------------------------------------------------------------
    # PDP
    # ------------------------------------------------------------------
    def pdp_plot(self, model, X, feature_idxs, feature_names=None):
        if isinstance(feature_idxs, (list, tuple)) and len(feature_idxs) == 2:
            return compute_pdp_2d(model, X, tuple(feature_idxs), feature_names=feature_names)
        return compute_pdp_1d(model, X, int(feature_idxs), feature_name=feature_names[0] if feature_names else None)

    # ------------------------------------------------------------------
    # Local explanation
    # ------------------------------------------------------------------
    def local_explanation(self, model, instance, feature_names, task_type="classification") -> LocalExplanation:
        instance = np.array(instance).reshape(1, -1)
        prediction = model.predict(instance)

        try:
            shap_values = compute_shap_values(model, instance, feature_names)
            vals = shap_values.values
            if vals.ndim > 2:
                vals = vals[:, :, 0]
            contrib = vals[0]
            pairs = list(zip(feature_names, contrib))
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            return LocalExplanation(prediction=prediction[0], contributions=pairs)
        except Exception as exc:
            logger.warning("SHAP fallback for local explanation: %s", exc)

        contrib = self._fallback_contributions(model, instance[0], feature_names)
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        return LocalExplanation(prediction=prediction[0], contributions=contrib)

    def _fallback_contributions(self, model, instance, feature_names):
        if hasattr(model, "coef_"):
            coef = np.array(model.coef_)
            if coef.ndim > 1:
                coef = coef[0]
            values = coef * instance
        elif hasattr(model, "feature_importances_"):
            values = np.array(model.feature_importances_) * instance
        else:
            values = np.zeros(len(instance))
        return list(zip(feature_names, values.tolist()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(model, X_data):
        shape = getattr(X_data, "shape", None)
        return f"{id(model)}-{shape}"