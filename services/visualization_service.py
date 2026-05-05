"""
XAI Studio — Visualization Service
==================================
Matplotlib/Plotly visualization engine for evaluation and XAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ui.components.plot_canvas import apply_plot_style, C
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False


@dataclass
class PlotPayload:
    figure: plt.Figure | None
    html: str | None
    engine: str
    title: str


class VisualizationService:
    """Create plots for evaluation dashboards and reports."""

    def __init__(self):
        self.plotly_available = _PLOTLY_AVAILABLE

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def render_confusion_matrix(self, cm, labels=None, title="Confusion Matrix") -> PlotPayload:
        fig = self._fig_base(figsize=(6.5, 5.5))
        ax = fig.add_subplot(111)
        ax.set_facecolor("#111c2e")

        cm = np.array(cm)
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        labels = labels or [str(i) for i in range(cm.shape[0])]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center", color=C.TEXT)

        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues"))
            fig_pl.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    def render_roc_curve(self, fpr, tpr, label="Model", title="ROC Curve") -> PlotPayload:
        fig = self._fig_base(figsize=(6.8, 5.2))
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, color=C.ACCENT, linewidth=2.2, label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", color=C.TEXT_DIM, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label))
            fig_pl.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
            fig_pl.update_layout(title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    def render_precision_recall_curve(self, recall, precision, label="Model", title="Precision-Recall Curve") -> PlotPayload:
        fig = self._fig_base(figsize=(6.8, 5.2))
        ax = fig.add_subplot(111)
        ax.plot(recall, precision, color=C.INFO, linewidth=2.2, label=label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=label))
            fig_pl.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    def render_multi_roc(self, curves, title="ROC Curves") -> PlotPayload:
        fig = self._fig_base(figsize=(7.2, 5.4))
        ax = fig.add_subplot(111)
        for c in curves:
            ax.plot(c["fpr"], c["tpr"], linewidth=2.0, label=c.get("label", "Model"))
        ax.plot([0, 1], [0, 1], linestyle="--", color=C.TEXT_DIM, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            for c in curves:
                fig_pl.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=c.get("label", "Model")))
            fig_pl.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
            fig_pl.update_layout(title=title, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    def render_multi_pr(self, curves, title="Precision-Recall Curves") -> PlotPayload:
        fig = self._fig_base(figsize=(7.2, 5.4))
        ax = fig.add_subplot(111)
        for c in curves:
            ax.plot(c["recall"], c["precision"], linewidth=2.0, label=c.get("label", "Model"))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            for c in curves:
                fig_pl.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=c.get("label", "Model")))
            fig_pl.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    def render_predicted_vs_actual(self, y_true, y_pred, title="Predicted vs Actual") -> PlotPayload:
        fig = self._fig_base(figsize=(6.8, 5.2))
        ax = fig.add_subplot(111)
        ax.scatter(y_true, y_pred, s=24, color=C.ACCENT, alpha=0.75)
        ax.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], linestyle="--", color=C.TEXT_DIM)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Predictions"))
            fig_pl.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    def render_residuals(self, y_true, y_pred, title="Residuals") -> PlotPayload:
        residuals = np.array(y_true) - np.array(y_pred)
        fig = self._fig_base(figsize=(6.8, 5.2))
        ax = fig.add_subplot(111)
        ax.scatter(y_pred, residuals, s=24, color=C.INFO, alpha=0.75)
        ax.axhline(0, color=C.TEXT_DIM, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", name="Residuals"))
            fig_pl.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Residual")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    def render_error_distribution(self, y_true, y_pred, title="Error Distribution") -> PlotPayload:
        errors = np.array(y_true) - np.array(y_pred)
        fig = self._fig_base(figsize=(6.8, 5.2))
        ax = fig.add_subplot(111)
        ax.hist(errors, bins=30, color=C.WARNING, alpha=0.75)
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Histogram(x=errors, name="Errors"))
            fig_pl.update_layout(title=title, xaxis_title="Error", yaxis_title="Count")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def render_cluster_scatter(self, X, labels, title="Cluster Scatter") -> PlotPayload:
        coords = self._project_2d(X)
        fig = self._fig_base(figsize=(6.8, 5.2))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=26, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                                        marker=dict(color=labels, colorscale="Viridis")))
            fig_pl.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    def render_cluster_distribution(self, labels, title="Cluster Distribution") -> PlotPayload:
        labels = np.array(labels)
        unique, counts = np.unique(labels, return_counts=True)

        fig = self._fig_base(figsize=(6.2, 4.8))
        ax = fig.add_subplot(111)
        ax.bar([str(u) for u in unique], counts, color=C.SUCCESS)
        ax.set_title(title)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Bar(x=[str(u) for u in unique], y=counts))
            fig_pl.update_layout(title=title, xaxis_title="Cluster", yaxis_title="Count")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    # ------------------------------------------------------------------
    # Generic
    # ------------------------------------------------------------------
    def render_feature_importance(self, importances, feature_names, title="Feature Importance") -> PlotPayload:
        importances = np.array(importances)
        sorted_idx = np.argsort(importances)
        names = [feature_names[i] for i in sorted_idx]
        vals = importances[sorted_idx]

        fig = self._fig_base(figsize=(7.0, 5.4))
        ax = fig.add_subplot(111)
        ax.barh(names, vals, color=C.ACCENT)
        ax.set_title(title)
        ax.set_xlabel("Importance")
        fig.tight_layout()

        html = None
        if self.plotly_available:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Bar(x=vals, y=names, orientation="h"))
            fig_pl.update_layout(title=title, xaxis_title="Importance")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    def render_learning_curve(self, train_sizes, train_scores, val_scores, title="Learning Curve") -> PlotPayload:
        fig = self._fig_base(figsize=(6.8, 5.2))
        ax = fig.add_subplot(111)
        if train_sizes is None or train_scores is None or val_scores is None:
            ax.text(0.5, 0.5, "Learning curve data not available", ha="center", va="center", color=C.TEXT_MUTED)
        else:
            ax.plot(train_sizes, train_scores, label="Train", color=C.ACCENT)
            ax.plot(train_sizes, val_scores, label="Validation", color=C.INFO)
            ax.set_xlabel("Training size")
            ax.set_ylabel("Score")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
        ax.set_title(title)
        fig.tight_layout()

        html = None
        if self.plotly_available and train_sizes is not None:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Scatter(x=train_sizes, y=train_scores, mode="lines", name="Train"))
            fig_pl.add_trace(go.Scatter(x=train_sizes, y=val_scores, mode="lines", name="Validation"))
            fig_pl.update_layout(title=title, xaxis_title="Training size", yaxis_title="Score")
            html = pio.to_html(fig_pl, include_plotlyjs="cdn", full_html=True)

        return PlotPayload(fig, html, self._engine_label(html), title)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fig_base(self, figsize=(6, 4)):
        apply_plot_style()
        fig = plt.figure(figsize=figsize, facecolor=C.BG_CARD)
        return fig

    @staticmethod
    def _project_2d(X):
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] == 2:
            return X
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=42)
            return pca.fit_transform(X)
        except Exception:
            return X[:, :2]

    @staticmethod
    def _engine_label(html):
        return "plotly" if html else "matplotlib"
