"""
XAI Studio — XAI Panel
======================
Reusable panel for local explainability visuals.
"""

from __future__ import annotations

import tkinter as tk

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ui.widgets import C, F, Card
from ui.components.plot_canvas import PlotCanvas, apply_plot_style
from services.xai_service import XAIService
from utils.logger import get_logger

logger = get_logger(__name__)


class XAIPanel(tk.Frame):
    """Reusable local explanation panel with contributions and waterfall plots."""

    def __init__(self, parent, xai_service: XAIService | None = None, title="Local XAI"):
        super().__init__(parent, bg=C.BG_MAIN)
        self._xai = xai_service or XAIService()
        self._title = title
        self._build()

    def _build(self):
        card = Card(self, accent_color=C.INFO, pad=16)
        card.pack(fill="both", expand=True)

        header = tk.Frame(card.inner, bg=C.BG_CARD)
        header.pack(fill="x")
        tk.Label(header, text=self._title, font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(side="left")
        self._status = tk.Label(header, text="-", font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_MUTED)
        self._status.pack(side="right")

        self._bar_canvas = PlotCanvas(card.inner, bg=C.BG_CARD, auto_size=False)
        self._bar_canvas.pack(fill="both", expand=True, pady=(10, 8))

        self._waterfall_canvas = PlotCanvas(card.inner, bg=C.BG_CARD, auto_size=False)
        self._waterfall_canvas.pack(fill="both", expand=True)

    def update(self, model, instance, feature_names, task_type="classification"):
        self._status.configure(text="Computing...")
        try:
            exp = self._xai.local_explanation(model, instance, feature_names, task_type)
            fig_bar = self._plot_contributions(exp.contributions)
            self._bar_canvas.update_figure(fig_bar)
        except Exception as exc:
            logger.error("Local explanation error: %s", exc)

        try:
            fig_waterfall = self._xai.shap_plot(model, np.array(instance).reshape(1, -1), feature_names, plot_type="waterfall", instance_idx=0)
            self._waterfall_canvas.update_figure(fig_waterfall)
        except Exception as exc:
            logger.warning("Waterfall plot unavailable: %s", exc)

        self._status.configure(text="Ready")

    def _plot_contributions(self, contributions):
        apply_plot_style()
        names = [c[0] for c in contributions][:12]
        vals = [c[1] for c in contributions][:12]
        colors = ["#ef4444" if v < 0 else "#10b981" for v in vals]
        fig = plt.figure(figsize=(7, 4.2), facecolor=C.BG_CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#111c2e")
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0, color=C.TEXT_DIM, linestyle="--", linewidth=0.8)
        ax.set_title("Feature contributions")
        fig.tight_layout()
        return fig
