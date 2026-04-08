"""
XAI Studio — Visualization View
===================================
Interactive PDP plots embedded in the Tkinter interface.
"""

import threading
import tkinter as tk
from tkinter import ttk

from ui.widgets import C, F, Card, ModernButton, SectionHeader
from ui.components.plot_canvas import PlotCanvas
from ui.components.dialogs import show_error
from services.pipeline_service import PipelineService

from utils.logger import get_logger

logger = get_logger(__name__)


class VisualizationView(ttk.Frame):
    """PDP visualization dashboard."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._feature_names = []
        self._build()

    def _build(self):
        canvas = tk.Canvas(self, bg=C.BG_MAIN, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll = tk.Frame(canvas, bg=C.BG_MAIN)
        self._scroll.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._scroll, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ct = self._scroll
        px = 28

        # ── Header ──────────────────────────────────────────
        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(
            header, icon="📈", title="Visualisations",
            subtitle="Partial Dependence Plots (PDP) — Effet marginal des features",
        ).pack(side="left")

        # ── Controls card ───────────────────────────────────
        ctrl_card = Card(ct, accent_color=C.INFO, pad=16)
        ctrl_card.pack(fill="x", padx=px, pady=(16, 0))

        # Row 1: Plot type
        row1 = tk.Frame(ctrl_card.inner, bg=C.BG_CARD)
        row1.pack(fill="x", pady=(0, 10))

        tk.Label(row1, text="Type de PDP :", font=F.BODY,
                 bg=C.BG_CARD, fg=C.TEXT).pack(side="left", padx=(0, 8))

        self._pdp_type = ttk.Combobox(
            row1, values=["1D (une feature)", "2D (deux features)"],
            state="readonly", width=20,
        )
        self._pdp_type.set("1D (une feature)")
        self._pdp_type.pack(side="left", padx=(0, 16))
        self._pdp_type.bind("<<ComboboxSelected>>", self._on_type_change)

        # Row 2: Feature selection
        row2 = tk.Frame(ctrl_card.inner, bg=C.BG_CARD)
        row2.pack(fill="x", pady=(0, 10))

        tk.Label(row2, text="Feature 1 :", font=F.BODY,
                 bg=C.BG_CARD, fg=C.TEXT).pack(side="left", padx=(0, 8))

        self._feat1_combo = ttk.Combobox(row2, state="readonly", width=25)
        self._feat1_combo.pack(side="left", padx=(0, 16))

        self._feat2_label = tk.Label(row2, text="Feature 2 :", font=F.BODY,
                                      bg=C.BG_CARD, fg=C.TEXT)
        self._feat2_combo = ttk.Combobox(row2, state="readonly", width=25)

        # Row 3: Actions
        row3 = tk.Frame(ctrl_card.inner, bg=C.BG_CARD)
        row3.pack(fill="x")

        ModernButton(
            row3, text="Générer PDP", icon="▶",
            style="primary", command=self._run_pdp, bg=C.BG_CARD,
        ).pack(side="left")

        ModernButton(
            row3, text="Exporter PNG", icon="💾",
            style="secondary", command=lambda: self._plot_canvas.export_png(),
            bg=C.BG_CARD,
        ).pack(side="right")

        # ── Plot area ───────────────────────────────────────
        self._plot_canvas = PlotCanvas(ct)
        self._plot_canvas.pack(fill="both", expand=True, padx=px, pady=(16, 24))

    def on_enter(self):
        """Called when the view becomes active — refresh feature list."""
        self._refresh_features()

    def _refresh_features(self):
        pr = self._service.preprocessing_result
        if pr and pr.feature_names:
            self._feature_names = pr.feature_names
        else:
            _, meta = self._service.get_active_model()
            self._feature_names = meta.get("feature_names", []) if meta else []

        self._feat1_combo["values"] = self._feature_names
        self._feat2_combo["values"] = self._feature_names

        if self._feature_names:
            self._feat1_combo.set(self._feature_names[0])
            if len(self._feature_names) > 1:
                self._feat2_combo.set(self._feature_names[1])

    def _on_type_change(self, event=None):
        pdp_type = self._pdp_type.get()
        if "2D" in pdp_type:
            self._feat2_label.pack(side="left", padx=(0, 8))
            self._feat2_combo.pack(side="left")
        else:
            self._feat2_label.pack_forget()
            self._feat2_combo.pack_forget()

    def _run_pdp(self):
        try:
            model, meta = self._service.get_active_model()
            if model is None:
                raise RuntimeError("Aucun modèle disponible.")
        except Exception as e:
            show_error("Erreur", str(e))
            return

        pr = self._service.preprocessing_result
        if pr is None:
            show_error("Erreur", "Données non préprocessées.")
            return

        feat1_name = self._feat1_combo.get()
        if not feat1_name or feat1_name not in self._feature_names:
            show_error("Erreur", "Sélectionnez une feature valide.")
            return

        feat1_idx = self._feature_names.index(feat1_name)
        pdp_type = self._pdp_type.get()
        is_2d = "2D" in pdp_type

        if is_2d:
            feat2_name = self._feat2_combo.get()
            if not feat2_name or feat2_name not in self._feature_names:
                show_error("Erreur", "Sélectionnez une deuxième feature valide.")
                return
            feat2_idx = self._feature_names.index(feat2_name)

        def _compute():
            try:
                from core.explainability.pdp_explainer import compute_pdp_1d, compute_pdp_2d

                X = pr.X_test

                if is_2d:
                    fig = compute_pdp_2d(
                        model, X, (feat1_idx, feat2_idx),
                        feature_names=[feat1_name, feat2_name],
                    )
                else:
                    fig = compute_pdp_1d(model, X, feat1_idx, feat1_name)

                self._plot_canvas.after(
                    0, lambda: self._plot_canvas.update_figure(fig)
                )
            except Exception as exc:
                logger.error("PDP error: %s", exc)
                self._plot_canvas.after(
                    0, lambda: show_error("Erreur PDP", str(exc))
                )

        threading.Thread(target=_compute, daemon=True).start()
