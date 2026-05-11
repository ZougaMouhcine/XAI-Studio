"""
XAI Studio — XAI Tools View
================================
Tkinter tab panel with SHAP, LIME, and Feature Importance tools.
"""

import threading
import tkinter as tk
from tkinter import ttk
import numpy as np

from ui.widgets import C, F, Card, ModernButton, SectionHeader
from ui.components.plot_canvas import PlotCanvas
from ui.components.dialogs import show_error, show_info
from services.pipeline_service import PipelineService
from services.i18n import _

from utils.logger import get_logger

logger = get_logger(__name__)


class XAIView(ttk.Frame):
    """XAI Tools dashboard with SHAP / LIME / Feature Importance tabs."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._build()

    def _build(self):
        # Scrollable container
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
            header, icon="🔍", title=_("xai_title"),
            subtitle=_("xai_subtitle"),
        ).pack(side="left")

        # ── Notebook (tabs) ─────────────────────────────────
        self._notebook = ttk.Notebook(ct)
        self._notebook.pack(fill="both", expand=True, padx=px, pady=(16, 24))

        self._build_shap_tab()
        self._build_lime_tab()
        self._build_fi_tab()

    # ── SHAP Tab ────────────────────────────────────────────
    def _build_shap_tab(self):
        tab = tk.Frame(self._notebook, bg=C.BG_MAIN)
        self._notebook.add(tab, text=" 🔬 SHAP ")

        controls = tk.Frame(tab, bg=C.BG_MAIN)
        controls.pack(fill="x", padx=16, pady=(12, 0))

        tk.Label(controls, text=_("xai_lbl_plot_type"), font=F.BODY,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))

        self._shap_plot_type = ttk.Combobox(
            controls, values=["bar", "beeswarm", "waterfall", "force"],
            state="readonly", width=15,
        )
        self._shap_plot_type.set("bar")
        self._shap_plot_type.pack(side="left", padx=(0, 12))

        tk.Label(controls, text=_("xai_lbl_inst"), font=F.BODY,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))

        self._shap_instance = ttk.Spinbox(controls, from_=0, to=999, width=8)
        self._shap_instance.set(0)
        self._shap_instance.pack(side="left", padx=(0, 12))

        ModernButton(
            controls, text=_("xai_btn_shap"), icon="▶",
            style="primary", command=self._run_shap, bg=C.BG_MAIN,
        ).pack(side="left", padx=(8, 0))

        ModernButton(
            controls, text=_("xai_btn_export"), icon="💾",
            style="secondary", command=lambda: self._shap_canvas.export_png(),
            bg=C.BG_MAIN,
        ).pack(side="right")

        self._shap_canvas = PlotCanvas(tab)
        self._shap_canvas.pack(fill="both", expand=True, padx=16, pady=(12, 16))

    # ── LIME Tab ────────────────────────────────────────────
    def _build_lime_tab(self):
        tab = tk.Frame(self._notebook, bg=C.BG_MAIN)
        self._notebook.add(tab, text=" 🍋 LIME ")

        controls = tk.Frame(tab, bg=C.BG_MAIN)
        controls.pack(fill="x", padx=16, pady=(12, 0))

        tk.Label(controls, text=_("xai_lbl_inst"), font=F.BODY,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))

        self._lime_instance = ttk.Spinbox(controls, from_=0, to=999, width=8)
        self._lime_instance.set(0)
        self._lime_instance.pack(side="left", padx=(0, 12))

        tk.Label(controls, text=_("xai_lbl_feat"), font=F.BODY,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))

        self._lime_nfeat = ttk.Spinbox(controls, from_=5, to=30, width=6)
        self._lime_nfeat.set(10)
        self._lime_nfeat.pack(side="left", padx=(0, 12))

        ModernButton(
            controls, text=_("xai_btn_lime"), icon="▶",
            style="primary", command=self._run_lime, bg=C.BG_MAIN,
        ).pack(side="left", padx=(8, 0))

        ModernButton(
            controls, text=_("xai_btn_export"), icon="💾",
            style="secondary", command=lambda: self._lime_canvas.export_png(),
            bg=C.BG_MAIN,
        ).pack(side="right")

        self._lime_canvas = PlotCanvas(tab)
        self._lime_canvas.pack(fill="both", expand=True, padx=16, pady=(12, 16))

    # ── Feature Importance Tab ──────────────────────────────
    def _build_fi_tab(self):
        tab = tk.Frame(self._notebook, bg=C.BG_MAIN)
        self._notebook.add(tab, text=" 📊 Feature Importance ")

        controls = tk.Frame(tab, bg=C.BG_MAIN)
        controls.pack(fill="x", padx=16, pady=(12, 0))

        ModernButton(
            controls, text=_("xai_btn_fi"), icon="▶",
            style="primary", command=self._run_fi, bg=C.BG_MAIN,
        ).pack(side="left")

        ModernButton(
            controls, text=_("xai_btn_export"), icon="💾",
            style="secondary", command=lambda: self._fi_canvas.export_png(),
            bg=C.BG_MAIN,
        ).pack(side="right")

        self._fi_canvas = PlotCanvas(tab)
        self._fi_canvas.pack(fill="both", expand=True, padx=16, pady=(12, 16))

    # ── Helpers ─────────────────────────────────────────────
    def _get_model_and_data(self):
        """Return (model, X_test, y_test, feature_names, task_type) or raise."""
        model, meta = self._service.get_active_model()
        if model is None:
            raise RuntimeError(_("xai_err_no_model"))

        pr = self._service.preprocessing_result
        if pr is None:
            raise RuntimeError(_("xai_err_no_data"))

        feature_names = meta.get("feature_names", None) or pr.feature_names
        task_type = meta.get("task_type", None) or pr.task_type

        return model, pr.X_test, pr.y_test, feature_names, task_type

    # ── SHAP ────────────────────────────────────────────────
    def _run_shap(self):
        try:
            model, X_test, y_test, feature_names, task_type = self._get_model_and_data()
        except RuntimeError as e:
            show_error(_("xai_err_title"), str(e))
            return

        plot_type = self._shap_plot_type.get()
        instance_idx = int(self._shap_instance.get())

        def _compute():
            try:
                from core.explainability.shap_explainer import (
                    compute_shap_values, plot_shap_summary,
                    plot_shap_waterfall, plot_shap_force,
                )

                # Limit data for performance
                X_sample = X_test[:200] if len(X_test) > 200 else X_test

                shap_values = compute_shap_values(model, X_sample, feature_names)

                if plot_type in ("bar", "beeswarm"):
                    fig = plot_shap_summary(shap_values, X_sample, feature_names, plot_type)
                elif plot_type == "waterfall":
                    idx = min(instance_idx, len(X_sample) - 1)
                    fig = plot_shap_waterfall(shap_values, idx, feature_names)
                elif plot_type == "force":
                    idx = min(instance_idx, len(X_sample) - 1)
                    fig = plot_shap_force(shap_values, idx, feature_names)
                else:
                    fig = plot_shap_summary(shap_values, X_sample, feature_names, "bar")

                self._shap_canvas.after(0, lambda: self._shap_canvas.update_figure(fig))
            except Exception as exc:
                logger.error("SHAP error: %s", exc)
                self._shap_canvas.after(
                    0, lambda: show_error(_("xai_err_shap"), str(exc))
                )

        threading.Thread(target=_compute, daemon=True).start()

    # ── LIME ────────────────────────────────────────────────
    def _run_lime(self):
        try:
            model, X_test, y_test, feature_names, task_type = self._get_model_and_data()
        except RuntimeError as e:
            show_error(_("xai_err_title"), str(e))
            return

        instance_idx = int(self._lime_instance.get())
        num_features = int(self._lime_nfeat.get())

        pr = self._service.preprocessing_result
        if pr is None:
            show_error(_("xai_err_title"), _("xai_err_no_data"))
            return

        def _compute():
            try:
                from core.explainability.lime_explainer import (
                    create_lime_explainer, explain_instance, plot_lime_explanation,
                )

                mode = "classification" if task_type == "classification" else "regression"
                explainer = create_lime_explainer(pr.X_train, feature_names, mode)

                idx = min(instance_idx, len(X_test) - 1)
                explanation = explain_instance(
                    explainer, model, X_test[idx], num_features,
                )
                fig = plot_lime_explanation(
                    explanation,
                    title=_("xai_title_lime").format(idx),
                )
                self._lime_canvas.after(0, lambda: self._lime_canvas.update_figure(fig))
            except Exception as exc:
                logger.error("LIME error: %s", exc)
                self._lime_canvas.after(
                    0, lambda: show_error(_("xai_err_lime"), str(exc))
                )

        threading.Thread(target=_compute, daemon=True).start()

    # ── Feature Importance ──────────────────────────────────
    def _run_fi(self):
        try:
            model, X_test, y_test, feature_names, task_type = self._get_model_and_data()
        except RuntimeError as e:
            show_error(_("xai_err_title"), str(e))
            return

        def _compute():
            try:
                from core.explainability.feature_importance import (
                    get_feature_importance, plot_feature_importance,
                )

                result = get_feature_importance(
                    model, X_test, y_test, feature_names,
                )
                fig = plot_feature_importance(
                    result["importances"], result["feature_names"],
                    title=_("xai_title_fi"),
                )
                self._fi_canvas.after(0, lambda: self._fi_canvas.update_figure(fig))
            except Exception as exc:
                logger.error("Feature importance error: %s", exc)
                self._fi_canvas.after(
                    0, lambda: show_error(_("xai_err_fi"), str(exc))
                )

        threading.Thread(target=_compute, daemon=True).start()
