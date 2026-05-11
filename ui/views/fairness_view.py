"""
XAI Studio — Fairness View
==============================
Bias detection dashboard with fairness metrics and comparison charts.
"""

import threading
import tkinter as tk
from tkinter import ttk
import numpy as np

from ui.widgets import C, F, Card, ModernButton, SectionHeader, StyledTreeview
from ui.components.plot_canvas import PlotCanvas
from ui.components.dialogs import show_error
from services.pipeline_service import PipelineService
from services.i18n import _

from utils.logger import get_logger

logger = get_logger(__name__)


class FairnessView(ttk.Frame):
    """Bias detection and fairness metrics dashboard."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._metrics_result = None
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
            header, icon="⚖", title=_("fair_title"),
            subtitle=_("fair_subtitle"),
        ).pack(side="left")

        # ── Controls card ───────────────────────────────────
        ctrl_card = Card(ct, accent_color=C.WARNING, pad=16)
        ctrl_card.pack(fill="x", padx=px, pady=(16, 0))

        row1 = tk.Frame(ctrl_card.inner, bg=C.BG_CARD)
        row1.pack(fill="x", pady=(0, 10))

        tk.Label(row1, text=_("fair_lbl_sens"), font=F.BODY,
                 bg=C.BG_CARD, fg=C.TEXT).pack(side="left", padx=(0, 8))

        self._sensitive_combo = ttk.Combobox(row1, state="readonly", width=25)
        self._sensitive_combo.pack(side="left", padx=(0, 16))

        tk.Label(row1, text=_("fair_lbl_metric"), font=F.BODY,
                 bg=C.BG_CARD, fg=C.TEXT).pack(side="left", padx=(0, 8))

        self._metric_combo = ttk.Combobox(
            row1, values=["accuracy", "positive_rate", "tpr", "fpr"],
            state="readonly", width=18,
        )
        self._metric_combo.set("accuracy")
        self._metric_combo.pack(side="left")

        row2 = tk.Frame(ctrl_card.inner, bg=C.BG_CARD)
        row2.pack(fill="x")

        ModernButton(
            row2, text=_("fair_btn_analyze"), icon="▶",
            style="primary", command=self._run_analysis, bg=C.BG_CARD,
        ).pack(side="left")

        ModernButton(
            row2, text=_("fair_btn_export"), icon="💾",
            style="secondary",
            command=lambda: self._plot_canvas.export_png(),
            bg=C.BG_CARD,
        ).pack(side="right")

        # ── Alerts banner ───────────────────────────────────
        self._alerts_frame = tk.Frame(ct, bg=C.BG_MAIN)
        self._alerts_frame.pack(fill="x", padx=px, pady=(12, 0))

        # ── Summary metrics ─────────────────────────────────
        tk.Label(ct, text=_("fair_summary_title"), font=F.H2,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="w", padx=px, pady=(12, 8))

        self._summary_frame = tk.Frame(ct, bg=C.BG_MAIN)
        self._summary_frame.pack(fill="x", padx=px)

        # ── Per-group table ─────────────────────────────────
        tk.Label(ct, text=_("fair_table_title"), font=F.H2,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="w", padx=px, pady=(16, 8))

        self._table_container = tk.Frame(ct, bg=C.BG_MAIN)
        self._table_container.pack(fill="x", padx=px, pady=(0, 8))

        cols = (_("fair_col_grp"), _("fair_col_n"), _("fair_col_acc"), _("fair_col_pos"), _("fair_col_tpr"), _("fair_col_fpr"))
        widths = {cols[0]: 150, cols[1]: 80, cols[2]: 100,
                  cols[3]: 100, cols[4]: 80, cols[5]: 80}
        self._stv = StyledTreeview(
            self._table_container, columns=cols, col_widths=widths, height=6,
        )
        self._stv.pack(fill="x")

        # ── Plot area ───────────────────────────────────────
        self._plot_canvas = PlotCanvas(ct)
        self._plot_canvas.pack(fill="both", expand=True, padx=px, pady=(8, 24))

    def on_enter(self):
        """Refresh sensitive attribute list when entering view."""
        self._refresh_columns()

    def _refresh_columns(self):
        """Populate the sensitive attribute dropdown from the loaded dataset."""
        df = self._service.dataframe
        if df is not None:
            columns = df.columns.tolist()
            self._sensitive_combo["values"] = columns
            if columns:
                self._sensitive_combo.set(columns[0])
        else:
            self._sensitive_combo["values"] = []

    def _run_analysis(self):
        sensitive_col = self._sensitive_combo.get()
        if not sensitive_col:
            show_error(_("fair_err_title"), _("fair_err_no_sens"))
            return

        try:
            model, meta = self._service.get_active_model()
            if model is None:
                raise RuntimeError(_("fair_err_no_model"))
        except Exception as e:
            show_error(_("fair_err_title"), str(e))
            return

        pr = self._service.preprocessing_result
        if pr is None:
            show_error(_("fair_err_title"), _("fair_err_no_data"))
            return

        df = self._service.dataframe
        if df is None:
            show_error(_("fair_err_title"), _("fair_err_no_ds"))
            return

        if sensitive_col not in df.columns:
            show_error(_("fair_err_title"), _("fair_err_col_not_found").format(sensitive_col))
            return

        metric_key = self._metric_combo.get()

        def _compute():
            try:
                from core.fairness.bias_detector import (
                    compute_fairness_metrics, plot_fairness_comparison,
                )

                # Get predictions on test set
                y_pred = model.predict(pr.X_test)
                y_true = pr.y_test

                # Get sensitive attribute values for test set
                # We need to align with the test split indices
                # Use the original dataframe to get the sensitive column
                target_col = self._service.target_column
                X_original = df.drop(columns=[target_col])

                # Total samples before split
                total = len(df)
                test_size = len(pr.X_test)
                train_size = len(pr.X_train)

                # Reconstruct test indices (same random_state as preprocessing)
                from sklearn.model_selection import train_test_split
                indices = np.arange(total)

                # Handle dropped rows from missing target values
                y_orig = df[target_col]
                valid_mask = y_orig.notnull()
                valid_indices = indices[valid_mask]

                from config.settings import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
                _, test_indices = train_test_split(
                    valid_indices, test_size=DEFAULT_TEST_SIZE,
                    random_state=DEFAULT_RANDOM_STATE,
                )

                sensitive_values = df[sensitive_col].values[test_indices[:len(y_true)]]

                results = compute_fairness_metrics(y_true, y_pred, sensitive_values)
                fig = plot_fairness_comparison(results, metric_key)

                self._metrics_result = results

                # Update UI on main thread
                self.after(0, lambda: self._display_results(results))
                self._plot_canvas.after(
                    0, lambda: self._plot_canvas.update_figure(fig)
                )
            except Exception as exc:
                logger.error("Fairness analysis error: %s", exc)
                self.after(
                    0, lambda: show_error(_("fair_err_analysis_title"), str(exc))
                )

        threading.Thread(target=_compute, daemon=True).start()

    def _display_results(self, results):
        """Update the UI with fairness analysis results."""
        # ── Alerts ──────────────────────────────────────────
        for w in self._alerts_frame.winfo_children():
            w.destroy()

        if results["bias_detected"]:
            alert_card = Card(self._alerts_frame, accent_color=C.DANGER, pad=12)
            alert_card.pack(fill="x", pady=(0, 4))

            tk.Label(
                alert_card.inner, text=_("fair_alert_bias"),
                font=F.H3, bg=C.BG_CARD, fg=C.DANGER,
            ).pack(anchor="w", pady=(0, 6))

            for alert in results["alerts"]:
                tk.Label(
                    alert_card.inner, text=alert,
                    font=F.SMALL, bg=C.BG_CARD, fg=C.WARNING,
                    wraplength=700, justify="left",
                ).pack(anchor="w", pady=1)
        else:
            ok_card = Card(self._alerts_frame, accent_color=C.SUCCESS, pad=12)
            ok_card.pack(fill="x")
            tk.Label(
                ok_card.inner,
                text=_("fair_alert_ok"),
                font=F.H4, bg=C.BG_CARD, fg=C.SUCCESS,
            ).pack(anchor="w")

        # ── Summary metrics ─────────────────────────────────
        for w in self._summary_frame.winfo_children():
            w.destroy()

        summary = results["summary"]
        metrics_info = [
            ("Dem. Parity Diff", f"{summary['demographic_parity_diff']:.4f}", C.ACCENT),
            ("Eq. Odds (TPR)", f"{summary['equalized_odds_tpr_diff']:.4f}", C.INFO),
            ("Eq. Odds (FPR)", f"{summary['equalized_odds_fpr_diff']:.4f}", C.INFO),
            ("Disparate Impact", f"{summary['disparate_impact_ratio']:.4f}", C.WARNING),
            ("Accuracy Spread", f"{summary['accuracy_spread']:.4f}", C.SUCCESS),
        ]

        from ui.widgets import MetricTile
        for icon, val, color in metrics_info:
            # Determine color based on thresholds
            tile = MetricTile(
                self._summary_frame, value=val, label=icon, color=color,
            )
            tile.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # ── Per-group table ─────────────────────────────────
        self._stv.tree.delete(*self._stv.tree.get_children())

        for g in results["groups"]:
            # Color-code values
            acc = g["accuracy"]
            self._stv.tree.insert("", "end", values=(
                g["group"],
                g["count"],
                f"{acc:.4f}",
                f"{g['positive_rate']:.4f}",
                f"{g['tpr']:.4f}",
                f"{g['fpr']:.4f}",
            ))
