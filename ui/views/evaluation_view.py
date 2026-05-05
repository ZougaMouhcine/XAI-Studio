"""
XAI Studio — Evaluation View (v2 — Dashboard Design)
=======================================================
Compare model metrics in a styled dashboard with detail tabs.
"""

import tkinter as tk
from tkinter import ttk

from ui.widgets import C, F, Card, MetricTile, ModernButton, SectionHeader, StyledTreeview, LogPanel, bind_mousewheel_to
from ui.components.dialogs import show_error, show_info
from services.pipeline_service import PipelineService


class EvaluationView(ttk.Frame):
    """Evaluation dashboard with comparison table and per-model details."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._build()

    def _build(self):
        canvas = tk.Canvas(self, bg=C.BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll = tk.Frame(canvas, bg=C.BG_MAIN)
        self._scroll.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=self._scroll, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        bind_mousewheel_to(canvas, self._scroll)

        ct = self._scroll
        px = 24

        # Header
        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(header, icon="", title="Évaluation",
                      subtitle="Comparez les performances de vos modèles").pack(side="left")
        ModernButton(header, text="Évaluer les modèles", icon="",
                     style="primary", command=self._on_evaluate,
                     bg=C.BG_MAIN).pack(side="right", pady=6)

        # Results area
        self._results_area = tk.Frame(ct, bg=C.BG_MAIN)
        self._results_area.pack(fill="both", expand=True, padx=px, pady=(16, 24))

        tk.Label(self._results_area,
                 text="Entraînez vos modèles puis cliquez sur « Évaluer »",
                 font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT_DIM).pack(pady=24)

        nav_row = tk.Frame(ct, bg=C.BG_MAIN)
        nav_row.pack(anchor="e", padx=px, pady=(0, 16))
        ModernButton(
            nav_row,
            text="Passer aux modèles",
            style="primary",
            command=lambda: self._navigate_to("models"),
            bg=C.BG_MAIN,
        ).pack(side="right")


    def on_enter(self):
        if self._service.evaluation_results:
            self._show_results()

    def _on_evaluate(self):
        if self._service.trained_models is None:
            show_error("Erreur", "Aucun modèle entraîné.")
            return
        try:
            self._service.run_evaluation()
        except Exception as exc:
            show_error("Erreur", str(exc))
            return
        self._show_results()
        show_info("Succès", "Évaluation terminée !")

    def _show_results(self):
        for w in self._results_area.winfo_children():
            w.destroy()

        comparison = self._service.get_comparison_table()
        results = self._service.evaluation_results
        if comparison is None or comparison.empty:
            return

        task_type = self._service.preprocessing_result.task_type if self._service.preprocessing_result else ""

        # ── Best model highlight ─────────────────────────────────
        best = comparison.iloc[0]
        best_name = best.get("Model", "?")

        trophy = Card(self._results_area, accent_color=C.SUCCESS, pad=16)
        trophy.pack(fill="x", pady=(0, 16))

        trophy_row = tk.Frame(trophy.inner, bg=C.BG_CARD)
        trophy_row.pack(fill="x")

        tk.Label(trophy_row, text="TOP", font=(F.FAM, 11, "bold"),
                 bg=C.BG_CARD, fg=C.WARNING).pack(side="left", padx=(0, 12))

        trophy_text = tk.Frame(trophy_row, bg=C.BG_CARD)
        trophy_text.pack(side="left")

        tk.Label(trophy_text, text="Meilleur modèle", font=F.SMALL,
                 bg=C.BG_CARD, fg=C.TEXT_SEC).pack(anchor="w")
        tk.Label(trophy_text, text=best_name, font=F.H2,
                 bg=C.BG_CARD, fg=C.SUCCESS).pack(anchor="w")

        # Show primary metric
        if task_type == "classification":
            primary_key, primary_label = "accuracy", "Accuracy"
        else:
            primary_key, primary_label = "r2", "R²"

        primary_val = best.get(primary_key, "—")
        display_val = f"{primary_val:.4f}" if isinstance(primary_val, float) else str(primary_val)

        tk.Label(trophy_row, text=display_val, font=(F.FAM, 24, "bold"),
                 bg=C.BG_CARD, fg=C.ACCENT).pack(side="right", padx=(0, 8))
        tk.Label(trophy_row, text=primary_label, font=F.SMALL,
                 bg=C.BG_CARD, fg=C.TEXT_MUTED).pack(side="right", padx=(0, 4))

        # ── Comparison table ─────────────────────────────────────
        tk.Label(self._results_area, text="Tableau comparatif", font=F.H2,
             bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="center", pady=(0, 16))

        cols = list(comparison.columns)
        widths = {c: max(len(c) * 11, 100) for c in cols}
        widths["Model"] = 200

        stv = StyledTreeview(self._results_area, columns=cols,
                              col_widths=widths, height=min(len(comparison), 10))
        stv.pack(fill="x", pady=(0, 16))

        for _, row in comparison.iterrows():
            vals = []
            for v in row:
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v) if v is not None else "—")
            stv.tree.insert("", "end", values=vals)

        # ── Per-model detail tabs ────────────────────────────────
        if not results:
            return

        tk.Label(self._results_area, text="Détails par modèle", font=F.H2,
             bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="center", pady=(0, 16))

        notebook = ttk.Notebook(self._results_area)
        notebook.pack(fill="both", expand=True)

        for name, metrics in results.items():
            tab = tk.Frame(notebook, bg=C.BG_MAIN)
            notebook.add(tab, text=f"  {name}  ")

            if "error" in metrics:
                tk.Label(tab, text=f"❌  {metrics['error']}", font=F.H4,
                         bg=C.BG_MAIN, fg=C.DANGER).pack(padx=24, pady=24)
                continue

            # Metric tiles
            tiles_row = tk.Frame(tab, bg=C.BG_MAIN)
            tiles_row.pack(fill="x", padx=16, pady=(16, 0))

            if task_type == "classification":
                keys = [("Accuracy", "accuracy", C.ACCENT),
                        ("Precision", "precision", C.INFO),
                        ("Recall", "recall", C.SUCCESS),
                        ("F1-Score", "f1_score", C.WARNING)]
            else:
                keys = [("MAE", "mae", C.WARNING),
                        ("MSE", "mse", C.DANGER),
                        ("RMSE", "rmse", C.INFO),
                        ("R²", "r2", C.SUCCESS)]

            for label, key, color in keys:
                val = metrics.get(key, "—")
                display = f"{val:.4f}" if isinstance(val, float) else str(val)
                MetricTile(tiles_row, value=display, label=label, color=color).pack(
                    side="left", fill="x", expand=True, padx=(0, 6))

            # Confusion matrix / report
            if task_type == "classification":
                cm = metrics.get("confusion_matrix")
                report = metrics.get("classification_report", "")

                detail_row = tk.Frame(tab, bg=C.BG_MAIN)
                detail_row.pack(fill="both", expand=True, padx=16, pady=(16, 16))

                if cm:
                    cm_str = "\n".join("  ".join(f"{v:>6}" for v in row) for row in cm)
                    log1 = LogPanel(detail_row, height=max(len(cm) + 1, 4),
                                     label="Matrice de confusion")
                    log1.pack(side="left", fill="both", expand=True, padx=(0, 6))
                    log1.set_content(cm_str)

                if report:
                    log2 = LogPanel(detail_row, height=10, label="Rapport de classification")
                    log2.pack(side="left", fill="both", expand=True, padx=(6, 0))
                    log2.set_content(report)

    def _navigate_to(self, view_name: str) -> None:
        root = self.winfo_toplevel()
        navigate = getattr(root, "navigate_to", None)
        if callable(navigate):
            navigate(view_name)
