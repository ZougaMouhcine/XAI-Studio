"""
XAI Studio — Evaluation & Explainable AI View
==============================================
Modern evaluation dashboard with model registry, comparison, and XAI tools.
"""

import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ui.widgets import (
    C,
    F,
    Card,
    MetricTile,
    ModernButton,
    SectionHeader,
    StyledTreeview,
    LogPanel,
    Badge,
    bind_mousewheel_to,
)
from ui.components.plot_canvas import PlotCanvas, apply_plot_style
from ui.components.dialogs import show_error, show_info
from services.evaluation_controller import EvaluationController
from services.visualization_service import VisualizationService


class EvaluationView(ttk.Frame):
    """Evaluation + XAI dashboard with registry, plots, and reporting."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._controller = EvaluationController()
        self._viz = VisualizationService()
        self._evaluation_cache: dict[str, dict] = {}
        self._active_entry_id: str | None = None
        self._local_feature_indices: list[int] = []
        self._local_values: np.ndarray | None = None
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
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
        SectionHeader(
            header,
            icon="",
            title="Evaluation & Explainable AI",
            subtitle="Analysez, comparez et expliquez vos modeles avec des visuels modernes",
        ).pack(side="left")

        action_row = tk.Frame(header, bg=C.BG_MAIN)
        action_row.pack(side="right")
        ModernButton(action_row, text="Rafraichir", style="secondary", command=self._refresh_registry, bg=C.BG_MAIN).pack(
            side="left", padx=(0, 8), pady=6
        )
        ModernButton(action_row, text="Charger modele", style="secondary", command=self._load_external_model, bg=C.BG_MAIN).pack(
            side="left", padx=(0, 8), pady=6
        )
        ModernButton(action_row, text="Importer artefacts", style="secondary", command=self._import_artifacts, bg=C.BG_MAIN).pack(
            side="left", padx=(0, 8), pady=6
        )
        ModernButton(action_row, text="Evaluer selection", style="primary", command=self._on_evaluate_selected, bg=C.BG_MAIN).pack(
            side="left", pady=6
        )

        # Registry card
        registry_card = Card(ct, accent_color=C.INFO, pad=16)
        registry_card.pack(fill="x", padx=px, pady=(16, 0))

        tk.Label(registry_card.inner, text="Model Registry", font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")
        self._registry_hint = tk.Label(
            registry_card.inner,
            text="Selectionnez un ou plusieurs modeles pour lancer l'evaluation",
            font=F.SMALL,
            bg=C.BG_CARD,
            fg=C.TEXT_MUTED,
        )
        self._registry_hint.pack(anchor="w", pady=(2, 8))

        columns = ("Nom", "Source", "Tache", "Dataset", "Entraine", "Statut")
        widths = {"Nom": 220, "Source": 100, "Tache": 120, "Dataset": 140, "Entraine": 140, "Statut": 90}
        self._registry_table = StyledTreeview(registry_card.inner, columns=columns, col_widths=widths, height=6, selectmode="extended")
        self._registry_table.pack(fill="x")
        self._registry_table.tree.bind("<<TreeviewSelect>>", self._on_registry_select)

        # Active model card
        self._active_card = Card(ct, accent_color=C.ACCENT, pad=16)
        self._active_card.pack(fill="x", padx=px, pady=(16, 0))

        active_row = tk.Frame(self._active_card.inner, bg=C.BG_CARD)
        active_row.pack(fill="x")
        self._active_name = tk.Label(active_row, text="Modele actif: —", font=F.H3, bg=C.BG_CARD, fg=C.TEXT)
        self._active_name.pack(side="left")
        self._active_badge_wrap = tk.Frame(active_row, bg=C.BG_CARD)
        self._active_badge_wrap.pack(side="right")

        self._active_meta = tk.Label(
            self._active_card.inner,
            text="Selectionnez un modele pour afficher les details",
            font=F.SMALL,
            bg=C.BG_CARD,
            fg=C.TEXT_MUTED,
        )
        self._active_meta.pack(anchor="w", pady=(6, 0))

        # Metrics card
        metrics_card = Card(ct, accent_color=C.SUCCESS, pad=16)
        metrics_card.pack(fill="x", padx=px, pady=(16, 0))
        tk.Label(metrics_card.inner, text="Metrics", font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w", pady=(0, 8))
        self._metrics_row = tk.Frame(metrics_card.inner, bg=C.BG_CARD)
        self._metrics_row.pack(fill="x")
        self._metric_tiles = [
            MetricTile(self._metrics_row, value="—", label="Accuracy", color=C.ACCENT),
            MetricTile(self._metrics_row, value="—", label="Precision", color=C.INFO),
            MetricTile(self._metrics_row, value="—", label="Recall", color=C.SUCCESS),
            MetricTile(self._metrics_row, value="—", label="F1", color=C.WARNING),
        ]
        for tile in self._metric_tiles:
            tile.pack(side="left", fill="x", expand=True, padx=(0, 8))

        report_row = tk.Frame(ct, bg=C.BG_MAIN)
        report_row.pack(fill="x", padx=px, pady=(10, 0))
        ModernButton(report_row, text="Generer HTML", style="secondary", command=lambda: self._generate_report("html"), bg=C.BG_MAIN).pack(
            side="left", padx=(0, 8)
        )
        ModernButton(report_row, text="Generer PDF", style="secondary", command=lambda: self._generate_report("pdf"), bg=C.BG_MAIN).pack(
            side="left"
        )

        # Visualization notebook
        tk.Label(ct, text="Visualisations", font=F.H2, bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="w", padx=px, pady=(18, 10))
        self._plot_notebook = ttk.Notebook(ct)
        self._plot_notebook.pack(fill="both", expand=True, padx=px)

        self._build_plot_tabs()

        # Comparison section
        tk.Label(ct, text="Comparaison des modeles", font=F.H2, bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="w", padx=px, pady=(20, 8))
        self._comparison_table = StyledTreeview(ct, columns=("Model", "Task", "Score", "Training Time", "Complexity", "Status"), height=6)
        self._comparison_table.pack(fill="x", padx=px, pady=(0, 16))

        # XAI notebook
        tk.Label(ct, text="Explainable AI", font=F.H2, bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="w", padx=px, pady=(10, 8))
        self._xai_notebook = ttk.Notebook(ct)
        self._xai_notebook.pack(fill="both", expand=True, padx=px, pady=(0, 24))
        self._build_xai_tabs()

    def _build_plot_tabs(self):
        # Classification
        self._tab_classif = tk.Frame(self._plot_notebook, bg=C.BG_MAIN)
        self._plot_notebook.add(self._tab_classif, text="Classification")

        class_row = tk.Frame(self._tab_classif, bg=C.BG_MAIN)
        class_row.pack(fill="both", expand=True, padx=12, pady=12)

        left = tk.Frame(class_row, bg=C.BG_MAIN)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = tk.Frame(class_row, bg=C.BG_MAIN)
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self._cm_canvas = PlotCanvas(left)
        self._cm_canvas.pack(fill="both", expand=True, pady=(0, 10))
        self._roc_canvas = PlotCanvas(left)
        self._roc_canvas.pack(fill="both", expand=True)

        self._pr_canvas = PlotCanvas(right)
        self._pr_canvas.pack(fill="both", expand=True, pady=(0, 10))
        self._report_panel = LogPanel(right, height=12, label="Classification report")
        self._report_panel.pack(fill="both", expand=True)

        # Regression
        self._tab_reg = tk.Frame(self._plot_notebook, bg=C.BG_MAIN)
        self._plot_notebook.add(self._tab_reg, text="Regression")

        reg_row = tk.Frame(self._tab_reg, bg=C.BG_MAIN)
        reg_row.pack(fill="both", expand=True, padx=12, pady=12)

        self._reg_pred_canvas = PlotCanvas(reg_row)
        self._reg_pred_canvas.pack(fill="both", expand=True, pady=(0, 10))
        self._reg_res_canvas = PlotCanvas(reg_row)
        self._reg_res_canvas.pack(fill="both", expand=True, pady=(0, 10))
        self._reg_err_canvas = PlotCanvas(reg_row)
        self._reg_err_canvas.pack(fill="both", expand=True)

        # Clustering
        self._tab_cluster = tk.Frame(self._plot_notebook, bg=C.BG_MAIN)
        self._plot_notebook.add(self._tab_cluster, text="Clustering")

        cl_row = tk.Frame(self._tab_cluster, bg=C.BG_MAIN)
        cl_row.pack(fill="both", expand=True, padx=12, pady=12)

        self._cluster_scatter_canvas = PlotCanvas(cl_row)
        self._cluster_scatter_canvas.pack(fill="both", expand=True, pady=(0, 10))
        self._cluster_dist_canvas = PlotCanvas(cl_row)
        self._cluster_dist_canvas.pack(fill="both", expand=True)

    def _build_xai_tabs(self):
        # Feature importance
        tab_fi = tk.Frame(self._xai_notebook, bg=C.BG_MAIN)
        self._xai_notebook.add(tab_fi, text="Feature Importance")

        fi_controls = tk.Frame(tab_fi, bg=C.BG_MAIN)
        fi_controls.pack(fill="x", padx=12, pady=(12, 0))
        ModernButton(fi_controls, text="Calculer", style="primary", command=self._run_fi, bg=C.BG_MAIN).pack(side="left")
        self._fi_canvas = PlotCanvas(tab_fi)
        self._fi_canvas.pack(fill="both", expand=True, padx=12, pady=12)

        # SHAP
        tab_shap = tk.Frame(self._xai_notebook, bg=C.BG_MAIN)
        self._xai_notebook.add(tab_shap, text="SHAP")

        shap_controls = tk.Frame(tab_shap, bg=C.BG_MAIN)
        shap_controls.pack(fill="x", padx=12, pady=(12, 0))
        tk.Label(shap_controls, text="Plot:", font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))
        self._shap_plot_type = ttk.Combobox(shap_controls, values=["bar", "beeswarm", "waterfall", "force"], state="readonly", width=14)
        self._shap_plot_type.set("bar")
        self._shap_plot_type.pack(side="left", padx=(0, 12))
        tk.Label(shap_controls, text="Instance:", font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))
        self._shap_instance = ttk.Spinbox(shap_controls, from_=0, to=999, width=8)
        self._shap_instance.set(0)
        self._shap_instance.pack(side="left", padx=(0, 12))
        ModernButton(shap_controls, text="Generer SHAP", style="primary", command=self._run_shap, bg=C.BG_MAIN).pack(side="left")
        self._shap_canvas = PlotCanvas(tab_shap)
        self._shap_canvas.pack(fill="both", expand=True, padx=12, pady=12)

        # LIME
        tab_lime = tk.Frame(self._xai_notebook, bg=C.BG_MAIN)
        self._xai_notebook.add(tab_lime, text="LIME")

        lime_controls = tk.Frame(tab_lime, bg=C.BG_MAIN)
        lime_controls.pack(fill="x", padx=12, pady=(12, 0))
        tk.Label(lime_controls, text="Instance:", font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))
        self._lime_instance = ttk.Spinbox(lime_controls, from_=0, to=999, width=8)
        self._lime_instance.set(0)
        self._lime_instance.pack(side="left", padx=(0, 12))
        tk.Label(lime_controls, text="Features:", font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))
        self._lime_nfeat = ttk.Spinbox(lime_controls, from_=5, to=30, width=6)
        self._lime_nfeat.set(10)
        self._lime_nfeat.pack(side="left", padx=(0, 12))
        ModernButton(lime_controls, text="Generer LIME", style="primary", command=self._run_lime, bg=C.BG_MAIN).pack(side="left")
        self._lime_canvas = PlotCanvas(tab_lime)
        self._lime_canvas.pack(fill="both", expand=True, padx=12, pady=12)

        # PDP
        tab_pdp = tk.Frame(self._xai_notebook, bg=C.BG_MAIN)
        self._xai_notebook.add(tab_pdp, text="PDP")

        pdp_controls = tk.Frame(tab_pdp, bg=C.BG_MAIN)
        pdp_controls.pack(fill="x", padx=12, pady=(12, 0))
        tk.Label(pdp_controls, text="Type:", font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))
        self._pdp_type = ttk.Combobox(pdp_controls, values=["1D", "2D"], state="readonly", width=6)
        self._pdp_type.set("1D")
        self._pdp_type.pack(side="left", padx=(0, 12))
        self._pdp_type.bind("<<ComboboxSelected>>", self._on_pdp_type_change)

        tk.Label(pdp_controls, text="Feature 1:", font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))
        self._pdp_feat1 = ttk.Combobox(pdp_controls, state="readonly", width=18)
        self._pdp_feat1.pack(side="left", padx=(0, 12))

        self._pdp_feat2_label = tk.Label(pdp_controls, text="Feature 2:", font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT)
        self._pdp_feat2 = ttk.Combobox(pdp_controls, state="readonly", width=18)

        ModernButton(pdp_controls, text="Generer PDP", style="primary", command=self._run_pdp, bg=C.BG_MAIN).pack(side="left", padx=(12, 0))
        self._pdp_canvas = PlotCanvas(tab_pdp)
        self._pdp_canvas.pack(fill="both", expand=True, padx=12, pady=12)

        # Local explanation
        tab_local = tk.Frame(self._xai_notebook, bg=C.BG_MAIN)
        self._xai_notebook.add(tab_local, text="Explication locale")

        local_controls = tk.Frame(tab_local, bg=C.BG_MAIN)
        local_controls.pack(fill="x", padx=12, pady=(12, 0))
        tk.Label(local_controls, text="Instance:", font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))
        self._local_instance = ttk.Spinbox(local_controls, from_=0, to=999, width=8)
        self._local_instance.set(0)
        self._local_instance.pack(side="left", padx=(0, 12))
        ModernButton(local_controls, text="Charger", style="secondary", command=self._load_local_instance, bg=C.BG_MAIN).pack(side="left")
        ModernButton(local_controls, text="Mettre a jour explication", style="primary", command=self._run_local_explanation, bg=C.BG_MAIN).pack(
            side="left", padx=(8, 0)
        )

        self._local_pred_label = tk.Label(tab_local, text="Prediction: —", font=F.H4, bg=C.BG_MAIN, fg=C.TEXT)
        self._local_pred_label.pack(anchor="w", padx=12, pady=(10, 6))

        self._local_slider_frame = tk.Frame(tab_local, bg=C.BG_MAIN)
        self._local_slider_frame.pack(fill="x", padx=12)

        self._local_canvas = PlotCanvas(tab_local)
        self._local_canvas.pack(fill="both", expand=True, padx=12, pady=(12, 12))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_enter(self):
        self._refresh_registry()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def _refresh_registry(self):
        entries = self._controller.refresh_registry()
        self._registry_table.tree.delete(*self._registry_table.tree.get_children())

        for entry in entries:
            status = "Pret" if entry.X_test is not None else "Incomplet"
            trained_at = entry.trained_at[:19] if entry.trained_at else "—"
            self._registry_table.tree.insert(
                "",
                "end",
                values=(entry.name, entry.source, entry.task_type, entry.dataset_name or "—", trained_at, status),
                tags=(entry.entry_id,),
            )

        if entries and not self._active_entry_id:
            self._set_active_entry(entries[0].entry_id)

    def _on_registry_select(self, _=None):
        sel = self._registry_table.tree.selection()
        if not sel:
            return
        entry_id = self._registry_table.tree.item(sel[0]).get("tags", [None])[0]
        if entry_id:
            self._set_active_entry(entry_id)

    def _set_active_entry(self, entry_id: str):
        entry = self._controller.registry.get(entry_id)
        if not entry:
            return
        self._active_entry_id = entry_id
        self._active_name.configure(text=f"Modele actif: {entry.name}")

        for w in self._active_badge_wrap.winfo_children():
            w.destroy()
        Badge(self._active_badge_wrap, text=entry.task_type.upper(), color=C.ACCENT).pack()

        meta = entry.metadata or {}
        algo = meta.get("algorithm", "—")
        features = meta.get("n_features", "—")
        self._active_meta.configure(text=f"{algo} · Features: {features} · Source: {entry.source}")

        feat_names = meta.get("feature_names", [])
        self._pdp_feat1["values"] = feat_names
        self._pdp_feat2["values"] = feat_names
        if feat_names:
            self._pdp_feat1.set(feat_names[0])
            if len(feat_names) > 1:
                self._pdp_feat2.set(feat_names[1])

    def _load_external_model(self):
        filepath = filedialog.askopenfilename(
            title="Charger un modele",
            filetypes=[("Model Files", "*.pkl *.joblib"), ("Pickle", "*.pkl"), ("Joblib", "*.joblib")],
        )
        if not filepath:
            return
        try:
            self._controller.register_external_model(filepath)
            show_info("Succes", "Modele charge avec succes")
            self._refresh_registry()
        except Exception as exc:
            show_error("Erreur", str(exc))

    def _import_artifacts(self):
        if not self._active_entry_id:
            show_error("Erreur", "Selectionnez un modele pour importer des artefacts")
            return
        filepath = filedialog.askopenfilename(
            title="Importer artefacts (.npz)",
            filetypes=[("NumPy NPZ", "*.npz"), ("All Files", "*.*")],
        )
        if not filepath:
            return

        try:
            data = np.load(filepath, allow_pickle=True)
            X_test = data.get("X_test")
            y_test = data.get("y_test")
            y_pred = data.get("y_pred")
            y_proba = data.get("y_proba")
            self._controller.attach_artifacts(self._active_entry_id, X_test=X_test, y_test=y_test, y_pred=y_pred, y_proba=y_proba)
            show_info("Succes", "Artefacts importes")
            self._refresh_registry()
        except Exception as exc:
            show_error("Erreur", str(exc))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _on_evaluate_selected(self):
        entry_ids = self._selected_entry_ids()
        if not entry_ids:
            show_error("Erreur", "Selectionnez un ou plusieurs modeles")
            return

        def _compute():
            return self._controller.evaluate_models(entry_ids)

        def _done(results):
            self._evaluation_cache.update(results)
            active_id = self._active_entry_id or entry_ids[0]
            self._render_active(active_id)
            self._render_comparison(entry_ids)
            show_info("Succes", "Evaluation terminee")

        self._run_async(_compute, _done)

    def _render_active(self, entry_id: str):
        entry = self._controller.registry.get(entry_id)
        if not entry:
            return
        metrics = self._evaluation_cache.get(entry_id) or self._controller.evaluate_models([entry_id]).get(entry_id)
        if not metrics or "error" in metrics:
            show_error("Erreur", metrics.get("error", "Evaluation invalide"))
            return

        self._set_active_entry(entry_id)
        self._render_metrics(entry.task_type, metrics)
        self._render_plots(entry, metrics)

    def _render_metrics(self, task_type: str, metrics: dict):
        if task_type == "classification":
            labels = [("Accuracy", "accuracy", C.ACCENT), ("Precision", "precision", C.INFO), ("Recall", "recall", C.SUCCESS), ("F1", "f1_score", C.WARNING)]
        elif task_type == "regression":
            labels = [("RMSE", "rmse", C.WARNING), ("MAE", "mae", C.INFO), ("R2", "r2", C.SUCCESS), ("MAPE", "mape", C.ACCENT)]
        else:
            labels = [("Silhouette", "silhouette", C.SUCCESS), ("Davies-B", "davies_bouldin", C.WARNING), ("Calinski", "calinski_harabasz", C.INFO), ("Score", "score_global", C.ACCENT)]

        for tile, (label, key, color) in zip(self._metric_tiles, labels):
            val = metrics.get(key, "—")
            display = f"{val:.4f}" if isinstance(val, float) else str(val)
            tile.set(display, label=label)
            tile._value_lbl.configure(fg=color)

    def _render_plots(self, entry, metrics):
        if entry.task_type == "classification":
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                payload = self._viz.render_confusion_matrix(cm, title="Confusion Matrix")
                self._cm_canvas.update_figure(payload.figure)

            roc = metrics.get("roc_curve")
            if roc:
                payload = self._viz.render_roc_curve(roc["fpr"], roc["tpr"], label=entry.name)
                self._roc_canvas.update_figure(payload.figure)

            pr = metrics.get("pr_curve")
            if pr:
                payload = self._viz.render_precision_recall_curve(pr["recall"], pr["precision"], label=entry.name)
                self._pr_canvas.update_figure(payload.figure)

            report = metrics.get("classification_report", "")
            self._report_panel.set_content(report)

        elif entry.task_type == "regression":
            y_true = entry.y_test
            y_pred = metrics.get("y_pred")
            if y_true is not None and y_pred is not None:
                payload = self._viz.render_predicted_vs_actual(y_true, y_pred)
                self._reg_pred_canvas.update_figure(payload.figure)
                payload = self._viz.render_residuals(y_true, y_pred)
                self._reg_res_canvas.update_figure(payload.figure)
                payload = self._viz.render_error_distribution(y_true, y_pred)
                self._reg_err_canvas.update_figure(payload.figure)

        elif entry.task_type == "clustering":
            labels = metrics.get("labels")
            if entry.X_test is not None and labels is not None:
                payload = self._viz.render_cluster_scatter(entry.X_test, labels)
                self._cluster_scatter_canvas.update_figure(payload.figure)
                payload = self._viz.render_cluster_distribution(labels)
                self._cluster_dist_canvas.update_figure(payload.figure)

    def _render_comparison(self, entry_ids: list[str]):
        df = self._controller.comparison_table(entry_ids)
        self._comparison_table.tree.delete(*self._comparison_table.tree.get_children())
        if df is None or df.empty:
            return
        for _, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v) if v is not None else "—")
            self._comparison_table.tree.insert("", "end", values=vals)

    def _selected_entry_ids(self) -> list[str]:
        ids = []
        for sel in self._registry_table.tree.selection():
            entry_id = self._registry_table.tree.item(sel).get("tags", [None])[0]
            if entry_id:
                ids.append(entry_id)
        return ids

    def _generate_report(self, fmt: str):
        if not self._active_entry_id:
            show_error("Erreur", "Aucun modele actif")
            return

        filetypes = [("HTML", "*.html")] if fmt == "html" else [("PDF", "*.pdf")]
        filepath = filedialog.asksaveasfilename(
            title="Exporter rapport",
            defaultextension=f".{fmt}",
            filetypes=filetypes,
        )
        if not filepath:
            return

        def _compute():
            return self._controller.generate_report(self._active_entry_id, filepath, fmt=fmt)

        def _done(_):
            show_info("Succes", f"Rapport genere: {filepath}")

        self._run_async(_compute, _done)

    # ------------------------------------------------------------------
    # XAI actions
    # ------------------------------------------------------------------
    def _run_fi(self):
        entry_id = self._active_entry_id
        if not entry_id:
            show_error("Erreur", "Aucun modele actif")
            return

        def _compute():
            return self._controller.feature_importance(entry_id)

        def _done(result):
            _, fig = result
            self._fi_canvas.update_figure(fig)

        self._run_async(_compute, _done)

    def _run_shap(self):
        entry_id = self._active_entry_id
        if not entry_id:
            show_error("Erreur", "Aucun modele actif")
            return
        plot_type = self._shap_plot_type.get()
        instance_idx = int(self._shap_instance.get())

        def _compute():
            return self._controller.shap_plot(entry_id, plot_type, instance_idx)

        def _done(fig):
            self._shap_canvas.update_figure(fig)

        self._run_async(_compute, _done)

    def _run_lime(self):
        entry_id = self._active_entry_id
        if not entry_id:
            show_error("Erreur", "Aucun modele actif")
            return
        instance_idx = int(self._lime_instance.get())
        num_features = int(self._lime_nfeat.get())

        def _compute():
            return self._controller.lime_plot(entry_id, instance_idx, num_features)

        def _done(fig):
            self._lime_canvas.update_figure(fig)

        self._run_async(_compute, _done)

    def _on_pdp_type_change(self, _=None):
        if self._pdp_type.get() == "2D":
            self._pdp_feat2_label.pack(side="left", padx=(0, 8))
            self._pdp_feat2.pack(side="left", padx=(0, 12))
        else:
            self._pdp_feat2_label.pack_forget()
            self._pdp_feat2.pack_forget()

    def _run_pdp(self):
        entry_id = self._active_entry_id
        if not entry_id:
            show_error("Erreur", "Aucun modele actif")
            return

        feat1 = self._pdp_feat1.get()
        feat2 = self._pdp_feat2.get() if self._pdp_type.get() == "2D" else None

        entry = self._controller.registry.get(entry_id)
        if not entry:
            show_error("Erreur", "Modele introuvable")
            return
        feature_names = entry.metadata.get("feature_names", [])
        if feat1 not in feature_names:
            show_error("Erreur", "Feature invalide")
            return
        idx1 = feature_names.index(feat1)
        if feat2:
            if feat2 not in feature_names:
                show_error("Erreur", "Feature invalide")
                return
            idx2 = feature_names.index(feat2)
            feature_idxs = (idx1, idx2)
            names = [feat1, feat2]
        else:
            feature_idxs = idx1
            names = [feat1]

        def _compute():
            return self._controller.pdp_plot(entry_id, feature_idxs, feature_names=names)

        def _done(fig):
            self._pdp_canvas.update_figure(fig)

        self._run_async(_compute, _done)

    def _load_local_instance(self):
        entry = self._controller.registry.get(self._active_entry_id or "")
        if not entry or entry.X_test is None:
            show_error("Erreur", "X_test indisponible")
            return

        idx = min(int(self._local_instance.get()), len(entry.X_test) - 1)
        instance = np.array(entry.X_test[idx])
        self._local_values = instance.copy()
        self._build_local_sliders(entry, instance)
        self._update_local_prediction(entry)

    def _build_local_sliders(self, entry, instance):
        for w in self._local_slider_frame.winfo_children():
            w.destroy()

        feature_names = entry.metadata.get("feature_names", [])
        if not feature_names:
            return

        max_features = min(8, len(feature_names))
        self._local_feature_indices = list(range(max_features))

        X = entry.X_test
        for idx in self._local_feature_indices:
            fname = feature_names[idx]
            col = X[:, idx]
            min_val = float(np.nanmin(col))
            max_val = float(np.nanmax(col))
            row = tk.Frame(self._local_slider_frame, bg=C.BG_MAIN)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=fname, font=F.SMALL, bg=C.BG_MAIN, fg=C.TEXT).pack(side="left", padx=(0, 8))
            scale = tk.Scale(
                row,
                from_=min_val,
                to=max_val,
                orient="horizontal",
                resolution=(max_val - min_val) / 100 if max_val != min_val else 1,
                length=260,
                bg=C.BG_MAIN,
                fg=C.TEXT,
                highlightthickness=0,
                troughcolor=C.BG_INPUT,
                command=lambda v, i=idx: self._on_local_slider(i, v),
            )
            scale.set(instance[idx])
            scale.pack(side="right", fill="x", expand=True)

    def _on_local_slider(self, idx, value):
        if self._local_values is None:
            return
        self._local_values[idx] = float(value)
        entry = self._controller.registry.get(self._active_entry_id or "")
        if entry:
            self._update_local_prediction(entry)

    def _update_local_prediction(self, entry):
        if self._local_values is None:
            return
        model = entry.model or self._controller.registry.ensure_loaded(entry).model
        if model is None:
            return
        pred = model.predict(self._local_values.reshape(1, -1))
        text = f"Prediction: {pred[0]}"
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(self._local_values.reshape(1, -1))
                text = f"Prediction: {pred[0]} · Proba: {np.max(proba):.3f}"
            except Exception:
                pass
        self._local_pred_label.configure(text=text)

    def _run_local_explanation(self):
        if self._active_entry_id is None or self._local_values is None:
            show_error("Erreur", "Chargez une instance")
            return

        def _compute():
            return self._controller.local_explanation(self._active_entry_id, self._local_values)

        def _done(exp):
            fig = self._plot_local_contributions(exp.contributions)
            self._local_canvas.update_figure(fig)

        self._run_async(_compute, _done)

    def _plot_local_contributions(self, contributions):
        apply_plot_style()
        names = [c[0] for c in contributions][:12]
        vals = [c[1] for c in contributions][:12]
        colors = ["#ef4444" if v < 0 else "#10b981" for v in vals]
        fig = plt.figure(figsize=(7, 4.5), facecolor=C.BG_CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#111c2e")
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(0, color=C.TEXT_DIM, linestyle="--", linewidth=0.8)
        ax.set_title("Contribution des features")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Async helper
    # ------------------------------------------------------------------
    def _run_async(self, fn, on_done):
        def _done(result):
            self.after(0, lambda: on_done(result))

        def _err(exc):
            self.after(0, lambda: show_error("Erreur", str(exc)))

        self._controller.run_async(fn, _done, _err)
