"""
XAI Studio — Preprocessing View (v2 — Dashboard Design)
=========================================================
Configure and run the preprocessing pipeline with a polished card layout.
"""

import tkinter as tk
from tkinter import ttk

from ui.widgets import C, F, Card, MetricTile, ModernButton, SectionHeader, PipelineStep, LogPanel
from ui.components.dialogs import show_error, show_info
from services.pipeline_service import PipelineService


class PreprocessingView(ttk.Frame):
    """Preprocessing configuration and results dashboard."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._target_var = tk.StringVar()
        self._test_size_var = tk.StringVar(value="0.2")
        self._random_state_var = tk.StringVar(value="42")
        self._build()

    def _build(self):
        # Scrollable wrapper
        canvas = tk.Canvas(self, bg=C.BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll = tk.Frame(canvas, bg=C.BG_MAIN)
        self._scroll.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._scroll, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        ct = self._scroll
        px = 28

        # ── Header ───────────────────────────────────────────────
        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(header, icon="⚙", title="Préprocessing",
                      subtitle="Configurez et lancez le pipeline automatique").pack(side="left")

        # ── Config card ──────────────────────────────────────────
        cfg = Card(ct, accent_color=C.ACCENT, pad=20)
        cfg.pack(fill="x", padx=px, pady=(16, 0))
        inner = cfg.inner

        tk.Label(inner, text="Configuration du pipeline", font=F.H3,
                 bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w", pady=(0, 16))

        # Grid for parameters
        grid = tk.Frame(inner, bg=C.BG_CARD)
        grid.pack(fill="x")

        # Target column
        tk.Label(grid, text="Colonne cible", font=F.H4,
                 bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=0, sticky="w", padx=(0, 12), pady=6)
        self._target_combo = ttk.Combobox(grid, textvariable=self._target_var,
                                           state="readonly", width=28)
        self._target_combo.grid(row=0, column=1, sticky="w", padx=(0, 32), pady=6)

        # Test size
        tk.Label(grid, text="Test size", font=F.H4,
                 bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=2, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(grid, textvariable=self._test_size_var, width=8).grid(
            row=0, column=3, sticky="w", pady=6)

        # Random state
        tk.Label(grid, text="Random state", font=F.H4,
                 bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(grid, textvariable=self._random_state_var, width=8).grid(
            row=1, column=1, sticky="w", pady=6)

        # Run button
        btn_row = tk.Frame(inner, bg=C.BG_CARD)
        btn_row.pack(fill="x", pady=(20, 0))

        ModernButton(btn_row, text="Lancer le préprocessing", icon="▶",
                     style="primary", command=self._on_run,
                     bg=C.BG_CARD).pack(side="left")

        # ── Results area ─────────────────────────────────────────
        self._results_area = tk.Frame(ct, bg=C.BG_MAIN)
        self._results_area.pack(fill="both", expand=True, padx=px, pady=(16, 24))

        # Empty state
        self._empty = tk.Label(self._results_area,
                                text="Les résultats apparaîtront ici après l'exécution du pipeline",
                                font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT_DIM)
        self._empty.pack(pady=40)

    # ──────────────────────────────────────────────────────────────
    def on_enter(self):
        columns = self._service.get_columns()
        self._target_combo["values"] = columns
        if self._service.target_column and self._service.target_column in columns:
            self._target_var.set(self._service.target_column)
        elif columns:
            self._target_var.set(columns[-1])

    def _on_run(self):
        target = self._target_var.get()
        if not target:
            show_error("Erreur", "Veuillez sélectionner une colonne cible.")
            return
        try:
            test_size = float(self._test_size_var.get())
            random_state = int(self._random_state_var.get())
        except ValueError:
            show_error("Erreur", "Paramètres invalides.")
            return
        if not (0 < test_size < 1):
            show_error("Erreur", "test_size doit être entre 0 et 1.")
            return

        self._service.set_target_column(target)
        try:
            result = self._service.run_preprocessing(test_size=test_size, random_state=random_state)
        except Exception as exc:
            show_error("Erreur", str(exc))
            return

        self._show_results(result)
        show_info("Succès", "Préprocessing terminé avec succès !")

    def _show_results(self, result):
        for w in self._results_area.winfo_children():
            w.destroy()

        s = result.summary

        # ── Metrics row ──────────────────────────────────────────
        tk.Label(self._results_area, text="Résultats", font=F.H2,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="w", pady=(0, 12))

        metrics_row = tk.Frame(self._results_area, bg=C.BG_MAIN)
        metrics_row.pack(fill="x", pady=(0, 16))

        tiles = [
            ("", s["task_type"].upper(), "Type de tâche", C.ACCENT),
            ("", str(s["features_count"]), "Features finales", C.INFO),
            ("", str(s["train_size"]), "Échantillons train", C.SUCCESS),
            ("", str(s["test_size"]), "Échantillons test", C.WARNING),
        ]
        for icon, val, lbl, color in tiles:
            MetricTile(metrics_row, icon=icon, value=val, label=lbl, color=color).pack(
                side="left", fill="x", expand=True, padx=(0, 8))

        # ── Pipeline steps card ──────────────────────────────────
        steps_card = Card(self._results_area, accent_color=C.SUCCESS, pad=16)
        steps_card.pack(fill="x", pady=(0, 16))

        tk.Label(steps_card.inner, text="Pipeline exécuté", font=F.H3,
                 bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w", pady=(0, 12))

        steps = [
            ("Imputation valeurs manquantes (médiane / mode)", "done"),
            ("Encodage cible (LabelEncoder)" if result.label_encoder else "Encodage cible (non nécessaire)",
             "done" if result.label_encoder else "pending"),
            (f"One-Hot Encoding ({s['categorical_original']} colonnes)" if s["categorical_original"] > 0
             else "One-Hot Encoding (aucune colonne catégorielle)",
             "done" if s["categorical_original"] > 0 else "pending"),
            (f"StandardScaler ({s['features_count']} features)", "done"),
            (f"Train/Test Split ({s['train_size']}/{s['test_size']})", "done"),
        ]
        for text, status in steps:
            PipelineStep(steps_card.inner, text=text, status=status,
                         bg=C.BG_CARD).pack(anchor="w", pady=3)

        # ── Feature names panel ──────────────────────────────────
        features_log = LogPanel(self._results_area, height=5,
                                 label=f"Features finales ({len(result.feature_names)})")
        features_log.pack(fill="both", expand=True, pady=(0, 16))
        features_log.set_content(", ".join(result.feature_names))
