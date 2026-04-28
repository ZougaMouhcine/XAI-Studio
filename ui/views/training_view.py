"""
XAI Studio — Training View (v2 — Dashboard Design)
=====================================================
Select and train ML models with progress feedback and results table.
"""

import ast
import threading
import tkinter as tk
from tkinter import ttk

from ui.widgets import C, F, Card, ModernButton, SectionHeader, StyledTreeview, Badge, bind_mousewheel_to
from ui.components.dialogs import show_error, show_info, ProgressDialog
from services.pipeline_service import PipelineService


class TrainingView(ttk.Frame):
    """Model selection and training dashboard."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._check_vars: dict[str, tk.BooleanVar] = {}
        self._params_entries: dict[str, ttk.Entry] = {}
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

        # ── Header ───────────────────────────────────────────────
        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(header, icon="", title="Entraînement",
                      subtitle="Sélectionnez les modèles et lancez l'entraînement").pack(side="left")

        # ── Model selection card ──────────────────────────────────
        self._sel_card = Card(ct, accent_color=C.INFO, pad=16)
        self._sel_card.pack(fill="x", padx=px, pady=(16, 0))

        top = tk.Frame(self._sel_card.inner, bg=C.BG_CARD)
        top.pack(fill="x", pady=(0, 16))

        tk.Label(top, text="Modèles disponibles", font=F.H3,
                 bg=C.BG_CARD, fg=C.TEXT).pack(side="left")

        self._task_badge_frame = tk.Frame(top, bg=C.BG_CARD)
        self._task_badge_frame.pack(side="left", padx=(12, 0))

        self._checks_frame = tk.Frame(self._sel_card.inner, bg=C.BG_CARD)
        self._checks_frame.pack(fill="x")

        self._no_data_label = tk.Label(
            self._checks_frame,
            text="⚠  Chargez les données et effectuez le préprocessing d'abord",
            font=F.BODY, bg=C.BG_CARD, fg=C.TEXT_MUTED)
        self._no_data_label.pack(anchor="w", pady=8)

        # Button row
        btn_row = tk.Frame(self._sel_card.inner, bg=C.BG_CARD)
        btn_row.pack(fill="x", pady=(16, 0))

        ModernButton(btn_row, text="Tout sélectionner", style="ghost",
                     command=self._select_all, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(btn_row, text="Tout désélectionner", style="ghost",
                     command=self._deselect_all, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(btn_row, text="Entraîner les modèles", icon="",
                     style="primary", command=self._on_train,
                     bg=C.BG_CARD).pack(side="right")

        # ── Results ──────────────────────────────────────────────
        self._results_area = tk.Frame(ct, bg=C.BG_MAIN)
        self._results_area.pack(fill="both", expand=True, padx=px, pady=(16, 24))

    # ──────────────────────────────────────────────────────────────
    def on_enter(self):
        model_names = self._service.get_model_names()

        for w in self._checks_frame.winfo_children():
            w.destroy()
        for w in self._task_badge_frame.winfo_children():
            w.destroy()
        self._check_vars.clear()
        self._params_entries.clear()

        if not model_names:
            tk.Label(self._checks_frame,
                     text="⚠  Chargez les données et effectuez le préprocessing d'abord",
                     font=F.BODY, bg=C.BG_CARD, fg=C.TEXT_MUTED).pack(anchor="w", pady=8)
            return

        task = self._service.preprocessing_result.task_type if self._service.preprocessing_result else ""
        Badge(self._task_badge_frame, text=task.upper(),
              color=C.ACCENT).pack(side="left")

        # Model checkboxes + params (2 columns)
        grid = tk.Frame(self._checks_frame, bg=C.BG_CARD)
        grid.pack(fill="x")

        registry = self._service.get_model_registry()
        n_cols = 2
        for idx, name in enumerate(model_names):
            var = tk.BooleanVar(value=True)
            self._check_vars[name] = var

            cb_frame = tk.Frame(grid, bg=C.BG_CARD)
            cb_frame.grid(row=idx // n_cols, column=idx % n_cols,
                          sticky="ew", padx=(0, 20), pady=8)

            cb_frame.columnconfigure(1, weight=1)

            cb = ttk.Checkbutton(cb_frame, text=name, variable=var,
                                  style="Card.TCheckbutton")
            cb.grid(row=0, column=0, sticky="w", padx=(0, 10))

            defaults = registry.get(name, {}).get("default_params", {})
            params_var = tk.StringVar(value="")
            entry = ttk.Entry(cb_frame, textvariable=params_var, width=50)
            entry.grid(row=0, column=1, sticky="ew")
            self._params_entries[name] = entry

            tk.Label(
                cb_frame,
                text=f"format: key=value, key2=value2 | profil suggéré: {self._format_params(defaults) or 'sklearn defaults'}",
                bg=C.BG_CARD,
                fg=C.TEXT_DIM,
                font=F.TINY,
            ).grid(row=1, column=1, sticky="w", pady=(2, 0))

    def _select_all(self):
        for v in self._check_vars.values():
            v.set(True)

    def _deselect_all(self):
        for v in self._check_vars.values():
            v.set(False)

    def _on_train(self):
        selected = [n for n, v in self._check_vars.items() if v.get()]
        if not selected:
            show_error("Erreur", "Sélectionnez au moins un modèle.")
            return
        if self._service.preprocessing_result is None:
            show_error("Erreur", "Effectuez le préprocessing d'abord.")
            return

        try:
            model_params_map = self._collect_model_params(selected)
        except ValueError as exc:
            show_error("Erreur paramètres", str(exc))
            return

        dlg = ProgressDialog(self.winfo_toplevel(), "Entraînement en cours…", total=len(selected))

        def progress_cb(cur, tot, name):
            self.after(0, lambda: dlg.update_progress(cur, f"Terminé : {name}"))

        def thread():
            try:
                self._service.run_training(
                    selected_models=selected,
                    model_params_map=model_params_map,
                    progress_callback=progress_cb,
                )
                self.after(0, lambda: self._done(dlg))
            except Exception as exc:
                self.after(0, lambda: self._fail(dlg, exc))

        threading.Thread(target=thread, daemon=True).start()

    def _done(self, dlg):
        dlg.close()
        self._show_results()
        show_info("Succès", "Entraînement terminé !")

    def _fail(self, dlg, exc):
        dlg.close()
        show_error("Erreur", str(exc))

    def _show_results(self):
        for w in self._results_area.winfo_children():
            w.destroy()

        models = self._service.trained_models
        if not models:
            return

        tk.Label(self._results_area, text="Résultats de l'entraînement", font=F.H2,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="center", pady=(0, 16))

        cols = ("Modèle", "Statut", "Temps (s)", "Classe")
        widths = {"Modèle": 200, "Statut": 100, "Temps (s)": 100, "Classe": 280}
        stv = StyledTreeview(self._results_area, columns=cols,
                              col_widths=widths, height=min(len(models), 10))
        stv.pack(fill="x")

        for name, entry in models.items():
            if entry.get("model") is not None:
                vals = (name, "✅ OK", f"{entry['training_time']:.3f}",
                        type(entry["model"]).__name__)
            else:
                vals = (name, "❌ Échec", "—", entry.get("error", "?"))
            stv.tree.insert("", "end", values=vals)

        ok_count = sum(1 for e in models.values() if e.get("model"))
        tk.Label(self._results_area,
                 text=f"{ok_count}/{len(models)} modèles entraînés avec succès",
                 font=F.H4, bg=C.BG_MAIN, fg=C.ACCENT).pack(anchor="center", pady=(16, 0))

    @staticmethod
    def _format_params(params: dict) -> str:
        if not params:
            return ""
        parts = [f"{k}={repr(v)}" for k, v in params.items()]
        return ", ".join(parts)

    def _collect_model_params(self, selected_models: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name in selected_models:
            entry = self._params_entries.get(name)
            if entry is None:
                out[name] = {}
                continue

            raw = entry.get().strip()
            if not raw:
                out[name] = {}
                continue

            parsed: dict[str, object] = {}
            chunks = [c.strip() for c in raw.split(",") if c.strip()]
            for chunk in chunks:
                if "=" not in chunk:
                    raise ValueError(f"Paramètre invalide pour '{name}': {chunk}")
                k, v = chunk.split("=", 1)
                key = k.strip()
                value_str = v.strip()
                if not key:
                    raise ValueError(f"Nom de paramètre vide pour '{name}'.")
                try:
                    value = ast.literal_eval(value_str)
                except Exception:
                    value = value_str
                parsed[key] = value

            out[name] = parsed

        return out
