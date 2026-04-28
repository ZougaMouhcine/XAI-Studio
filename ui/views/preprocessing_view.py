"""Professional modular preprocessing workspace UI."""

import json
import tkinter as tk
from tkinter import ttk

from ui.widgets import C, F, Card, ModernButton, SectionHeader, StyledTreeview, LogPanel, bind_mousewheel_to
from ui.components.dialogs import (
    ask_export_python_file,
    ask_open_pipeline_file,
    ask_save_pipeline_file,
    show_error,
    show_info,
)
from services.pipeline_service import PipelineService


class PreprocessingView(ttk.Frame):
    """Professional preprocessing module for interactive ML workflows."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._catalog = self._service.get_preprocessing_catalog()
        self._viz_img = None

        self._target_var = tk.StringVar()
        self._test_size_var = tk.StringVar(value="0.2")
        self._random_state_var = tk.StringVar(value="42")

        self._category_var = tk.StringVar(value="data_cleaning")
        self._method_var = tk.StringVar()
        self._options_var = tk.StringVar(value="{}")
        self._status_var = tk.StringVar(value="Prêt")

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
        self._field_width = 28

        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(
            header,
            title="Préprocessing Pro",
            subtitle="Module interactif, modulaire et extensible pour ML professionnel",
            icon="",
        ).pack(side="left")

        top_card = Card(ct, accent_color=C.ACCENT, pad=16)
        top_card.pack(fill="x", padx=px, pady=(16, 0))

        self._build_controls(top_card.inner)

        layout = tk.Frame(ct, bg=C.BG_MAIN)
        layout.pack(fill="both", expand=True, padx=px, pady=(16, 24))
        layout.columnconfigure(0, weight=1, uniform="bottom_cards")
        layout.columnconfigure(1, weight=1, uniform="bottom_cards")

        left = tk.Frame(layout, bg=C.BG_MAIN)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        right = tk.Frame(layout, bg=C.BG_MAIN)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_pipeline_panel(left)
        self._build_preview_panel(right)

        status = tk.Label(ct, textvariable=self._status_var, font=F.SMALL, bg=C.BG_MAIN, fg=C.TEXT_SEC)
        status.pack(anchor="w", padx=px, pady=(0, 24))

        self._on_category_change()

    def _build_controls(self, parent):
        # Layout: label / input pairs across columns 0..7. Reserve column 7 for right-aligned actions.
        for i in range(8):
            parent.columnconfigure(i, weight=0)
        parent.columnconfigure(7, weight=1)

        tk.Label(parent, text="Cible", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=0, column=0, sticky="w")
        self._target_combo = ttk.Combobox(parent, textvariable=self._target_var, width=self._field_width, state="readonly")
        self._target_combo.grid(row=0, column=1, padx=(8, 16), sticky="w")

        tk.Label(parent, text="Test size", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=0, column=2, sticky="w")
        ttk.Entry(parent, textvariable=self._test_size_var, width=self._field_width).grid(row=0, column=3, padx=(8, 16), sticky="w")

        tk.Label(parent, text="Random state", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=0, column=4, sticky="w")
        ttk.Entry(parent, textvariable=self._random_state_var, width=self._field_width).grid(row=0, column=5, padx=(8, 8), sticky="w")

        ModernButton(parent, text="Préparer entraînement", style="primary", command=self._prepare_training, bg=C.BG_CARD).grid(row=0, column=7, sticky="e")

        tk.Label(parent, text="Catégorie", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=0, sticky="w", pady=(14, 0))
        cat_combo = ttk.Combobox(parent, textvariable=self._category_var, width=self._field_width, state="readonly", values=list(self._catalog.keys()))
        cat_combo.grid(row=1, column=1, padx=(8, 16), pady=(14, 0), sticky="w")
        cat_combo.bind("<<ComboboxSelected>>", lambda _: self._on_category_change())

        tk.Label(parent, text="Méthode", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=2, sticky="w", pady=(14, 0))
        self._method_combo = ttk.Combobox(parent, textvariable=self._method_var, width=self._field_width, state="readonly")
        self._method_combo.grid(row=1, column=3, columnspan=2, padx=(8, 16), pady=(14, 0), sticky="w")

        tk.Label(parent, text="Colonnes", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=2, column=0, sticky="nw", pady=(14, 0))
        self._columns_list = tk.Listbox(parent, selectmode="extended", height=5, exportselection=False, width=self._field_width)
        # Keep the columns listbox the same character width as other fields (no extra columnspan)
        self._columns_list.grid(row=2, column=1, columnspan=1, sticky="w", pady=(14, 0), padx=(8, 16))
        tk.Label(
            parent,
            text="Sélection multiple autorisée: Ctrl / Shift",
            font=F.TINY,
            bg=C.BG_CARD,
            fg=C.TEXT_MUTED,
        ).grid(row=3, column=1, columnspan=2, sticky="w", padx=(8, 16), pady=(2, 0))

        tk.Label(parent, text="Options JSON", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=2, column=3, sticky="nw", pady=(14, 0))
        ttk.Entry(parent, textvariable=self._options_var, width=self._field_width).grid(row=2, column=4, sticky="w", pady=(14, 0), padx=(8, 0))

        actions = tk.Frame(parent, bg=C.BG_CARD)
        actions.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(14, 0))

        ModernButton(actions, text="Exécuter étape", style="primary", command=self._run_single_step, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text="Ajouter au pipeline", style="secondary", command=self._add_step, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text="Exécuter pipeline", style="secondary", command=self._run_pipeline, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text="Undo", style="ghost", command=self._undo, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text="Redo", style="ghost", command=self._redo, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text="Auto recommandations", style="ghost", command=self._recommend, bg=C.BG_CARD).pack(side="right")

    def _build_pipeline_panel(self, parent):
        card = Card(parent, accent_color=C.ACCENT, pad=16)
        card.pack(fill="both", expand=True)

        tk.Label(card.inner, text="Pipeline dynamique", font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        cols = ("#", "Catégorie", "Méthode", "Colonnes")
        widths = {"#": 40, "Catégorie": 160, "Méthode": 180, "Colonnes": 260}
        self._pipeline_table = StyledTreeview(card.inner, columns=cols, col_widths=widths, height=8, bg=C.BG_CARD)
        self._pipeline_table.pack(fill="both", expand=True, pady=(10, 10))

        row = tk.Frame(card.inner, bg=C.BG_CARD)
        row.pack(fill="x")
        ModernButton(row, text="Supprimer", style="danger", command=self._remove_selected_step, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(row, text="Monter", style="secondary", command=lambda: self._move_selected_step("up"), bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(row, text="Descendre", style="secondary", command=lambda: self._move_selected_step("down"), bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(row, text="Sauvegarder", style="ghost", command=self._save_pipeline, bg=C.BG_CARD).pack(side="right", padx=(8, 0))
        ModernButton(row, text="Charger", style="ghost", command=self._load_pipeline, bg=C.BG_CARD).pack(side="right", padx=(8, 0))
        ModernButton(row, text="Exporter code", style="ghost", command=self._export_code, bg=C.BG_CARD).pack(side="right")

        self._recommend_log = LogPanel(card.inner, height=5, label="Suggestions intelligentes", bg_outer=C.BG_CARD)
        self._recommend_log.pack(fill="both", expand=True, pady=(12, 0))

    def _build_preview_panel(self, parent):
        card = Card(parent, accent_color=C.ACCENT, pad=16)
        card.pack(fill="both", expand=True)

        tk.Label(card.inner, text="Prévisualisation temps réel", font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        self._preview_table = StyledTreeview(card.inner, columns=["A"], col_widths={"A": 120}, height=8, bg=C.BG_CARD)
        self._preview_table.pack(fill="both", expand=True, pady=(10, 10))

        self._viz_label = tk.Label(card.inner, text="Visualisation: aucune", bg=C.BG_CARD, fg=C.TEXT_SEC, anchor="w")
        self._viz_label.pack(fill="x")
        self._viz_image_holder = tk.Label(card.inner, bg=C.BG_CARD)
        self._viz_image_holder.pack(fill="x", pady=(8, 0))

        self._log_panel = LogPanel(card.inner, height=7, label="Logs de transformations", bg_outer=C.BG_CARD)
        self._log_panel.pack(fill="both", expand=True, pady=(12, 0))

    def on_enter(self):
        columns = self._service.get_columns()
        self._target_combo["values"] = columns
        if self._service.target_column and self._service.target_column in columns:
            self._target_var.set(self._service.target_column)
        elif columns:
            self._target_var.set(columns[-1])

        self._refresh_columns_list()
        self._refresh_pipeline()
        self._refresh_preview()
        self._refresh_logs()
        self._recommend_log.set_content(self._service.get_preprocessing_help())

    def _on_category_change(self):
        methods = self._catalog.get(self._category_var.get(), [])
        self._method_combo["values"] = methods
        if methods:
            self._method_var.set(methods[0])

    def _selected_columns(self) -> list[str]:
        indexes = self._columns_list.curselection()
        return [self._columns_list.get(i) for i in indexes]

    def _validate_column_selection(self) -> None:
        if self._category_var.get() != "data_visualization":
            return

        method = self._method_var.get()
        cols = self._selected_columns()
        multi_required = {"scatter", "pairplot", "correlation_heatmap", "boxplot"}
        if method in multi_required and len(cols) < 2:
            raise ValueError("Cette visualisation nécessite au moins deux colonnes sélectionnées.")
        if not cols:
            raise ValueError("Sélectionnez au moins une colonne pour la visualisation.")

    def _parse_options(self) -> dict:
        raw = self._options_var.get().strip() or "{}"
        try:
            options = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Options JSON invalides: {exc}") from exc

        target = self._target_var.get().strip()
        if target:
            options.setdefault("target_column", target)

        try:
            options.setdefault("test_size", float(self._test_size_var.get()))
        except ValueError:
            pass
        try:
            options.setdefault("random_state", int(self._random_state_var.get()))
        except ValueError:
            pass

        return options

    def _add_step(self):
        try:
            self._validate_column_selection()
            self._service.add_preprocessing_step(
                category=self._category_var.get(),
                method=self._method_var.get(),
                columns=self._selected_columns(),
                options=self._parse_options(),
            )
            self._refresh_pipeline()
            self._set_status("Étape ajoutée au pipeline")
        except Exception as exc:
            show_error("Erreur", str(exc))

    def _run_single_step(self):
        try:
            self._validate_column_selection()
            out = self._service.run_preprocessing_step(
                category=self._category_var.get(),
                method=self._method_var.get(),
                columns=self._selected_columns(),
                options=self._parse_options(),
            )
        except Exception as exc:
            show_error("Erreur", str(exc))
            return

        if out["ok"]:
            self._set_status(out["message"])
            self._refresh_preview()
            self._refresh_logs()
            self._display_visual_if_any(out.get("details", {}))
            show_info("Succès", out["message"])
        else:
            show_error("Erreur", out["message"])

    def _run_pipeline(self):
        try:
            outcomes = self._service.run_preprocessing_pipeline_advanced()
        except Exception as exc:
            show_error("Erreur", str(exc))
            return

        ok = sum(1 for o in outcomes if o["ok"])
        total = len(outcomes)
        self._set_status(f"Pipeline exécuté: {ok}/{total} étapes réussies")
        self._refresh_preview()
        self._refresh_logs()
        self._refresh_pipeline()
        show_info("Pipeline", f"Exécution terminée: {ok}/{total} étapes réussies")

    def _undo(self):
        if self._service.preprocessing_undo():
            self._set_status("Undo effectué")
            self._refresh_preview()
            self._refresh_logs()
        else:
            show_info("Undo", "Aucune opération à annuler.")

    def _redo(self):
        if self._service.preprocessing_redo():
            self._set_status("Redo effectué")
            self._refresh_preview()
            self._refresh_logs()
        else:
            show_info("Redo", "Aucune opération à rétablir.")

    def _recommend(self):
        recs = self._service.get_preprocessing_recommendations()
        issues = self._service.get_preprocessing_issues()
        text = "Issues détectés:\n" + json.dumps(issues, ensure_ascii=False, indent=2) + "\n\n"
        text += "Recommandations:\n- " + "\n- ".join(recs) if recs else "Aucune recommandation."
        self._recommend_log.set_content(text)
        self._set_status("Recommandations générées")

    def _remove_selected_step(self):
        idx = self._selected_pipeline_index()
        if idx is None:
            show_error("Erreur", "Sélectionnez une étape du pipeline.")
            return
        self._service.remove_preprocessing_step(idx)
        self._refresh_pipeline()

    def _move_selected_step(self, direction: str):
        idx = self._selected_pipeline_index()
        if idx is None:
            show_error("Erreur", "Sélectionnez une étape du pipeline.")
            return
        self._service.move_preprocessing_step(idx, direction)
        self._refresh_pipeline()

    def _save_pipeline(self):
        path = ask_save_pipeline_file()
        if not path:
            return
        self._service.save_preprocessing_pipeline(path)
        self._set_status("Pipeline sauvegardé")

    def _load_pipeline(self):
        path = ask_open_pipeline_file()
        if not path:
            return
        self._service.load_preprocessing_pipeline(path)
        self._refresh_pipeline()
        self._set_status("Pipeline chargé")

    def _export_code(self):
        path = ask_export_python_file()
        if not path:
            return
        self._service.export_preprocessing_pipeline_code(path)
        self._set_status("Code preprocessing exporté")

    def _prepare_training(self):
        target = self._target_var.get().strip()
        if not target:
            show_error("Erreur", "Sélectionnez une colonne cible.")
            return
        self._service.set_target_column(target)
        try:
            test_size = float(self._test_size_var.get())
            random_state = int(self._random_state_var.get())
        except ValueError:
            show_error("Erreur", "Test size / random state invalides.")
            return

        try:
            self._service.run_preprocessing(test_size=test_size, random_state=random_state, options={
                "apply_imputation": True,
                "apply_encoding": True,
                "apply_scaling": True,
                "numeric_impute_strategy": "median",
                "categorical_impute_strategy": "most_frequent",
            })
            show_info("Succès", "Préprocessing final prêt pour entraînement.")
            self._set_status("Données prêtes pour entraînement")
        except Exception as exc:
            show_error("Erreur", str(exc))

    def _selected_pipeline_index(self) -> int | None:
        sel = self._pipeline_table.tree.selection()
        if not sel:
            return None
        item = self._pipeline_table.tree.item(sel[0])
        return int(item["values"][0]) - 1

    def _refresh_columns_list(self):
        self._columns_list.delete(0, "end")
        for col in self._service.get_columns():
            self._columns_list.insert("end", col)

    def _refresh_pipeline(self):
        self._pipeline_table.tree.delete(*self._pipeline_table.tree.get_children())
        steps = self._service.get_preprocessing_pipeline()
        for i, s in enumerate(steps, start=1):
            cols = ", ".join(s.get("columns", []))[:80]
            self._pipeline_table.tree.insert("", "end", values=(i, s["category"], s["method"], cols))

    def _refresh_preview(self):
        df = self._service.get_current_dataframe_preview(80)
        cols = list(df.columns)
        if not cols:
            return

        self._preview_table.destroy()
        widths = {c: max(120, min(220, len(str(c)) * 11)) for c in cols}
        self._preview_table = StyledTreeview(self._preview_table.master, columns=cols, col_widths=widths, height=8, bg=C.BG_CARD)
        self._preview_table.pack(fill="both", expand=True, pady=(10, 10), before=self._viz_label)

        for _, row in df.iterrows():
            vals = [str(v) for v in row.values]
            self._preview_table.tree.insert("", "end", values=vals)

    def _refresh_logs(self):
        logs = self._service.get_preprocessing_logs()
        self._log_panel.set_content("\n".join(logs[-200:]) if logs else "Aucun log.")

    def _display_visual_if_any(self, details: dict):
        image_path = details.get("image_path")
        if not image_path:
            self._viz_label.configure(text="Visualisation: aucune")
            self._viz_image_holder.configure(image="", text="")
            self._viz_img = None
            return
        try:
            self._viz_img = tk.PhotoImage(file=image_path)
            self._viz_image_holder.configure(image=self._viz_img)
            self._viz_label.configure(text=f"Visualisation: {image_path}")
        except Exception:
            self._viz_label.configure(text=f"Visualisation générée: {image_path}")

    def _set_status(self, text: str):
        self._status_var.set(text)

    def _sync_bottom_cards_width(self):
        self.after_idle(self._apply_bottom_cards_width)

    def _apply_bottom_cards_width(self):
        try:
            left_width = self._pipeline_table.winfo_toplevel().winfo_width()
            _ = left_width
        except Exception:
            pass
