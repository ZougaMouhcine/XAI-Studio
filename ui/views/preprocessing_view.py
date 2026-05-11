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
from services.i18n import _


class PreprocessingView(ttk.Frame):
    """Professional preprocessing module for interactive ML workflows."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._catalog = self._service.get_preprocessing_catalog()
        self._viz_img = None
        self._cluster_token = "__clustering__"

        self._target_var = tk.StringVar()
        self._test_size_var = tk.StringVar(value="0.2")
        self._random_state_var = tk.StringVar(value="42")

        self._category_var = tk.StringVar(value="data_cleaning")
        self._method_var = tk.StringVar()
        self._status_var = tk.StringVar(value=_("prep_status_ready"))

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
            title=_("prep_title"),
            subtitle=_("prep_subtitle"),
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

        nav_row = tk.Frame(ct, bg=C.BG_MAIN)
        nav_row.pack(anchor="e", padx=px, pady=(0, 16))
        ModernButton(
            nav_row,
            text=_("prep_btn_validate"),
            style="primary",
            command=self._prepare_training,
            bg=C.BG_MAIN,
        ).pack(side="right")
        self._on_category_change()

    def _build_controls(self, parent):
        # Layout: label / input pairs across columns 0..7. Reserve column 7 for right-aligned actions.
        for i in range(8):
            parent.columnconfigure(i, weight=0)
        parent.columnconfigure(7, weight=1)

        target_label_frame = tk.Frame(parent, bg=C.BG_CARD)
        target_label_frame.grid(row=0, column=0, sticky="nw")
        tk.Label(target_label_frame, text=_("prep_target"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")
        self._target_mode_btn = tk.Button(
            target_label_frame, text="Mode: Multi", font=F.TINY, bg=C.BG_CARD, fg=C.ACCENT,
            bd=0, cursor="hand2", command=self._toggle_target_mode
        )
        self._target_mode_btn.pack(anchor="w", pady=(2, 0))

        self._target_list = tk.Listbox(parent, selectmode="extended", height=3, exportselection=False, width=self._field_width)
        self._target_list.grid(row=0, column=1, padx=(8, 16), sticky="w")

        tk.Label(parent, text="Test size", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=0, column=2, sticky="w")
        ttk.Entry(parent, textvariable=self._test_size_var, width=self._field_width).grid(row=0, column=3, padx=(8, 16), sticky="w")

        random_state_row = tk.Frame(parent, bg=C.BG_CARD)
        random_state_row.grid(row=0, column=4, columnspan=2, sticky="w")
        tk.Label(random_state_row, text="Random state", font=F.H4, bg=C.BG_CARD, fg=C.TEXT).pack(side="left")
        ttk.Entry(random_state_row, textvariable=self._random_state_var, width=self._field_width).pack(side="left", padx=(8, 0))


        tk.Label(parent, text=_("prep_cat"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=0, sticky="w", pady=(14, 0))
        cat_combo = ttk.Combobox(parent, textvariable=self._category_var, width=self._field_width, state="readonly", values=list(self._catalog.keys()))
        cat_combo.grid(row=1, column=1, padx=(8, 16), pady=(14, 0), sticky="w")
        cat_combo.bind("<<ComboboxSelected>>", lambda _: self._on_category_change())

        tk.Label(parent, text=_("prep_method"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=2, sticky="w", pady=(14, 0))
        self._method_combo = ttk.Combobox(parent, textvariable=self._method_var, width=self._field_width, state="readonly")
        self._method_combo.grid(row=1, column=3, columnspan=2, padx=(8, 16), pady=(14, 0), sticky="w")

        tk.Label(parent, text=_("prep_cols"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).grid(row=2, column=0, sticky="nw", pady=(14, 0))
        self._columns_list = tk.Listbox(parent, selectmode="extended", height=5, exportselection=False, width=self._field_width)
        # Keep the columns listbox the same character width as other fields (no extra columnspan)
        self._columns_list.grid(row=2, column=1, columnspan=1, sticky="w", pady=(14, 0), padx=(8, 16))
        tk.Label(
            parent,
            text=_("prep_multi"),
            font=F.TINY,
            bg=C.BG_CARD,
            fg=C.TEXT_MUTED,
        ).grid(row=3, column=1, columnspan=2, sticky="w", padx=(8, 16), pady=(2, 0))

        actions = tk.Frame(parent, bg=C.BG_CARD)
        actions.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(14, 0))

        ModernButton(actions, text=_("prep_btn_exec_step"), style="primary", command=self._run_single_step, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text=_("prep_btn_add_step"), style="secondary", command=self._add_step, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text=_("prep_btn_exec_pipe"), style="secondary", command=self._run_pipeline, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text="Undo", style="ghost", command=self._undo, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text="Redo", style="ghost", command=self._redo, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text=_("prep_btn_recom"), style="ghost", command=self._recommend, bg=C.BG_CARD).pack(side="right")

    def _build_pipeline_panel(self, parent):
        card = Card(parent, accent_color=C.ACCENT, pad=16)
        card.pack(fill="both", expand=True)

        tk.Label(card.inner, text=_("prep_pipe_dyn"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        cols = ("#", _("prep_cat"), _("prep_method"), _("prep_cols"))
        widths = {"#": 40, _("prep_cat"): 160, _("prep_method"): 180, _("prep_cols"): 260}
        self._pipeline_table = StyledTreeview(card.inner, columns=cols, col_widths=widths, height=8, bg=C.BG_CARD)
        self._pipeline_table.pack(fill="both", expand=True, pady=(10, 10))

        row = tk.Frame(card.inner, bg=C.BG_CARD)
        row.pack(fill="x")
        ModernButton(row, text=_("prep_btn_del"), style="danger", command=self._remove_selected_step, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(row, text=_("prep_btn_up"), style="secondary", command=lambda: self._move_selected_step("up"), bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(row, text=_("prep_btn_down"), style="secondary", command=lambda: self._move_selected_step("down"), bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(row, text=_("prep_btn_save"), style="ghost", command=self._save_pipeline, bg=C.BG_CARD).pack(side="right", padx=(8, 0))
        ModernButton(row, text=_("prep_btn_load"), style="ghost", command=self._load_pipeline, bg=C.BG_CARD).pack(side="right", padx=(8, 0))
        ModernButton(row, text=_("prep_btn_export"), style="ghost", command=self._export_code, bg=C.BG_CARD).pack(side="right")

        self._recommend_log = LogPanel(card.inner, height=5, label=_("prep_recom_log"), bg_outer=C.BG_CARD, scrollbar=True)
        self._recommend_log.pack(fill="both", expand=True, pady=(12, 0))

    def _build_preview_panel(self, parent):
        card = Card(parent, accent_color=C.ACCENT, pad=16)
        card.pack(fill="both", expand=True)

        tk.Label(card.inner, text=_("prep_preview"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        self._preview_table = StyledTreeview(card.inner, columns=["A"], col_widths={"A": 120}, height=8, bg=C.BG_CARD)
        self._preview_table.pack(fill="both", expand=True, pady=(10, 10))

        self._viz_label = tk.Label(card.inner, text=_("prep_viz_none"), bg=C.BG_CARD, fg=C.TEXT_SEC, anchor="w")
        self._viz_label.pack(fill="x")
        self._viz_image_holder = tk.Label(card.inner, bg=C.BG_CARD)
        self._viz_image_holder.pack(fill="x", pady=(8, 0))

        self._log_panel = LogPanel(card.inner, height=7, label=_("prep_log_title"), bg_outer=C.BG_CARD)
        self._log_panel.pack(fill="both", expand=True, pady=(12, 0))

    def _toggle_target_mode(self):
        current = self._target_list.cget("selectmode")
        if current == "extended":
            self._target_list.configure(selectmode="browse")
            self._target_mode_btn.configure(text="Mode: Single")
            sel = self._target_list.curselection()
            if len(sel) > 1:
                self._target_list.selection_clear(0, "end")
                self._target_list.selection_set(sel[0])
        else:
            self._target_list.configure(selectmode="extended")
            self._target_mode_btn.configure(text="Mode: Multi")

    def on_enter(self):
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
            raise ValueError(_("prep_err_multi_viz"))
        if not cols:
            raise ValueError(_("prep_err_no_col_viz"))

    def _parse_options(self) -> dict:
        options = {}

        indexes = self._target_list.curselection()
        targets = [self._target_list.get(i) for i in indexes if self._target_list.get(i) != _("prep_clustering")]
        if targets:
            options.setdefault("target_columns", targets)

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
            self._set_status(_("prep_added_step"))
        except Exception as exc:
            show_error(_("prep_err_title"), str(exc))

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
            show_error(_("prep_err_title"), str(exc))
            return

        if out["ok"]:
            self._set_status(out["message"])
            self._refresh_columns_list()
            self._refresh_preview()
            self._refresh_logs()
            self._display_visual_if_any(out.get("details", {}))
            show_info(_("prep_success_title"), out["message"])
        else:
            show_error(_("prep_err_title"), out["message"])

    def _run_pipeline(self):
        try:
            outcomes = self._service.run_preprocessing_pipeline_advanced()
        except Exception as exc:
            show_error(_("prep_err_title"), str(exc))
            return

        ok = sum(1 for o in outcomes if o["ok"])
        total = len(outcomes)
        self._set_status(_("prep_pipe_exec").format(ok, total))
        self._refresh_columns_list()
        self._refresh_preview()
        self._refresh_logs()
        self._refresh_pipeline()
        show_info("Pipeline", _("prep_pipe_exec_done").format(ok, total))

    def _undo(self):
        if self._service.preprocessing_undo():
            self._set_status(_("prep_undo_done"))
            self._refresh_columns_list()
            self._refresh_preview()
            self._refresh_logs()
        else:
            show_info("Undo", _("prep_undo_empty"))

    def _redo(self):
        if self._service.preprocessing_redo():
            self._set_status(_("prep_redo_done"))
            self._refresh_columns_list()
            self._refresh_preview()
            self._refresh_logs()
        else:
            show_info("Redo", _("prep_redo_empty"))

    def _recommend(self):
        recs = self._service.get_preprocessing_recommendations()
        issues = self._service.get_preprocessing_issues()
        text = _("prep_recom_issues") + json.dumps(issues, ensure_ascii=False, indent=2) + "\n\n"
        text += _("prep_recom_list") + "\n- ".join(recs) if recs else _("prep_recom_none")
        self._recommend_log.set_content(text)
        self._set_status(_("prep_recom_done"))

    def _remove_selected_step(self):
        idx = self._selected_pipeline_index()
        if idx is None:
            show_error(_("prep_err_title"), _("prep_err_sel_step"))
            return
        self._service.remove_preprocessing_step(idx)
        self._refresh_pipeline()

    def _move_selected_step(self, direction: str):
        idx = self._selected_pipeline_index()
        if idx is None:
            show_error(_("prep_err_title"), _("prep_err_sel_step"))
            return
        self._service.move_preprocessing_step(idx, direction)
        self._refresh_pipeline()

    def _save_pipeline(self):
        path = ask_save_pipeline_file()
        if not path:
            return
        self._service.save_preprocessing_pipeline(path)
        self._set_status(_("prep_pipe_saved"))

    def _load_pipeline(self):
        path = ask_open_pipeline_file()
        if not path:
            return
        self._service.load_preprocessing_pipeline(path)
        self._refresh_pipeline()
        self._set_status(_("prep_pipe_loaded"))

    def _export_code(self):
        path = ask_export_python_file()
        if not path:
            return
        self._service.export_preprocessing_pipeline_code(path)
        self._set_status(_("prep_code_exported"))

    def _prepare_training(self):
        indexes = self._target_list.curselection()
        targets = [self._target_list.get(i) for i in indexes]

        if not targets:
            show_error(_("prep_err_title"), _("prep_err_target"))
            return

        if _("prep_clustering") in targets:
            self._service.set_target_columns([])
        else:
            self._service.set_target_columns(targets)
        try:
            test_size = float(self._test_size_var.get())
            random_state = int(self._random_state_var.get())
        except ValueError:
            show_error(_("prep_err_title"), _("prep_err_params"))
            return

        try:
            self._service.run_preprocessing(test_size=test_size, random_state=random_state, options={
                "apply_imputation": True,
                "apply_encoding": True,
                "apply_scaling": True,
                "numeric_impute_strategy": "median",
                "categorical_impute_strategy": "most_frequent",
            })
            show_info(_("prep_success_title"), _("prep_success_final"))
            self._set_status(_("prep_status_ready_train"))
            self._navigate_to("training")
        except Exception as exc:
            show_error(_("prep_err_title"), str(exc))

    def _selected_pipeline_index(self) -> int | None:
        sel = self._pipeline_table.tree.selection()
        if not sel:
            return None
        item = self._pipeline_table.tree.item(sel[0])
        return int(item["values"][0]) - 1

    def _refresh_columns_list(self):
        columns = self._service.get_columns()

        # Update columns list
        self._columns_list.delete(0, "end")
        for col in columns:
            self._columns_list.insert("end", col)

        # Update target list
        self._target_list.delete(0, "end")
        for col in columns + [_("prep_clustering")]:
            self._target_list.insert("end", col)

        if self._service.target_columns:
            for i, col in enumerate(columns):
                if col in self._service.target_columns:
                    self._target_list.selection_set(i)
        elif self._service.target_columns is None or not self._service.target_columns:
            # Select clustering
            self._target_list.selection_set("end")

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
        self._log_panel.set_content("\n".join(logs[-200:]) if logs else _("prep_no_log"))

    def _display_visual_if_any(self, details: dict):
        image_path = details.get("image_path")
        if not image_path:
            self._viz_label.configure(text=_("prep_viz_none"))
            self._viz_image_holder.configure(image="", text="")
            self._viz_img = None
            return
        try:
            self._viz_img = tk.PhotoImage(file=image_path)
            self._viz_image_holder.configure(image=self._viz_img)
            self._viz_label.configure(text=_("prep_viz_path").format(image_path))
        except Exception:
            self._viz_label.configure(text=_("prep_viz_gen").format(image_path))

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

    # ── Agent UI Automation ─────────────────────────────────────────────

    def agent_add_step(
        self,
        category: str,
        method: str,
        columns: list[str],
        options: dict | None = None,
        delay_ms: int = 300,
        on_done=None,
    ):
        """Visually construct a single pipeline step, mimicking user interaction.

        Scheduled entirely on the Tk main thread via `self.after()` chaining.
        Each sub-action (set category, set method, select columns, flash + add)
        is separated by *delay_ms* milliseconds so the user can follow along.

        Parameters
        ----------
        category : str
            Preprocessing category key (e.g. ``"data_cleaning"``).
        method : str
            Method name within the category (e.g. ``"impute"``).
        columns : list[str]
            Column names to select in the listbox.
        options : dict, optional
            Extra options (target_column, test_size, random_state, …).
        delay_ms : int
            Delay between each visual sub-step.
        on_done : callable, optional
            Callback invoked (with no args) after the step has been added.
        """
        opts = options or {}

        # Step 1 — Set category dropdown & refresh method list
        def _step1_set_category():
            self._category_var.set(category)
            self._on_category_change()
            self.after(delay_ms, _step2_set_method)

        # Step 2 — Set method dropdown
        def _step2_set_method():
            self._method_var.set(method)
            self.after(delay_ms, _step3_select_columns)

        # Step 3 — Select columns in listbox
        def _step3_select_columns():
            self._columns_list.selection_clear(0, "end")
            all_items = self._columns_list.get(0, "end")
            for col in columns:
                for idx, item in enumerate(all_items):
                    if item == col:
                        self._columns_list.selection_set(idx)
                        self._columns_list.see(idx)
                        break
            self.after(delay_ms, _step4_set_options)

        # Step 4 — Apply options to UI fields
        def _step4_set_options():
            if "target_columns" in opts:
                targets = opts["target_columns"]
                self._target_list.selection_clear(0, "end")
                for target in targets:
                    for i in range(self._target_list.size()):
                        if self._target_list.get(i) == target:
                            self._target_list.selection_set(i)
            if "test_size" in opts:
                self._test_size_var.set(str(opts["test_size"]))
            if "random_state" in opts:
                self._random_state_var.set(str(opts["random_state"]))
            self.after(delay_ms, _step5_flash_and_add)

        # Step 5 — Flash the Add button, then add step
        def _step5_flash_and_add():
            self._add_step()
            if on_done:
                on_done()

        # Kick off the chain
        self.after(0, _step1_set_category)

    def agent_add_steps(
        self,
        steps: list[dict],
        delay_ms: int = 400,
        on_all_done=None,
    ):
        """Queue multiple steps for sequential visual construction.

        Parameters
        ----------
        steps : list[dict]
            Each dict must have ``category``, ``method``, ``columns``.
            Optional keys: ``options``.
        delay_ms : int
            Delay between each visual sub-step within a single step.
        on_all_done : callable, optional
            Called after all steps have been added.
        """
        if not steps:
            if on_all_done:
                on_all_done()
            return

        remaining = list(steps)

        def _add_next():
            if not remaining:
                if on_all_done:
                    on_all_done()
                return
            step = remaining.pop(0)
            self.agent_add_step(
                category=step["category"],
                method=step["method"],
                columns=step.get("columns", []),
                options=step.get("options"),
                delay_ms=delay_ms,
                on_done=lambda: self.after(delay_ms, _add_next),
            )

        _add_next()

    def _navigate_to(self, view_name: str) -> None:
        root = self.winfo_toplevel()
        navigate = getattr(root, "navigate_to", None)
        if callable(navigate):
            navigate(view_name)
