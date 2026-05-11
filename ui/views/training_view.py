"""
XAI Studio — Training & Deep Learning Studio
============================================
Unified dashboard for classical ML training, AutoML, and neural networks.
"""

import json
import threading
import tkinter as tk
from tkinter import ttk

from ui.widgets import (
    C,
    F,
    Card,
    ModernButton,
    SectionHeader,
    LogPanel,
    Tooltip,
    bind_mousewheel_to,
)
from ui.components.dialogs import show_error, show_info
from services.pipeline_service import PipelineService
from ui.views.nn_builder_view import NNBuilderView
from services.i18n import _


class TrainingView(ttk.Frame):
    """Training & Deep Learning Studio view."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._model_checks: dict[str, tk.BooleanVar] = {}
        self._model_use_defaults: dict[str, tk.BooleanVar] = {}
        self._param_vars: dict[str, dict[str, tk.Variable]] = {}
        self._param_meta: dict[str, dict[str, dict]] = {}
        self._selected_model: str | None = None

        self._task_var = tk.StringVar(value="-")
        self._target_var = tk.StringVar(value="-")
        self._shape_var = tk.StringVar(value="-")
        self._train_status_var = tk.StringVar(value=_("train_status_ready"))
        self._train_eta_var = tk.StringVar(value="-")
        self._train_resource_var = tk.StringVar(value="-")
        self._train_progress_var = tk.DoubleVar(value=0.0)
        self._show_all_params_var = tk.BooleanVar(value=False)

        self._automl_search_var = tk.StringVar(value="grid")
        self._automl_cv_var = tk.StringVar(value="3")
        self._automl_iter_var = tk.StringVar(value="20")
        self._automl_status_var = tk.StringVar(value=_("train_status_ready"))
        self._automl_best_var = tk.StringVar(value="-")
        self._automl_score_var = tk.StringVar(value="-")

        self._build()

    def _build(self):
        ct = tk.Frame(self, bg=C.BG_MAIN)
        ct.pack(fill="both", expand=True)
        px = 24

        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(
            header,
            icon="",
            title=_("train_main_title"),
            subtitle=_("train_main_subtitle"),
        ).pack(side="left")

        self._notebook = ttk.Notebook(ct)
        self._notebook.pack(fill="both", expand=True, padx=px, pady=(16, 24))

        self._train_tab = self._create_scroll_tab(self._notebook, _("train_tab_dash"))
        self._automl_tab = self._create_scroll_tab(self._notebook, _("train_tab_automl"))
        self._dl_tab = self._create_scroll_tab(self._notebook, _("train_tab_dl"))

        self._build_training_tab(self._train_tab)
        self._build_automl_tab(self._automl_tab)
        self._build_dl_tab(self._dl_tab)

        nav_row = tk.Frame(ct, bg=C.BG_MAIN)
        nav_row.pack(anchor="e", padx=px, pady=(0, 16))
        ModernButton(
            nav_row,
            text=_("train_btn_pred"),
            style="secondary",
            command=lambda: self._navigate_to("prediction"),
            bg=C.BG_MAIN,
        ).pack(side="right", padx=(8, 0))
        ModernButton(
            nav_row,
            text=_("train_btn_eval"),
            style="primary",
            command=lambda: self._navigate_to("evaluation"),
            bg=C.BG_MAIN,
        ).pack(side="right")

    def _create_scroll_tab(self, notebook: ttk.Notebook, title: str) -> tk.Frame:
        tab = tk.Frame(notebook, bg=C.BG_MAIN)
        notebook.add(tab, text=title)

        canvas = tk.Canvas(tab, bg=C.BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=C.BG_MAIN)
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        bind_mousewheel_to(canvas, content)
        return content

    def _build_training_tab(self, parent: tk.Frame):
        px = 16

        overview = Card(parent, accent_color=C.ACCENT, pad=16)
        overview.pack(fill="x", padx=px, pady=(0, 16))

        info = tk.Frame(overview.inner, bg=C.BG_CARD)
        info.pack(fill="x")
        tk.Label(info, text=_("train_task"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=0, sticky="w")
        tk.Label(info, textvariable=self._task_var, font=F.H3, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=0, sticky="w")

        tk.Label(info, text=_("train_target"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=1, sticky="w", padx=(24, 0))
        tk.Label(info, textvariable=self._target_var, font=F.H3, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=1, sticky="w", padx=(24, 0))

        tk.Label(info, text=_("train_dims"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=2, sticky="w", padx=(24, 0))
        tk.Label(info, textvariable=self._shape_var, font=F.H3, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=2, sticky="w", padx=(24, 0))

        layout = tk.Frame(parent, bg=C.BG_MAIN)
        layout.pack(fill="both", expand=True, padx=px, pady=(0, 16))
        layout.columnconfigure(0, weight=1, uniform="train_cols")
        layout.columnconfigure(1, weight=1, uniform="train_cols")

        left = tk.Frame(layout, bg=C.BG_MAIN)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = tk.Frame(layout, bg=C.BG_MAIN)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._models_card = Card(left, accent_color=C.INFO, pad=16)
        self._models_card.pack(fill="both", expand=True)
        tk.Label(self._models_card.inner, text=_("train_avail_models"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        actions = tk.Frame(self._models_card.inner, bg=C.BG_CARD)
        actions.pack(fill="x", pady=(8, 8))
        ModernButton(actions, text=_("train_btn_sel_all"), style="ghost", command=self._select_all, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(actions, text=_("train_btn_desel_all"), style="ghost", command=self._deselect_all, bg=C.BG_CARD).pack(side="left")

        self._models_list = tk.Frame(self._models_card.inner, bg=C.BG_CARD)
        self._models_list.pack(fill="both", expand=True)

        self._param_card = Card(right, accent_color=C.SUCCESS, pad=16)
        self._param_card.pack(fill="both", expand=True)
        self._param_header = tk.Label(self._param_card.inner, text=_("train_hyper_title"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT)
        self._param_header.pack(anchor="w")

        self._param_toolbar = tk.Frame(self._param_card.inner, bg=C.BG_CARD)
        self._param_toolbar.pack(fill="x", pady=(6, 8))
        self._param_mode_label = tk.Label(self._param_toolbar, text=_("train_auto_mode"), font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_SEC)
        self._param_mode_label.pack(side="left")
        self._param_mode_switch = ttk.Checkbutton(self._param_toolbar, style="Card.TCheckbutton")
        self._param_mode_switch.pack(side="left", padx=(8, 16))

        self._param_toggle_btn = ModernButton(
            self._param_toolbar,
            text=_("train_btn_show_more"),
            style="ghost",
            command=self._toggle_more_params,
            bg=C.BG_CARD,
        )
        self._param_toggle_btn.pack(side="left")

        self._param_body = tk.Frame(self._param_card.inner, bg=C.BG_CARD)
        self._param_body.pack(fill="both", expand=True)

        self._train_card = Card(parent, accent_color=C.ACCENT, pad=16)
        self._train_card.pack(fill="x", padx=px, pady=(0, 16))
        tk.Label(self._train_card.inner, text=_("train_track_title"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        train_top = tk.Frame(self._train_card.inner, bg=C.BG_CARD)
        train_top.pack(fill="x", pady=(8, 8))

        self._progress = ttk.Progressbar(train_top, variable=self._train_progress_var, maximum=1.0)
        self._progress.pack(side="left", fill="x", expand=True, padx=(0, 12))

        ModernButton(train_top, text=_("train_btn_start"), style="primary", command=self._on_train, bg=C.BG_CARD).pack(side="left")

        train_meta = tk.Frame(self._train_card.inner, bg=C.BG_CARD)
        train_meta.pack(fill="x")
        tk.Label(train_meta, textvariable=self._train_status_var, font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_SEC).pack(side="left")
        tk.Label(train_meta, textvariable=self._train_eta_var, font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_DIM).pack(side="left", padx=(16, 0))
        tk.Label(train_meta, textvariable=self._train_resource_var, font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_DIM).pack(side="left", padx=(16, 0))

        self._train_log = LogPanel(self._train_card.inner, height=7, label=_("train_log_title"), bg_outer=C.BG_CARD, scrollbar=True)
        self._train_log.pack(fill="both", expand=True, pady=(10, 0))

    def _build_automl_tab(self, parent: tk.Frame):
        px = 16
        config = Card(parent, accent_color=C.ACCENT, pad=16)
        config.pack(fill="x", padx=px, pady=(0, 16))
        tk.Label(config.inner, text=_("train_automl_title"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        row = tk.Frame(config.inner, bg=C.BG_CARD)
        row.pack(fill="x", pady=(8, 8))
        tk.Label(row, text=_("train_automl_search"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).pack(side="left")
        ttk.Combobox(row, textvariable=self._automl_search_var, values=["grid", "random", "optuna"], width=16, state="readonly").pack(side="left", padx=(8, 16))

        tk.Label(row, text=_("train_automl_cv"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).pack(side="left")
        ttk.Entry(row, textvariable=self._automl_cv_var, width=6).pack(side="left", padx=(8, 16))

        tk.Label(row, text=_("train_automl_iter"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).pack(side="left")
        ttk.Entry(row, textvariable=self._automl_iter_var, width=8).pack(side="left", padx=(8, 16))

        ModernButton(row, text=_("train_btn_automl"), style="primary", command=self._run_automl, bg=C.BG_CARD).pack(side="right")

        models_row = tk.Frame(config.inner, bg=C.BG_CARD)
        models_row.pack(fill="x")
        tk.Label(models_row, text=_("train_automl_cands"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        list_frame = tk.Frame(config.inner, bg=C.BG_CARD)
        list_frame.pack(fill="x", pady=(6, 0))
        self._automl_list = tk.Listbox(list_frame, selectmode="extended", height=6)
        self._automl_list.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self._automl_list.yview)
        sb.pack(side="right", fill="y")
        self._automl_list.configure(yscrollcommand=sb.set)

        results = Card(parent, accent_color=C.INFO, pad=16)
        results.pack(fill="both", expand=True, padx=px)
        tk.Label(results.inner, text=_("train_automl_res"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        summary = tk.Frame(results.inner, bg=C.BG_CARD)
        summary.pack(fill="x", pady=(8, 8))
        tk.Label(summary, text=_("train_automl_best"), font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=0, sticky="w")
        tk.Label(summary, textvariable=self._automl_best_var, font=F.H3, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=0, sticky="w")

        tk.Label(summary, text=_("train_automl_score"), font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=1, sticky="w", padx=(24, 0))
        tk.Label(summary, textvariable=self._automl_score_var, font=F.H3, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=1, sticky="w", padx=(24, 0))

        tk.Label(summary, text=_("train_automl_status"), font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=2, sticky="w", padx=(24, 0))
        tk.Label(summary, textvariable=self._automl_status_var, font=F.H3, bg=C.BG_CARD, fg=C.TEXT).grid(row=1, column=2, sticky="w", padx=(24, 0))

        self._automl_log = LogPanel(results.inner, height=8, label=_("train_automl_details"), bg_outer=C.BG_CARD, scrollbar=True)
        self._automl_log.pack(fill="both", expand=True)

    def _build_dl_tab(self, parent: tk.Frame):
        px = 16
        builder = NNBuilderView(parent, service=self._service, show_header=False)
        builder.pack(fill="both", expand=True, padx=px, pady=(0, 16))

    def on_enter(self):
        self._refresh_overview()
        self._refresh_model_list()
        self._refresh_automl_models()

    def _refresh_overview(self):
        pr = self._service.preprocessing_result
        if pr is None:
            self._task_var.set("-")
            self._target_var.set(", ".join(self._service.target_columns) if self._service.target_columns else "-")
            self._shape_var.set("-")
            return
        self._task_var.set(pr.task_type)
        self._target_var.set(", ".join(self._service.target_columns) if self._service.target_columns else "-")
        self._shape_var.set(f"{pr.X_train.shape[0]} x {pr.X_train.shape[1]}")

    def _refresh_model_list(self):
        for w in self._models_list.winfo_children():
            w.destroy()
        self._model_checks.clear()
        self._model_use_defaults.clear()

        model_names = self._service.get_model_names()
        if not model_names:
            tk.Label(self._models_list, text=_("train_msg_load_data"), bg=C.BG_CARD, fg=C.TEXT_DIM, font=F.BODY).pack(anchor="w")
            return

        registry = self._service.get_model_registry()
        n_cols = 2
        grid = tk.Frame(self._models_list, bg=C.BG_CARD)
        grid.pack(fill="x")

        for idx, name in enumerate(model_names):
            var = tk.BooleanVar(value=True)
            self._model_checks[name] = var
            self._model_use_defaults[name] = tk.BooleanVar(value=True)

            row = tk.Frame(grid, bg=C.BG_CARD)
            row.grid(row=idx // n_cols, column=idx % n_cols, sticky="ew", padx=(0, 20), pady=6)
            row.columnconfigure(1, weight=1)

            ttk.Checkbutton(row, text=name, variable=var, style="Card.TCheckbutton").grid(row=0, column=0, sticky="w")
            ModernButton(row, text=_("train_btn_config"), style="ghost", command=lambda n=name: self._select_model(n), bg=C.BG_CARD, width=120).grid(row=0, column=1, sticky="e")

        if model_names:
            self._select_model(model_names[0])

    def _select_model(self, name: str):
        self._selected_model = name
        self._param_header.configure(text=_("train_hyper_title_model").format(name))
        self._refresh_param_editor()

    def _refresh_param_editor(self):
        for w in self._param_body.winfo_children():
            w.destroy()

        name = self._selected_model
        if not name:
            tk.Label(self._param_body, text=_("train_msg_sel_model"), bg=C.BG_CARD, fg=C.TEXT_DIM, font=F.BODY).pack(anchor="w")
            return

        use_default = self._model_use_defaults.get(name)
        if use_default is None:
            use_default = tk.BooleanVar(value=True)
            self._model_use_defaults[name] = use_default

        self._param_mode_switch.configure(variable=use_default, command=self._refresh_param_editor)
        self._param_mode_label.configure(text=_("train_auto_mode"))

        if use_default.get():
            tk.Label(self._param_body, text=_("train_msg_use_def"), bg=C.BG_CARD, fg=C.TEXT_DIM, font=F.BODY).pack(anchor="w")
            return

        schema = self._service.get_model_param_schema(name, include_all=self._show_all_params_var.get())
        if not schema:
            tk.Label(self._param_body, text=_("train_msg_no_params"), bg=C.BG_CARD, fg=C.TEXT_DIM, font=F.BODY).pack(anchor="w")
            return

        body = tk.Frame(self._param_body, bg=C.BG_CARD)
        body.pack(fill="both", expand=True)
        self._param_vars[name] = {}
        self._param_meta[name] = {}

        for r, meta in enumerate(schema):
            pname = meta["name"]
            self._param_meta[name][pname] = meta
            label = tk.Label(body, text=pname, bg=C.BG_CARD, fg=C.TEXT, font=F.SMALL)
            label.grid(row=r, column=0, sticky="w", pady=6, padx=(0, 12))
            if meta.get("desc"):
                Tooltip(label, meta.get("desc", ""))

            control = self._build_param_control(body, name, meta)
            control.grid(row=r, column=1, sticky="w", pady=2)

        body.columnconfigure(1, weight=1)

    def _build_param_control(self, parent: tk.Frame, model_name: str, meta: dict):
        ptype = meta.get("type")
        default = meta.get("default")

        if ptype == "bool":
            var = tk.StringVar(value="True" if bool(default) else "False")
            self._param_vars[model_name][meta["name"]] = var
            return ttk.Combobox(parent, textvariable=var, values=["True", "False"], width=24, state="readonly")

        if ptype == "choice" and meta.get("choices"):
            var = tk.StringVar(value=str(default) if default is not None else str(meta["choices"][0]))
            self._param_vars[model_name][meta["name"]] = var
            cb = ttk.Combobox(parent, textvariable=var, values=[str(c) for c in meta["choices"]], width=24, state="readonly")
            return cb

        if ptype in ("int", "float"):
            var = tk.StringVar(value=str(default) if default is not None else "")
            self._param_vars[model_name][meta["name"]] = var
            return ttk.Entry(parent, textvariable=var, width=24)

        var = tk.StringVar(value="" if default is None else str(default))
        self._param_vars[model_name][meta["name"]] = var
        return ttk.Entry(parent, textvariable=var, width=24)

    def _toggle_more_params(self):
        self._show_all_params_var.set(not self._show_all_params_var.get())
        if self._show_all_params_var.get():
            self._param_toggle_btn.itemconfig(self._param_toggle_btn._text, text=_("train_btn_show_less"))
        else:
            self._param_toggle_btn.itemconfig(self._param_toggle_btn._text, text=_("train_btn_show_more"))
        self._refresh_param_editor()

    def _select_all(self):
        for v in self._model_checks.values():
            v.set(True)

    def _deselect_all(self):
        for v in self._model_checks.values():
            v.set(False)

    def _collect_model_params(self, selected_models: list[str]) -> dict[str, dict]:
        params_map: dict[str, dict] = {}
        for name in selected_models:
            use_default = self._model_use_defaults.get(name)
            if use_default is None or use_default.get():
                params_map[name] = {}
                continue

            model_params = {}
            vars_map = self._param_vars.get(name, {})
            meta_map = self._param_meta.get(name, {})
            for pname, var in vars_map.items():
                meta = meta_map.get(pname, {})
                val = var.get()
                if isinstance(val, str):
                    if val.strip() == "":
                        continue
                    if val.strip().lower() == "none":
                        model_params[pname] = None
                        continue
                ptype = meta.get("type")
                try:
                    if ptype == "int":
                        model_params[pname] = int(float(val))
                    elif ptype == "float":
                        model_params[pname] = float(val)
                    elif ptype == "bool":
                        model_params[pname] = str(val).strip().lower() in ("true", "1", "yes", "y")
                    else:
                        model_params[pname] = val
                except Exception:
                    model_params[pname] = val
            params_map[name] = model_params

        return params_map

    def _on_train(self):
        selected = [n for n, v in self._model_checks.items() if v.get()]
        if not selected:
            show_error(_("train_err_title"), _("train_err_sel_model"))
            return
        if self._service.preprocessing_result is None:
            show_error(_("train_err_title"), _("train_err_prep_first"))
            return

        params_map = self._collect_model_params(selected)
        total = len(selected)
        self._progress.configure(maximum=total)
        self._train_progress_var.set(0)
        self._train_status_var.set(_("train_status_running"))
        self._train_eta_var.set("-")
        self._train_log.set_content(_("train_log_start"))

        durations: list[float] = []

        def progress_cb(cur, tot, name, duration: float | None = None):
            if duration is not None:
                durations.append(duration)

            remaining = "-"
            if durations and cur <= tot:
                avg = sum(durations) / len(durations)
                rem_count = max(0, tot - cur)
                rem_secs = int(rem_count * avg)
                m, s = divmod(rem_secs, 60)
                remaining = f"{m}m{s}s" if m else f"{s}s"

            resources = self._snapshot_resources()
            self.after(0, lambda: self._update_training_progress(cur, tot, name, remaining, resources))

        def thread():
            try:
                self._service.run_training(
                    selected_models=selected,
                    model_params_map=params_map,
                    progress_callback=progress_cb,
                )
                self.after(0, self._on_train_done)
            except Exception as exc:
                self.after(0, lambda exc=exc: show_error(_("train_err_title"), str(exc)))

        threading.Thread(target=thread, daemon=True).start()

    def _update_training_progress(self, cur: int, total: int, model_name: str, remaining: str, resources: str | None):
        self._train_progress_var.set(cur)
        self._train_status_var.set(_("train_status_done_model").format(model_name, cur, total))
        self._train_eta_var.set(_("train_eta").format(remaining))
        if resources:
            self._train_resource_var.set(resources)
        self._train_log.append(f"{cur}/{total} - {model_name}")

    def _on_train_done(self):
        self._train_status_var.set(_("train_status_done"))
        logs = "\n".join(self._service.get_training_logs())
        if logs:
            self._train_log.set_content(logs)
        show_info(_("train_success_title"), _("train_status_done"))

    def _snapshot_resources(self) -> str | None:
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            text = f"CPU {cpu:.0f}% | MEM {mem.percent:.0f}%"
        except Exception:
            return None

        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                text += f" | GPU {gpus[0].load * 100:.0f}%"
        except Exception:
            pass
        return text

    def _refresh_automl_models(self):
        self._automl_list.delete(0, "end")
        for name in self._service.get_model_names():
            self._automl_list.insert("end", name)

    def _run_automl(self):
        if self._service.preprocessing_result is None:
            show_error(_("train_err_title"), _("train_err_prep_first"))
            return

        selected = [self._automl_list.get(i) for i in self._automl_list.curselection()]
        cv = int(self._automl_cv_var.get() or 3)
        n_iter = int(self._automl_iter_var.get() or 20)
        search = self._automl_search_var.get().strip().lower()

        self._automl_status_var.set(_("train_automl_running"))
        self._automl_log.set_content(_("train_automl_start"))

        def thread():
            try:
                res = self._service.run_automl(candidate_models=selected or None, search=search, n_iter=n_iter, cv=cv)
                self.after(0, lambda: self._render_automl_results(res))
            except Exception as exc:
                self.after(0, lambda: show_error(_("train_err_title"), str(exc)))

        threading.Thread(target=thread, daemon=True).start()

    def _render_automl_results(self, res: dict):
        name = res.get("name") or "-"
        score = res.get("score")
        params = res.get("params")
        self._automl_best_var.set(name)
        self._automl_score_var.set(f"{score:.4f}" if isinstance(score, float) else str(score))
        text = {
            "best_model": name,
            "best_score": score,
            "best_params": params,
        }
        self._automl_log.set_content(json.dumps(text, indent=2, ensure_ascii=False))
        self._automl_status_var.set(_("train_automl_done"))

    def _navigate_to(self, view_name: str) -> None:
        root = self.winfo_toplevel()
        navigate = getattr(root, "navigate_to", None)
        if callable(navigate):
            navigate(view_name)

    @staticmethod
    def _format_params(params: dict) -> str:
        if not params:
            return ""
        parts = [f"{k}={repr(v)}" for k, v in params.items()]
        return ", ".join(parts)