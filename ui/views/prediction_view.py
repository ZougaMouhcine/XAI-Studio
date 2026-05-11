"""
XAI Studio — Prediction (Inference) View
========================================
Run predictions on new data and explain results.
"""

import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import pandas as pd

from ui.widgets import (
    C,
    F,
    Card,
    MetricTile,
    ModernButton,
    SectionHeader,
    StyledTreeview,
    LogPanel,
    bind_mousewheel_to,
)
from ui.components.dialogs import show_error
from ui.components.xai_panel import XAIPanel
from services.prediction_controller import PredictionController
from services.i18n import _


class PredictionView(ttk.Frame):
    """Prediction & local explanation dashboard."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._controller = PredictionController()
        self._registry_map: dict[str, str] = {}
        self._active_entry_id: str | None = None
        self._input_feature_names: list[str] = []
        self._feature_names: list[str] = []
        self._feature_schema: dict[str, dict] = {}
        self._manual_fields: dict[str, dict] = {}
        self._csv_df: pd.DataFrame | None = None
        self._what_if_values: np.ndarray | None = None
        self._what_if_after: str | None = None
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

        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(
            header,
            icon="",
            title=_("pred_title"),
            subtitle=_("pred_subtitle"),
        ).pack(side="left")

        action_row = tk.Frame(header, bg=C.BG_MAIN)
        action_row.pack(side="right")
        ModernButton(action_row, text=_("pred_btn_refresh"), style="secondary", command=self._refresh_registry, bg=C.BG_MAIN).pack(
            side="left", padx=(0, 8), pady=6
        )
        ModernButton(action_row, text=_("pred_btn_load"), style="secondary", command=self._load_external_model, bg=C.BG_MAIN).pack(
            side="left", pady=6
        )

        # Model selection
        model_card = Card(ct, accent_color=C.INFO, pad=16)
        model_card.pack(fill="x", padx=px, pady=(16, 0))

        top = tk.Frame(model_card.inner, bg=C.BG_CARD)
        top.pack(fill="x")
        tk.Label(top, text=_("eval_reg_title"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(side="left")

        self._model_select = ttk.Combobox(top, state="readonly", width=40)
        self._model_select.pack(side="right")
        self._model_select.bind("<<ComboboxSelected>>", self._on_model_selected)

        meta_row = tk.Frame(model_card.inner, bg=C.BG_CARD)
        meta_row.pack(fill="x", pady=(10, 0))

        self._meta_tiles = [
            MetricTile(meta_row, value="-", label=_("pred_meta_algo"), color=C.ACCENT),
            MetricTile(meta_row, value="-", label=_("pred_meta_task"), color=C.INFO),
            MetricTile(meta_row, value="-", label=_("pred_meta_feat"), color=C.SUCCESS),
            MetricTile(meta_row, value="-", label=_("pred_meta_class"), color=C.WARNING),
        ]
        for tile in self._meta_tiles:
            tile.pack(side="left", fill="x", expand=True, padx=(0, 8))

        meta_detail = tk.Frame(model_card.inner, bg=C.BG_CARD)
        meta_detail.pack(fill="x", pady=(12, 0))
        self._features_panel = LogPanel(meta_detail, height=5, label=_("pred_meta_feat_title"), bg_outer=C.BG_CARD)
        self._features_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._params_panel = LogPanel(meta_detail, height=5, label=_("pred_meta_hyper"), bg_outer=C.BG_CARD)
        self._params_panel.pack(side="left", fill="both", expand=True)

        # Main content
        content = tk.Frame(ct, bg=C.BG_MAIN)
        content.pack(fill="both", expand=True, padx=px, pady=(16, 24))
        content.columnconfigure(0, weight=3, uniform="pred_col")
        content.columnconfigure(1, weight=2, uniform="pred_col")

        left = tk.Frame(content, bg=C.BG_MAIN)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = tk.Frame(content, bg=C.BG_MAIN)
        right.grid(row=0, column=1, sticky="nsew")

        # Input card
        input_card = Card(left, accent_color=C.ACCENT, pad=16)
        input_card.pack(fill="both", expand=True)
        tk.Label(input_card.inner, text=_("pred_input_title"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        self._input_tabs = ttk.Notebook(input_card.inner)
        self._input_tabs.pack(fill="both", expand=True, pady=(8, 0))

        self._manual_tab = tk.Frame(self._input_tabs, bg=C.BG_CARD)
        self._csv_tab = tk.Frame(self._input_tabs, bg=C.BG_CARD)
        self._input_tabs.add(self._manual_tab, text=_("pred_tab_manual"))
        self._input_tabs.add(self._csv_tab, text=_("pred_tab_csv"))

        self._build_manual_tab(self._manual_tab)
        self._build_csv_tab(self._csv_tab)

        # Results card
        results_card = Card(left, accent_color=C.SUCCESS, pad=16)
        results_card.pack(fill="x", pady=(16, 0))
        tk.Label(results_card.inner, text=_("pred_res_title"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        res_row = tk.Frame(results_card.inner, bg=C.BG_CARD)
        res_row.pack(fill="x", pady=(8, 0))
        self._pred_tile = MetricTile(res_row, value="-", label=_("pred_res_pred"), color=C.ACCENT)
        self._pred_tile.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._conf_tile = MetricTile(res_row, value="-", label=_("pred_res_conf"), color=C.INFO)
        self._conf_tile.pack(side="left", fill="x", expand=True)

        self._proba_table = StyledTreeview(results_card.inner, columns=(_("pred_res_class"), _("pred_res_proba")), height=5, bg=C.BG_CARD)
        self._proba_table.pack(fill="x", pady=(10, 0))

        self._batch_table = StyledTreeview(results_card.inner, columns=(_("pred_res_row"), _("pred_res_pred"), _("pred_res_conf")), height=6, bg=C.BG_CARD)
        self._batch_table.pack(fill="x", pady=(10, 0))

        # What-if card
        what_if_card = Card(left, accent_color=C.WARNING, pad=16)
        what_if_card.pack(fill="both", expand=True, pady=(16, 0))
        tk.Label(what_if_card.inner, text=_("pred_wi_title"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")
        self._what_if_frame = tk.Frame(what_if_card.inner, bg=C.BG_CARD)
        self._what_if_frame.pack(fill="x", pady=(8, 0))

        # XAI panel
        self._xai_panel = XAIPanel(right, xai_service=self._controller.xai_service, title=_("pred_xai_title"))
        self._xai_panel.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    def _build_manual_tab(self, parent: tk.Frame):
        top = tk.Frame(parent, bg=C.BG_CARD)
        top.pack(fill="x", padx=12, pady=(12, 0))
        ModernButton(top, text=_("pred_btn_predict"), style="primary", command=self._run_predict_manual, bg=C.BG_CARD).pack(side="left")

        form_wrap = tk.Frame(parent, bg=C.BG_CARD)
        form_wrap.pack(fill="both", expand=True, padx=12, pady=(12, 12))

        canvas = tk.Canvas(form_wrap, bg=C.BG_CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_wrap, orient="vertical", command=canvas.yview)
        self._manual_form = tk.Frame(canvas, bg=C.BG_CARD)
        self._manual_form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._manual_form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_csv_tab(self, parent: tk.Frame):
        top = tk.Frame(parent, bg=C.BG_CARD)
        top.pack(fill="x", padx=12, pady=(12, 0))
        ModernButton(top, text=_("pred_btn_upload"), style="secondary", command=self._load_csv, bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(top, text=_("pred_btn_predict_csv"), style="primary", command=self._run_predict_csv, bg=C.BG_CARD).pack(side="left")

        self._csv_preview = tk.Frame(parent, bg=C.BG_CARD)
        self._csv_preview.pack(fill="both", expand=True, padx=12, pady=(12, 12))
        self._csv_empty = tk.Label(self._csv_preview, text=_("pred_csv_empty"), font=F.BODY, bg=C.BG_CARD, fg=C.TEXT_MUTED)
        self._csv_empty.pack(pady=24)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def on_enter(self):
        self._refresh_registry()

    def _refresh_registry(self):
        entries = self._controller.refresh_registry()
        self._registry_map.clear()
        items = []
        for entry in entries:
            label = f"{entry.name} ({entry.source})"
            items.append(label)
            self._registry_map[label] = entry.entry_id

        self._model_select["values"] = items
        if items:
            self._model_select.set(items[0])
            self._set_active_entry(self._registry_map[items[0]])

    def _on_model_selected(self, _=None):
        label = self._model_select.get()
        entry_id = self._registry_map.get(label)
        if entry_id:
            self._set_active_entry(entry_id)

    def _set_active_entry(self, entry_id: str):
        entry = self._controller.get_entry(entry_id)
        if not entry:
            return
        self._active_entry_id = entry_id
        meta = entry.metadata or {}
        algo = meta.get("algorithm", "-")
        task = entry.task_type or "-"
        n_feat = meta.get("n_features", "-")
        model_class = meta.get("model_class", "-")
        values = [algo, task, n_feat, model_class]
        for tile, val in zip(self._meta_tiles, values):
            tile.set(str(val))

        features = meta.get("feature_names", [])
        self._feature_names = list(features)
        self._input_feature_names = list(meta.get("input_feature_names", features))
        self._feature_schema = self._schema_by_name(meta)
        self._features_panel.set_content("\n".join(features) if features else "-")

        params = meta.get("params", {})
        if params:
            params_text = "\n".join(f"{k} = {v}" for k, v in params.items())
        else:
            params_text = "-"
        self._params_panel.set_content(params_text)

        self._build_manual_form_fields()

    def _schema_by_name(self, metadata: dict) -> dict[str, dict]:
        schema = metadata.get("feature_schema", []) or []
        if isinstance(schema, dict):
            items = []
            for name, spec in schema.items():
                if isinstance(spec, dict):
                    items.append({"name": name, **spec})
                else:
                    items.append({"name": name, "type": "choice", "choices": list(spec) if isinstance(spec, (list, tuple, set)) else [], "default": None})
            schema = items

        mapped: dict[str, dict] = {}
        for spec in schema:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if name:
                mapped[name] = spec
        return mapped

    def _load_external_model(self):
        filepath = filedialog.askopenfilename(
            title=_("eval_file_model"),
            filetypes=[("Model Files", "*.pkl *.joblib"), ("Pickle", "*.pkl"), ("Joblib", "*.joblib")],
        )
        if not filepath:
            return
        try:
            self._controller.register_external_model(filepath)
            self._refresh_registry()
        except Exception as exc:
            show_error(_("pred_err_title"), str(exc))

    # ------------------------------------------------------------------
    # Manual input
    # ------------------------------------------------------------------
    def _build_manual_form_fields(self):
        for w in self._manual_form.winfo_children():
            w.destroy()
        self._manual_fields.clear()

        if not self._input_feature_names:
            tk.Label(self._manual_form, text=_("pred_msg_sel_feat"), bg=C.BG_CARD, fg=C.TEXT_MUTED).pack(anchor="w")
            return

        for idx, name in enumerate(self._input_feature_names):
            row = tk.Frame(self._manual_form, bg=C.BG_CARD)
            row.pack(fill="x", pady=6)
            tk.Label(row, text=name, font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT).pack(side="left")
            spec = self._feature_schema.get(name, {})
            ftype = str(spec.get("type", "numeric")).lower()
            default = spec.get("default")
            choices = [str(v) for v in spec.get("choices", []) or []]
            if ftype in {"choice", "categorical", "string", "bool"} and choices:
                var = tk.StringVar(value=str(default) if default is not None else choices[0])
                widget = ttk.Combobox(row, textvariable=var, values=choices, width=20, state="readonly")
            else:
                var = tk.StringVar(value="" if default is None else str(default))
                widget = ttk.Entry(row, textvariable=var, width=18)
            widget.pack(side="right", padx=(12, 0))
            self._manual_fields[name] = {"var": var, "widget": widget, "spec": spec}

    def _collect_manual_values(self) -> list[object]:
        values: list[object] = []
        for name in self._input_feature_names:
            field = self._manual_fields.get(name)
            if field is None:
                raise RuntimeError("Missing manual input")
            var = field["var"]
            spec = field["spec"]
            val = var.get().strip()
            if val == "":
                raise RuntimeError(f"Missing value for {name}")
            ftype = str(spec.get("type", "numeric")).lower()
            if ftype in {"choice", "categorical", "string", "bool"}:
                values.append(val)
                continue
            try:
                values.append(float(val))
            except ValueError as exc:
                raise RuntimeError(f"Invalid value for {name}") from exc
        return values

    def _run_predict_manual(self):
        if not self._active_entry_id:
            show_error(_("pred_err_title"), _("pred_err_sel_model"))
            return
        try:
            values = self._collect_manual_values()
        except Exception as exc:
            show_error(_("pred_err_title"), str(exc))
            return

        self._predict_values(values)
        self._build_what_if(values)

    # ------------------------------------------------------------------
    # CSV input
    # ------------------------------------------------------------------
    def _load_csv(self):
        filepath = filedialog.askopenfilename(
            title=_("pred_btn_upload"),
            filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            show_error(_("pred_err_title"), str(exc))
            return

        self._csv_df = df
        self._render_csv_preview(df)

    def _render_csv_preview(self, df: pd.DataFrame):
        for w in self._csv_preview.winfo_children():
            w.destroy()

        cols = list(df.columns)
        widths = {c: max(120, min(220, len(str(c)) * 10)) for c in cols}
        table = StyledTreeview(self._csv_preview, columns=cols, col_widths=widths, height=min(len(df), 8), bg=C.BG_CARD)
        table.pack(fill="both", expand=True)

        for _, row in df.head(50).iterrows():
            table.tree.insert("", "end", values=[str(v) for v in row.values])

    def _run_predict_csv(self):
        if not self._active_entry_id:
            show_error(_("pred_err_title"), _("pred_err_sel_model"))
            return
        if self._csv_df is None:
            show_error(_("pred_err_title"), _("pred_err_csv_up"))
            return

        def _compute():
            return self._controller.predict_frame(self._active_entry_id, self._csv_df)

        def _done(result):
            self._render_batch_predictions(result)
            if len(result.predictions) > 0:
                source_cols = self._input_feature_names or self._feature_names
                first_row = self._csv_df[source_cols].iloc[0].values
                self._predict_values(first_row.tolist())
                self._build_what_if(first_row.tolist(), data_source=self._csv_df)

        self._run_async(_compute, _done)

    # ------------------------------------------------------------------
    # Prediction rendering
    # ------------------------------------------------------------------
    def _predict_values(self, values: list[float]):
        if not self._active_entry_id:
            return

        def _compute():
            return self._controller.predict_row(self._active_entry_id, values)

        def _done(result):
            self._render_single_prediction(result)
            entry = self._controller.ensure_loaded(self._active_entry_id)
            feature_names = entry.metadata.get("feature_names", [])
            self._xai_panel.update(entry.model, np.array(values, dtype=object), feature_names, entry.task_type)

        self._run_async(_compute, _done)

    def _render_single_prediction(self, result):
        pred = result.predictions[0] if len(result.predictions) else "-"
        self._pred_tile.set(str(pred))

        if result.confidence is not None and len(result.confidence):
            self._conf_tile.set(f"{result.confidence[0]:.3f}")
        else:
            self._conf_tile.set("-")

        self._proba_table.tree.delete(*self._proba_table.tree.get_children())
        if result.probabilities is not None and len(result.probabilities):
            probs = result.probabilities[0]
            for idx, val in enumerate(probs):
                self._proba_table.tree.insert("", "end", values=(idx, f"{val:.4f}"))

    def _render_batch_predictions(self, result):
        self._batch_table.tree.delete(*self._batch_table.tree.get_children())
        for i, pred in enumerate(result.predictions[:50]):
            conf = "-"
            if result.confidence is not None:
                conf = f"{result.confidence[i]:.3f}"
            self._batch_table.tree.insert("", "end", values=(i, pred, conf))

    # ------------------------------------------------------------------
    # What-if
    # ------------------------------------------------------------------
    def _build_what_if(self, values: list[float], data_source: pd.DataFrame | None = None):
        for w in self._what_if_frame.winfo_children():
            w.destroy()

        if not self._feature_names:
            return

        numeric_items = []
        for idx, name in enumerate(self._input_feature_names or self._feature_names):
            spec = self._feature_schema.get(name, {})
            ftype = str(spec.get("type", "numeric")).lower()
            if ftype in {"choice", "categorical", "string", "bool"}:
                continue
            try:
                numeric_items.append((idx, name, float(values[idx])))
            except Exception:
                continue

        if not numeric_items:
            self._what_if_values = None
            tk.Label(self._what_if_frame, text=_("pred_msg_wi_empty"), bg=C.BG_CARD, fg=C.TEXT_MUTED).pack(anchor="w")
            return

        self._what_if_values = np.array(values, dtype=object)
        for source_idx, name, val in numeric_items[:8]:
            if data_source is not None and name in data_source.columns:
                min_val = float(np.nanmin(data_source[name]))
                max_val = float(np.nanmax(data_source[name]))
            else:
                span = max(1.0, abs(val) * 0.5)
                min_val = val - span
                max_val = val + span

            row = tk.Frame(self._what_if_frame, bg=C.BG_CARD)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=name, font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT).pack(side="left", padx=(0, 8))
            slider = tk.Scale(
                row,
                from_=min_val,
                to=max_val,
                orient="horizontal",
                resolution=(max_val - min_val) / 100 if max_val != min_val else 1,
                length=220,
                bg=C.BG_CARD,
                fg=C.TEXT,
                highlightthickness=0,
                troughcolor=C.BG_INPUT,
                command=lambda v, i=source_idx: self._on_what_if_change(i, v),
            )
            slider.set(val)
            slider.pack(side="right", fill="x", expand=True)

    def _on_what_if_change(self, idx: int, value: str):
        if self._what_if_values is None:
            return
        self._what_if_values[idx] = float(value)
        self._queue_what_if_update()

    def _queue_what_if_update(self):
        if self._what_if_after is not None:
            self.after_cancel(self._what_if_after)
        self._what_if_after = self.after(200, self._run_what_if_update)

    def _run_what_if_update(self):
        if self._what_if_values is None:
            return
        self._predict_values(self._what_if_values.tolist())

    # ------------------------------------------------------------------
    # Async helper
    # ------------------------------------------------------------------
    def _run_async(self, fn, on_done):
        def _done(result):
            self.after(0, lambda: on_done(result))

        def _err(exc):
            self.after(0, lambda: show_error(_("pred_err_title"), str(exc)))

        self._controller.run_async(fn, _done, _err)
