"""
XAI Studio — Data View (v2 — Dashboard Design)
=================================================
Load CSV, preview data, and display summary stats in dashboard cards.
"""

import os
import tkinter as tk
from tkinter import ttk

from ui.widgets import C, F, Card, MetricTile, ModernButton, SectionHeader, StyledTreeview, bind_mousewheel_to
from ui.components.dialogs import ask_open_csv, show_error
from services.pipeline_service import PipelineService
from services.i18n import _


class DataView(ttk.Frame):
    """Dashboard-style data exploration view."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._has_header_var = tk.BooleanVar(value=True)
        self._build()

    def _build(self):
        # Scrollable container
        canvas = tk.Canvas(self, bg=C.BG_MAIN, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll_frame = tk.Frame(canvas, bg=C.BG_MAIN)
        self._scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        bind_mousewheel_to(canvas, self._scroll_frame)

        content = self._scroll_frame
        pad_x = 24

        # ── Header ───────────────────────────────────────────────
        header_row = tk.Frame(content, bg=C.BG_MAIN)
        header_row.pack(fill="x", padx=pad_x, pady=(24, 0))

        SectionHeader(header_row, icon="", title=_("data_title"),
                      subtitle=_("data_subtitle")).pack(side="left")

        controls = tk.Frame(header_row, bg=C.BG_MAIN)
        controls.pack(side="right")

        ttk.Checkbutton(
            controls,
            text=_("data_header_checkbox"),
            variable=self._has_header_var,
        ).pack(side="left", padx=(0, 12), pady=6)

        ModernButton(
            controls,
            text=_("data_btn"),
            icon="",
            style="primary",
            command=self._on_load,
            bg=C.BG_MAIN,
        ).pack(side="left")

        # ── File info card ────────────────────────────────────────
        self._info_card = Card(content, accent_color=C.ACCENT, pad=16)
        self._info_card.pack(fill="x", padx=pad_x, pady=(16, 0))

        self._file_icon = tk.Label(self._info_card.inner, text="CSV", font=F.H4,
                        bg=C.BG_CARD, fg=C.ACCENT)
        self._file_icon.pack(anchor="center", pady=(0, 8))

        self._info_text = tk.Frame(self._info_card.inner, bg=C.BG_CARD)
        self._info_text.pack(fill="x", expand=True)

        self._info_name = tk.Label(self._info_text, text=_("data_no_file"),
                        font=F.H4, bg=C.BG_CARD, fg=C.TEXT_SEC)
        self._info_name.pack(anchor="center")
        self._info_detail = tk.Label(self._info_text, text=_("data_detail"),
                                      font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_MUTED)
        self._info_detail.pack(anchor="center")

        # ── Metrics row ──────────────────────────────────────────
        self._metrics_frame = tk.Frame(content, bg=C.BG_MAIN)
        self._metrics_frame.pack(fill="x", padx=pad_x, pady=(16, 0))

        # Placeholder tiles
        placeholders = [
            ("", "—", _("data_rows"), C.ACCENT),
            ("", "—", _("data_cols"), C.INFO),
            ("", "—", _("data_num"), C.SUCCESS),
            ("", "—", _("data_cat"), C.WARNING),
            ("", "—", _("data_miss"), C.DANGER),
        ]
        self._tiles = []
        for icon, val, lbl, color in placeholders:
            tile = MetricTile(self._metrics_frame, icon=icon, value=val, label=lbl, color=color)
            tile.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self._tiles.append(tile)

        # ── Table section ────────────────────────────────────────
        table_header = tk.Frame(content, bg=C.BG_MAIN)
        table_header.pack(fill="x", padx=pad_x, pady=(16, 0))

        tk.Label(table_header, text=_("data_preview"), font=F.H2,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(side="left")
        self._rows_label = tk.Label(table_header, text="",
                                     font=F.SMALL, bg=C.BG_MAIN, fg=C.TEXT_MUTED)
        self._rows_label.pack(side="right")

        self._table_container = tk.Frame(content, bg=C.BG_MAIN)
        self._table_container.pack(fill="both", expand=True, padx=pad_x, pady=(16, 24))

        # Empty-state message
        self._empty = tk.Label(self._table_container,
                                text=_("data_empty"),
                                font=F.BODY, bg=C.BG_MAIN, fg=C.TEXT_DIM)
        self._empty.pack(pady=24)

        nav_row = tk.Frame(content, bg=C.BG_MAIN)
        nav_row.pack(anchor="e", padx=pad_x, pady=(0, 16))
        ModernButton(
            nav_row,
            text=_("data_next"),
            style="primary",
            command=lambda: self._navigate_to("preprocessing"),
            bg=C.BG_MAIN,
        ).pack(side="right")

    # ──────────────────────────────────────────────────────────────
    def _on_load(self):
        filepath = ask_open_csv()
        if not filepath:
            return
        try:
            df = self._service.load_data(filepath, has_header=self._has_header_var.get())
        except Exception as exc:
            show_error(_("data_err_title"), str(exc))
            return

        self._update_info(filepath, df)
        self._update_metrics()
        self._update_table(df)

    def _update_info(self, filepath, df):
        name = os.path.basename(filepath)
        n, c = df.shape
        target = self._service.target_column or "—"
        self._info_name.configure(text=name, fg=C.TEXT)
        header_mode = _("data_with_headers") if self._has_header_var.get() else _("data_no_headers")
        self._info_detail.configure(
            text=f"{n:,} {_('data_rows').lower()}  ×  {c} {_('data_cols').lower()}   ·   {header_mode}   ·   {_('data_target')} : {target}")

    def _update_metrics(self):
        s = self._service.get_data_summary()
        if not s:
            return
        data = [
            str(s["shape"][0]),
            str(s["shape"][1]),
            str(len(s["numeric_columns"])),
            str(len(s["categorical_columns"])),
            str(sum(s["missing_values"].values())),
        ]
        for tile, val in zip(self._tiles, data):
            tile.set(val)

    def _update_table(self, df):
        for w in self._table_container.winfo_children():
            w.destroy()

        cols = list(df.columns)
        widths = {col: max(len(str(col)) * 10, 80) for col in cols}

        stv = StyledTreeview(self._table_container, columns=cols,
                              col_widths=widths, height=min(len(df), 18))
        stv.pack(fill="both", expand=True)

        for _, row in df.head(100).iterrows():
            stv.tree.insert("", "end", values=[str(v) for v in row])

        self._rows_label.configure(text=f"{_('data_displaying').format(min(len(df), 100), len(df))}")

    def _navigate_to(self, view_name: str) -> None:
        root = self.winfo_toplevel()
        navigate = getattr(root, "navigate_to", None)
        if callable(navigate):
            navigate(view_name)
