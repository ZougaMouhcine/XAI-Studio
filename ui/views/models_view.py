"""
XAI Studio — Models View (v2 — Dashboard Design)
===================================================
Manage saved models with a polished card-based layout.
"""

import tkinter as tk
from tkinter import ttk

from ui.widgets import C, F, Card, ModernButton, SectionHeader, StyledTreeview, Badge, bind_mousewheel_to
from ui.components.dialogs import show_error, show_info, ask_confirm
from services.pipeline_service import PipelineService
from services.i18n import _


class ModelsView(ttk.Frame):
    """Model management dashboard."""

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

        # ── Header ───────────────────────────────────────────────
        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(header, icon="", title=_("models_title"),
                      subtitle=_("models_subtitle")).pack(side="left")

        btn_row = tk.Frame(header, bg=C.BG_MAIN)
        btn_row.pack(side="right")
        ModernButton(btn_row, text=_("models_btn_pred"), icon="",
                 style="secondary", command=lambda: self._navigate_to("prediction"),
                 bg=C.BG_MAIN).pack(side="left", padx=(0, 8), pady=6)
        ModernButton(btn_row, text=_("models_btn_save_all"), icon="",
                     style="primary", command=self._on_save_all,
                     bg=C.BG_MAIN).pack(side="left", padx=(0, 8), pady=6)
        ModernButton(btn_row, text=_("models_btn_refresh"), icon="",
                     style="secondary", command=self._refresh,
                     bg=C.BG_MAIN).pack(side="left", pady=6)

        # ── In-memory models card ────────────────────────────────
        mem_card = Card(ct, accent_color=C.INFO, pad=16)
        mem_card.pack(fill="x", padx=px, pady=(16, 0))

        tk.Label(mem_card.inner, text=_("models_mem_title"), font=F.H3,
                 bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w", pady=(0, 8))

        self._memory_frame = tk.Frame(mem_card.inner, bg=C.BG_CARD)
        self._memory_frame.pack(fill="x")

        # ── Saved models table ───────────────────────────────────
        tk.Label(ct, text=_("models_disk_title"), font=F.H2,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="w", padx=px, pady=(16, 16))

        self._table_container = tk.Frame(ct, bg=C.BG_MAIN)
        self._table_container.pack(fill="both", expand=True, padx=px, pady=(0, 16))

        cols = (_("models_col_file"), _("models_col_class"), _("models_col_task"), _("models_col_target"), _("models_col_size"), _("models_col_date"))
        widths = {cols[0]: 220, cols[1]: 180, cols[2]: 100,
                  cols[3]: 120, cols[4]: 90, cols[5]: 160}
        self._stv = StyledTreeview(self._table_container, columns=cols,
                                    col_widths=widths, height=8)
        self._stv.pack(fill="both", expand=True)

        # Action bar
        actions = tk.Frame(ct, bg=C.BG_MAIN)
        actions.pack(fill="x", padx=px, pady=(16, 24))

        ModernButton(actions, text=_("models_btn_del"), icon="",
                     style="danger", command=self._on_delete,
                     bg=C.BG_MAIN).pack(side="right")

    # ──────────────────────────────────────────────────────────────
    def on_enter(self):
        self._refresh_memory()
        self._refresh()

    def _refresh_memory(self):
        for w in self._memory_frame.winfo_children():
            w.destroy()

        models = self._service.trained_models
        if not models:
            tk.Label(self._memory_frame,
                     text=_("models_msg_no_mem"),
                     font=F.BODY, bg=C.BG_CARD, fg=C.TEXT_MUTED).pack(anchor="w", pady=4)
            return

        for name, entry in models.items():
            if entry.get("model") is None:
                continue

            row = tk.Frame(self._memory_frame, bg=C.BG_CARD)
            row.pack(fill="x", pady=3)

            tk.Label(row, text="●", font=F.ICON_S, bg=C.BG_CARD,
                     fg=C.SUCCESS).pack(side="left", padx=(0, 8))
            tk.Label(row, text=name, font=F.H4, bg=C.BG_CARD,
                     fg=C.TEXT).pack(side="left")
            tk.Label(row, text=type(entry["model"]).__name__, font=F.SMALL,
                     bg=C.BG_CARD, fg=C.TEXT_MUTED).pack(side="left", padx=(12, 0))

            ModernButton(row, text=_("models_btn_save"), style="secondary",
                         command=lambda n=name: self._on_save_single(n),
                         bg=C.BG_CARD, width=120).pack(side="right")

    def _refresh(self):
        self._stv.tree.delete(*self._stv.tree.get_children())
        models = self._service.get_saved_models()

        for m in models:
            self._stv.tree.insert("", "end", values=(
                m.get("filename", "?"),
                m.get("model_class", "?"),
                m.get("task_type", "?"),
                m.get("target_column", "?"),
                m.get("size_kb", "?"),
                m.get("saved_at", "?")[:19] if m.get("saved_at") else "?",
            ), tags=(m.get("filepath", ""),))

    def _on_save_single(self, model_name):
        try:
            path = self._service.save_trained_model(model_name)
            show_info(_("models_success_title"), _("models_success_saved").format(model_name, path))
            self._refresh()
        except Exception as exc:
            show_error(_("models_err_title"), str(exc))

    def _on_save_all(self):
        if not self._service.trained_models:
            show_error(_("models_err_title"), _("models_err_no_save"))
            return
        try:
            paths = self._service.save_all_trained_models()
            show_info(_("models_success_title"), _("models_success_save_all").format(len(paths)))
            self._refresh()
        except Exception as exc:
            show_error(_("models_err_title"), str(exc))

    def _on_delete(self):
        sel = self._stv.tree.selection()
        if not sel:
            show_error(_("models_err_title"), _("models_err_sel_del"))
            return

        item = self._stv.tree.item(sel[0])
        filename = item["values"][0]
        filepath = item["tags"][0] if item["tags"] else None

        if not filepath:
            show_error(_("models_err_title"), _("models_err_path"))
            return
        if not ask_confirm(_("models_confirm_title"), _("models_confirm_del").format(filename)):
            return
        if self._service.delete_saved_model(filepath):
            show_info(_("models_success_title"), _("models_success_del").format(filename))
            self._refresh()
        else:
            show_error(_("models_err_title"), _("models_err_del_fail"))

    def _navigate_to(self, view_name: str) -> None:
        root = self.winfo_toplevel()
        navigate = getattr(root, "navigate_to", None)
        if callable(navigate):
            navigate(view_name)
