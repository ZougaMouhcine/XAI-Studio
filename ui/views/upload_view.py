"""
XAI Studio — Upload Model View
=================================
Upload a saved .pkl / .joblib model and display its metadata.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog

from ui.widgets import C, F, Card, MetricTile, ModernButton, SectionHeader, Badge
from ui.components.dialogs import show_error, show_info
from services.pipeline_service import PipelineService
from services.i18n import _

from utils.logger import get_logger

logger = get_logger(__name__)


class UploadView(ttk.Frame):
    """Model upload dashboard."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._build()

    def _build(self):
        # Scrollable container
        canvas = tk.Canvas(self, bg=C.BG_MAIN, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll_frame = tk.Frame(canvas, bg=C.BG_MAIN)
        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        content = self._scroll_frame
        px = 28

        # ── Header ──────────────────────────────────────────────
        header_row = tk.Frame(content, bg=C.BG_MAIN)
        header_row.pack(fill="x", padx=px, pady=(24, 0))

        SectionHeader(
            header_row,
            icon="📤",
            title=_("upload_title"),
            subtitle=_("upload_subtitle"),
        ).pack(side="left")

        ModernButton(
            header_row,
            text=_("upload_btn"),
            icon="📂",
            style="primary",
            command=self._on_upload,
            bg=C.BG_MAIN,
        ).pack(side="right", pady=6)

        # ── Status card ─────────────────────────────────────────
        self._status_card = Card(content, accent_color=C.TEXT_DIM, pad=14)
        self._status_card.pack(fill="x", padx=px, pady=(16, 0))

        self._status_icon = tk.Label(
            self._status_card.inner, text="📦", font=F.ICON_M,
            bg=C.BG_CARD, fg=C.TEXT_MUTED,
        )
        self._status_icon.pack(side="left", padx=(0, 12))

        status_text = tk.Frame(self._status_card.inner, bg=C.BG_CARD)
        status_text.pack(side="left", fill="x", expand=True)

        self._status_name = tk.Label(
            status_text, text=_("upload_no_model"),
            font=F.H4, bg=C.BG_CARD, fg=C.TEXT_SEC,
        )
        self._status_name.pack(anchor="w")
        self._status_detail = tk.Label(
            status_text,
            text=_("upload_detail"),
            font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_MUTED,
        )
        self._status_detail.pack(anchor="w")

        # ── Metrics row ─────────────────────────────────────────
        self._metrics_frame = tk.Frame(content, bg=C.BG_MAIN)
        self._metrics_frame.pack(fill="x", padx=px, pady=(16, 0))

        placeholders = [
            ("", "—", _("upload_alg"), C.ACCENT),
            ("", "—", _("upload_task"), C.INFO),
            ("", "—", _("upload_feat"), C.SUCCESS),
            ("", "—", _("upload_class"), C.WARNING),
        ]
        self._tiles = []
        for icon, val, lbl, color in placeholders:
            tile = MetricTile(
                self._metrics_frame, icon=icon, value=val, label=lbl, color=color,
            )
            tile.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self._tiles.append(tile)

        # ── Parameters card ─────────────────────────────────────
        tk.Label(
            content, text=_("upload_params"), font=F.H2,
            bg=C.BG_MAIN, fg=C.TEXT,
        ).pack(anchor="w", padx=px, pady=(20, 8))

        self._params_card = Card(content, pad=16)
        self._params_card.pack(fill="x", padx=px, pady=(0, 8))

        self._params_text = tk.Text(
            self._params_card.inner, height=10, bg=C.BG_CARD, fg=C.TEXT,
            font=F.MONO_S, relief="flat", padx=12, pady=10, wrap="word",
            insertbackground=C.TEXT, selectbackground=C.TREE_SELECT,
        )
        self._params_text.pack(fill="both", expand=True)
        self._params_text.insert("1.0", _("upload_params_empty"))
        self._params_text.configure(state="disabled")

        # ── Feature names card ──────────────────────────────────
        tk.Label(
            content, text=_("upload_feat_exp"), font=F.H2,
            bg=C.BG_MAIN, fg=C.TEXT,
        ).pack(anchor="w", padx=px, pady=(12, 8))

        self._features_card = Card(content, pad=16)
        self._features_card.pack(fill="both", expand=True, padx=px, pady=(0, 24))

        self._features_text = tk.Text(
            self._features_card.inner, height=8, bg=C.BG_CARD, fg=C.TEXT,
            font=F.MONO_S, relief="flat", padx=12, pady=10, wrap="word",
            insertbackground=C.TEXT, selectbackground=C.TREE_SELECT,
        )
        self._features_text.pack(fill="both", expand=True)
        self._features_text.insert("1.0", _("upload_feat_empty"))
        self._features_text.configure(state="disabled")

    # ──────────────────────────────────────────────────────────────
    def _on_upload(self):
        filepath = filedialog.askopenfilename(
            title=_("upload_btn"),
            filetypes=[
                ("Model Files", "*.pkl *.joblib"),
                ("Pickle", "*.pkl"),
                ("Joblib", "*.joblib"),
                ("All Files", "*.*"),
            ],
        )
        if not filepath:
            return

        try:
            model, metadata = self._service.load_external_model(filepath)
            
            # Check compatibility with current preprocessing
            pr = self._service.preprocessing_result
            if pr:
                is_compat, msg = self._service.check_model_compatibility(metadata)
                if not is_compat:
                    show_error(_("upload_err_title"), msg)
                    return
        except Exception as exc:
            show_error(_("upload_err_title"), str(exc))
            logger.error("Failed to load model: %s", exc)
            return

        self._display_model_info(filepath, metadata)
        show_info(_("upload_success_title"), f"{_('upload_success_msg')} :\n{os.path.basename(filepath)}")

    def _display_model_info(self, filepath: str, meta: dict):
        name = os.path.basename(filepath)
        algo = meta.get("algorithm", _("upload_unknown"))
        task = meta.get("task_type", _("upload_unknown"))
        n_feat = meta.get("n_features", "—")
        model_class = meta.get("model_class", "—")

        # Status card
        self._status_name.configure(text=name, fg=C.SUCCESS)
        size_kb = os.path.getsize(filepath) / 1024
        self._status_detail.configure(
            text=f"{_('upload_loaded')} · {size_kb:.1f} KB · {algo}"
        )

        # Metric tiles
        tile_values = [str(algo), str(task), str(n_feat), str(model_class)]
        for tile, val in zip(self._tiles, tile_values):
            tile.set(val)

        # Parameters
        params = meta.get("params", {})
        params_str = "\n".join(f"{k} = {v}" for k, v in params.items()) if params else _("upload_no_params")
        self._params_text.configure(state="normal")
        self._params_text.delete("1.0", "end")
        self._params_text.insert("1.0", params_str)
        self._params_text.configure(state="disabled")

        # Feature names
        features = meta.get("feature_names", [])
        if features:
            feat_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(features))
        else:
            feat_str = _("upload_no_feat_names")
        self._features_text.configure(state="normal")
        self._features_text.delete("1.0", "end")
        self._features_text.insert("1.0", feat_str)
        self._features_text.configure(state="disabled")
