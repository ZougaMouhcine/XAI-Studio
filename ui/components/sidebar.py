"""
XAI Studio — Sidebar Navigation (v2 — Premium Design)
=======================================================
Sleek vertical nav with glow-style active indicator, subtle hover,
and a polished brand header.
"""

import tkinter as tk
from ui.widgets import C, F


class Sidebar(tk.Frame):
    """
    Dark sidebar with icon + label navigation buttons.

    Parameters
    ----------
    parent : tk widget
    on_navigate : callable(view_name: str)
    """

    NAV_ITEMS = [
        ("📂", "Données",        "data"),
        ("⚙",  "Préprocessing",  "preprocessing"),
        ("🚀", "Entraînement",   "training"),
        ("📊", "Évaluation",     "evaluation"),
        ("💾", "Modèles",        "models"),
    ]

    def __init__(self, parent, on_navigate=None):
        super().__init__(parent, bg=C.BG_SIDEBAR, width=230)
        self.pack_propagate(False)
        self._on_navigate = on_navigate
        self._buttons: dict[str, dict] = {}
        self._active_view = None
        self._build()

    # ─────────────────────────────────────────────────────────────
    def _build(self):
        # ── Brand header ─────────────────────────────────────────
        header = tk.Frame(self, bg=C.BG_SIDEBAR)
        header.pack(fill="x", pady=(28, 0), padx=20)

        # Logo row
        logo_row = tk.Frame(header, bg=C.BG_SIDEBAR)
        logo_row.pack(anchor="w")

        # Glow dot
        tk.Label(logo_row, text="🧠", font=(F.FAM, 26),
                 bg=C.BG_SIDEBAR, fg=C.ACCENT).pack(side="left", padx=(0, 10))

        title_col = tk.Frame(logo_row, bg=C.BG_SIDEBAR)
        title_col.pack(side="left")

        tk.Label(title_col, text="XAI Studio", font=(F.FAM, 15, "bold"),
                 bg=C.BG_SIDEBAR, fg=C.TEXT).pack(anchor="w")
        tk.Label(title_col, text="ML Core · Phase 1", font=F.TINY,
                 bg=C.BG_SIDEBAR, fg=C.TEXT_DIM).pack(anchor="w")

        # Accent line beneath header
        accent_bar = tk.Frame(self, bg=C.ACCENT, height=2)
        accent_bar.pack(fill="x", padx=20, pady=(16, 0))

        # Subtle fade
        fade = tk.Frame(self, bg=C.BG_SIDEBAR, height=16)
        fade.pack(fill="x")

        # ── Section label ────────────────────────────────────────
        tk.Label(self, text="NAVIGATION", font=(F.FAM, 9, "bold"),
                 bg=C.BG_SIDEBAR, fg=C.TEXT_DIM).pack(anchor="w", padx=24, pady=(4, 8))

        # ── Nav buttons ──────────────────────────────────────────
        nav_frame = tk.Frame(self, bg=C.BG_SIDEBAR)
        nav_frame.pack(fill="x", padx=12)

        for icon, label, view_name in self.NAV_ITEMS:
            self._create_nav_button(nav_frame, icon, label, view_name)

        # ── Spacer ───────────────────────────────────────────────
        tk.Frame(self, bg=C.BG_SIDEBAR).pack(fill="both", expand=True)

        # ── Footer ───────────────────────────────────────────────
        footer = tk.Frame(self, bg=C.BG_SIDEBAR)
        footer.pack(fill="x", padx=20, pady=(0, 20))

        sep = tk.Frame(footer, bg=C.BORDER, height=1)
        sep.pack(fill="x", pady=(0, 10))

        tk.Label(footer, text="v1.0.0  ·  Python + scikit-learn",
                 font=(F.FAM, 8), bg=C.BG_SIDEBAR, fg=C.TEXT_DIM).pack(anchor="w")

    # ─────────────────────────────────────────────────────────────
    def _create_nav_button(self, parent, icon, label, view_name):
        """Create a single nav row: indicator | icon | label."""
        # Outer container
        row = tk.Frame(parent, bg=C.BG_SIDEBAR, pady=1)
        row.pack(fill="x")

        # Active indicator (thin left bar, hidden by default)
        indicator = tk.Frame(row, bg=C.BG_SIDEBAR, width=3)
        indicator.pack(side="left", fill="y")

        # Clickable area
        btn = tk.Frame(row, bg=C.BG_SIDEBAR, padx=12, pady=10, cursor="hand2")
        btn.pack(side="left", fill="x", expand=True)

        icon_lbl = tk.Label(btn, text=icon, font=F.ICON_M,
                            bg=C.BG_SIDEBAR, fg=C.TEXT_MUTED)
        icon_lbl.pack(side="left", padx=(0, 12))

        text_lbl = tk.Label(btn, text=label, font=(F.FAM, 11),
                            bg=C.BG_SIDEBAR, fg=C.TEXT_MUTED, anchor="w")
        text_lbl.pack(side="left", fill="x", expand=True)

        self._buttons[view_name] = {
            "row": row, "btn": btn, "icon": icon_lbl,
            "text": text_lbl, "indicator": indicator,
        }

        # Bind events
        for w in (btn, icon_lbl, text_lbl):
            w.bind("<Button-1>", lambda e, v=view_name: self._on_click(v))
            w.bind("<Enter>", lambda e, v=view_name: self._on_enter(v))
            w.bind("<Leave>", lambda e, v=view_name: self._on_leave(v))

    def _on_click(self, view_name):
        self.set_active(view_name)
        if self._on_navigate:
            self._on_navigate(view_name)

    def _on_enter(self, view_name):
        if view_name == self._active_view:
            return
        w = self._buttons[view_name]
        hover_bg = "#111827"
        for widget in (w["btn"], w["icon"], w["text"]):
            widget.configure(bg=hover_bg)
        w["text"].configure(fg=C.TEXT_SEC)

    def _on_leave(self, view_name):
        if view_name == self._active_view:
            return
        w = self._buttons[view_name]
        for widget in (w["btn"], w["icon"], w["text"]):
            widget.configure(bg=C.BG_SIDEBAR)
        w["text"].configure(fg=C.TEXT_MUTED)
        w["icon"].configure(fg=C.TEXT_MUTED)

    def set_active(self, view_name):
        self._active_view = view_name
        for vn, w in self._buttons.items():
            if vn == view_name:
                active_bg = "#0c1e3d"
                w["indicator"].configure(bg=C.ACCENT)
                for widget in (w["btn"], w["icon"], w["text"]):
                    widget.configure(bg=active_bg)
                w["icon"].configure(fg=C.ACCENT)
                w["text"].configure(fg=C.TEXT, font=(F.FAM, 11, "bold"))
            else:
                w["indicator"].configure(bg=C.BG_SIDEBAR)
                for widget in (w["btn"], w["icon"], w["text"]):
                    widget.configure(bg=C.BG_SIDEBAR)
                w["icon"].configure(fg=C.TEXT_MUTED)
                w["text"].configure(fg=C.TEXT_MUTED, font=(F.FAM, 11))
