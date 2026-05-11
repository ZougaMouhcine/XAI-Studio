"""
XAI Studio — Sidebar Navigation
================================
Docker Desktop-inspired light sidebar navigation.
"""

import tkinter as tk

from ui.widgets import C, F
from services.i18n import _


class Sidebar(tk.Frame):
    """Sidebar with icon + label navigation buttons."""

    def _get_nav_items(self):
        return [
            (_("nav_data"), "data"),
            (_("nav_preprocessing"), "preprocessing"),
            (_("nav_training"), "training"),
            (_("nav_evaluation"), "evaluation"),
            (_("nav_prediction"), "prediction"),
            (_("nav_models"), "models"),
        ]

    def __init__(self, parent, on_navigate=None, on_toggle_theme=None):
        super().__init__(parent, bg=C.BG_SIDEBAR, width=248)
        self.pack_propagate(False)
        self._on_navigate = on_navigate
        self._on_toggle_theme = on_toggle_theme
        self._buttons: dict[str, dict] = {}
        self._active_view = None
        self._theme_btn = None
        self._theme_mode_lbl = None
        self._build()

    def _build(self):
        tk.Frame(self, bg=C.BG_SIDEBAR, height=24).pack(fill="x")

        workspace = tk.Frame(self, bg=C.BG_SIDEBAR)
        workspace.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(
            workspace,
            text="Workspace",
            font=(F.FAM, 12, "bold"),
            bg=C.BG_SIDEBAR,
            fg=C.SIDEBAR_TEXT,
        ).pack(side="left")
        tk.Label(
            workspace,
            text="BETA",
            font=(F.FAM, 8, "bold"),
            bg=C.ACCENT,
            fg="#ffffff",
            padx=8,
            pady=2,
        ).pack(side="left", padx=(10, 0))

        tk.Label(
            self,
            text="NAVIGATION",
            font=(F.FAM, 8, "bold"),
            bg=C.BG_SIDEBAR,
            fg=C.SIDEBAR_TEXT_DIM,
        ).pack(anchor="w", padx=16, pady=(8, 8))

        nav_frame = tk.Frame(self, bg=C.BG_SIDEBAR)
        nav_frame.pack(fill="x", padx=8)

        for label, view_name in self._get_nav_items():
            self._create_nav_button(nav_frame, label, view_name)

        tk.Frame(self, bg=C.BG_SIDEBAR).pack(fill="both", expand=True)

        footer = tk.Frame(self, bg=C.BG_SIDEBAR)
        footer.pack(fill="x", padx=16, pady=(0, 24))

        sep = tk.Frame(footer, bg=C.SIDEBAR_BORDER, height=1)
        sep.pack(fill="x", pady=(0, 16))

        self._theme_mode_lbl = tk.Label(
            footer,
            text=self._theme_label_text(),
            font=(F.FAM, 8, "bold"),
            bg=C.BG_SIDEBAR,
            fg=C.SIDEBAR_TEXT_DIM,
        )
        self._theme_mode_lbl.pack(anchor="w", pady=(0, 8))

        self._theme_btn = tk.Button(
            footer,
            text=self._theme_button_text(),
            font=(F.FAM, 9, "bold"),
            bg=C.BG_CARD,
            fg=C.TEXT,
            activebackground=C.SIDEBAR_HOVER,
            activeforeground=C.TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            command=self._toggle_theme,
        )
        self._theme_btn.pack(anchor="w")

    def _create_nav_button(self, parent, label, view_name):
        row = tk.Frame(parent, bg=C.BG_SIDEBAR, pady=2)
        row.pack(fill="x")

        btn = tk.Frame(row, bg=C.BG_SIDEBAR, padx=10, pady=8, cursor="hand2")
        btn.pack(fill="x", expand=True)

        icon_canvas = tk.Canvas(
            btn,
            width=20,
            height=20,
            bg=C.BG_SIDEBAR,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        icon_canvas.pack(side="left", padx=(0, 10))
        self._draw_icon(icon_canvas, view_name, C.SIDEBAR_TEXT_DIM)

        text_lbl = tk.Label(
            btn,
            text=label,
            font=(F.FAM, 10),
            bg=C.BG_SIDEBAR,
            fg=C.SIDEBAR_TEXT,
            anchor="w",
        )
        text_lbl.pack(side="left", fill="x", expand=True)

        self._buttons[view_name] = {
            "row": row,
            "btn": btn,
            "icon": icon_canvas,
            "text": text_lbl,
        }

        for w in (btn, icon_canvas, text_lbl):
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
        hover_bg = C.SIDEBAR_HOVER
        for widget in (w["btn"], w["icon"], w["text"]):
            widget.configure(bg=hover_bg)
        w["text"].configure(fg=C.SIDEBAR_TEXT)
        self._set_icon_color(w, C.SIDEBAR_TEXT_MUTED)

    def _on_leave(self, view_name):
        if view_name == self._active_view:
            return
        w = self._buttons[view_name]
        for widget in (w["btn"], w["icon"], w["text"]):
            widget.configure(bg=C.BG_SIDEBAR)
        w["text"].configure(fg=C.SIDEBAR_TEXT)
        self._set_icon_color(w, C.SIDEBAR_TEXT_DIM)

    def set_active(self, view_name):
        self._active_view = view_name
        for vn, w in self._buttons.items():
            if vn == view_name:
                active_bg = C.SIDEBAR_ACTIVE
                for widget in (w["btn"], w["icon"], w["text"]):
                    widget.configure(bg=active_bg)
                self._set_icon_color(w, C.ACCENT)
                w["text"].configure(fg=C.SIDEBAR_TEXT, font=(F.FAM, 11, "bold"))
            else:
                for widget in (w["btn"], w["icon"], w["text"]):
                    widget.configure(bg=C.BG_SIDEBAR)
                self._set_icon_color(w, C.SIDEBAR_TEXT_DIM)
                w["text"].configure(fg=C.SIDEBAR_TEXT, font=(F.FAM, 10))

    def _set_icon_color(self, button_def: dict, color: str):
        icon = button_def["icon"]
        for item in icon.find_withtag("stroke"):
            item_type = icon.type(item)
            if item_type == "line":
                icon.itemconfigure(item, fill=color)
            else:
                icon.itemconfigure(item, outline=color)

    def _draw_icon(self, canvas: tk.Canvas, view_name: str, color: str):
        canvas.delete("all")

        if view_name == "data":
            self._draw_data_icon(canvas, color)
        elif view_name == "preprocessing":
            self._draw_preprocessing_icon(canvas, color)
        elif view_name == "training":
            self._draw_training_icon(canvas, color)
        elif view_name == "evaluation":
            self._draw_evaluation_icon(canvas, color)
        elif view_name == "prediction":
            self._draw_prediction_icon(canvas, color)
        else:
            self._draw_models_icon(canvas, color)

    def _draw_data_icon(self, canvas: tk.Canvas, color: str):
        canvas.create_oval(2, 2, 12, 5, outline=color, width=1.4, tags="stroke")
        canvas.create_line(2, 3.5, 2, 11, fill=color, width=1.4, tags="stroke")
        canvas.create_line(12, 3.5, 12, 11, fill=color, width=1.4, tags="stroke")
        canvas.create_arc(2, 5.5, 12, 9.5, start=0, extent=-180, style="arc", outline=color, width=1.4, tags="stroke")
        canvas.create_arc(2, 9, 12, 13, start=0, extent=-180, style="arc", outline=color, width=1.4, tags="stroke")
        canvas.create_line(15, 7, 15, 13, fill=color, width=1.4, tags="stroke")
        canvas.create_line(13.5, 11.5, 15, 13, fill=color, width=1.4, tags="stroke")
        canvas.create_line(16.5, 11.5, 15, 13, fill=color, width=1.4, tags="stroke")

    def _draw_preprocessing_icon(self, canvas: tk.Canvas, color: str):
        canvas.create_line(2, 4, 17, 4, fill=color, width=1.4, tags="stroke")
        canvas.create_line(2, 10, 17, 10, fill=color, width=1.4, tags="stroke")
        canvas.create_line(2, 16, 17, 16, fill=color, width=1.4, tags="stroke")
        canvas.create_oval(5, 2.5, 8, 5.5, outline=color, width=1.4, tags="stroke")
        canvas.create_oval(10, 8.5, 13, 11.5, outline=color, width=1.4, tags="stroke")
        canvas.create_oval(7, 14.5, 10, 17.5, outline=color, width=1.4, tags="stroke")

    def _draw_training_icon(self, canvas: tk.Canvas, color: str):
        canvas.create_rectangle(4, 4, 14, 14, outline=color, width=1.4, tags="stroke")
        canvas.create_line(7, 7, 11, 9, fill=color, width=1.4, tags="stroke")
        canvas.create_line(11, 9, 7, 11, fill=color, width=1.4, tags="stroke")
        canvas.create_line(7, 11, 7, 7, fill=color, width=1.4, tags="stroke")
        for x in (6, 9, 12):
            canvas.create_line(x, 2, x, 4, fill=color, width=1.2, tags="stroke")
            canvas.create_line(x, 14, x, 16, fill=color, width=1.2, tags="stroke")

    def _draw_evaluation_icon(self, canvas: tk.Canvas, color: str):
        canvas.create_rectangle(3, 12, 5, 16, outline=color, width=1.4, tags="stroke")
        canvas.create_rectangle(8, 9, 10, 16, outline=color, width=1.4, tags="stroke")
        canvas.create_rectangle(13, 6, 15, 16, outline=color, width=1.4, tags="stroke")
        canvas.create_line(2, 11, 6, 8.5, 10, 10, 16, 5, fill=color, width=1.2, smooth=True, tags="stroke")

    def _draw_prediction_icon(self, canvas: tk.Canvas, color: str):
        canvas.create_line(2, 10, 14, 10, fill=color, width=1.4, tags="stroke")
        canvas.create_polygon(12, 6, 18, 10, 12, 14, outline=color, fill="", width=1.4, tags="stroke")
        canvas.create_oval(2, 6, 6, 10, outline=color, width=1.4, tags="stroke")

    def _draw_models_icon(self, canvas: tk.Canvas, color: str):
        canvas.create_rectangle(2, 3, 8, 9, outline=color, width=1.4, tags="stroke")
        canvas.create_rectangle(10, 3, 16, 9, outline=color, width=1.4, tags="stroke")
        canvas.create_rectangle(6, 11, 12, 17, outline=color, width=1.4, tags="stroke")
        canvas.create_line(8, 6, 10, 6, fill=color, width=1.2, tags="stroke")
        canvas.create_line(9, 9, 9, 11, fill=color, width=1.2, tags="stroke")

    def _theme_label_text(self):
        return _("theme_light").upper() if C.mode() == "light" else _("theme_dark").upper()

    def _theme_button_text(self):
        return f"☾  {_('theme_dark')}" if C.mode() == "light" else f"☀  {_('theme_light')}"

    def _toggle_theme(self):
        if self._on_toggle_theme:
            self._on_toggle_theme()
