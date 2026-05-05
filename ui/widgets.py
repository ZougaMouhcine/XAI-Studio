"""
XAI Studio — Custom Widget Library
=====================================
Modern, theme-aware custom Tkinter widgets.
"""

import tkinter as tk
from tkinter import ttk


_UI_SCALE = 1.0


def set_ui_scale(scale: float):
    """Set the global UI scale used by helper sizing functions."""
    global _UI_SCALE
    _UI_SCALE = max(1.0, min(2.0, float(scale)))


def scaled(value: int | float) -> int:
    return max(1, int(round(float(value) * _UI_SCALE)))


def scaled_font(size: int, weight: str | None = None):
    if weight:
        return (F.FAM, scaled(size), weight)
    return (F.FAM, scaled(size))


# ──────────────────────────────────────────────────────────────────────
# Color Palette  —  Professional Light/Dark Design Tokens
# ──────────────────────────────────────────────────────────────────────
class C:
    """Centralized color tokens with runtime light/dark switching."""

    _MODE = "light"

    _PALETTES = {
        "light": {
            "BG_ROOT": "#F6F8FA",
            "BG_SIDEBAR": "#F3F4F6",
            "BG_MAIN": "#F6F8FA",
            "BG_CARD": "#ffffff",
            "BG_CARD_ALT": "#F9FAFB",
            "BG_INPUT": "#FFFFFF",
            "BG_HOVER": "#F3F4F6",
            "BG_SURFACE": "#FFFFFF",
            "BG_STATUS": "#FFFFFF",
            "ACCENT": "#1D63ED",
            "ACCENT_LIGHT": "#174FCC",
            "ACCENT_DIM": "#123FA7",
            "ACCENT_GLOW": "#1D63ED",
            "SUCCESS": "#1D63ED",
            "SUCCESS_DIM": "#EFF6FF",
            "WARNING": "#1D63ED",
            "WARNING_DIM": "#EFF6FF",
            "DANGER": "#1D63ED",
            "DANGER_DIM": "#EFF6FF",
            "INFO": "#1D63ED",
            "INFO_DIM": "#EFF6FF",
            "TEXT": "#111827",
            "TEXT_SEC": "#6B7280",
            "TEXT_MUTED": "#9CA3AF",
            "TEXT_DIM": "#9CA3AF",
            "BORDER": "#E5E7EB",
            "BORDER_LIGHT": "#D1D5DB",
            "DIVIDER": "#E5E7EB",
            "INPUT_BORDER": "#D1D5DB",
            "INPUT_FOCUS": "#1D63ED",
            "BUTTON_SECONDARY_BG_HOVER": "#F3F4F6",
            "BUTTON_DANGER_BG_HOVER": "#FEF2F2",
            "CARD_SHADOW": "#EEF2F7",
            "HEADER_BG": "#1D63ED",
            "HEADER_BG_HOVER": "#174FCC",
            "HEADER_TEXT": "#FFFFFF",
            "HEADER_TEXT_MUTED": "#DCE8FF",
            "HEADER_SURFACE": "#2A70EE",
            "HEADER_SURFACE_HOVER": "#3A7AF1",
            "HEADER_BORDER": "#2D74EF",
            "TREE_BG": "#ffffff",
            "TREE_SELECT": "#EFF6FF",
            "TREE_STRIPE": "#F9FAFB",
            "SIDEBAR_TEXT": "#111827",
            "SIDEBAR_TEXT_MUTED": "#4B5563",
            "SIDEBAR_TEXT_DIM": "#9CA3AF",
            "SIDEBAR_HOVER": "#E7ECF4",
            "SIDEBAR_ACTIVE": "#DDE6F6",
            "SIDEBAR_BORDER": "#E5E7EB",
        },
        "dark": {
            "BG_ROOT": "#0d1320",
            "BG_SIDEBAR": "#111827",
            "BG_MAIN": "#141d2d",
            "BG_CARD": "#1b263a",
            "BG_CARD_ALT": "#202e45",
            "BG_INPUT": "#22314a",
            "BG_HOVER": "#2a3d5b",
            "BG_SURFACE": "#23324a",
            "BG_STATUS": "#0f1724",
            "ACCENT": "#5f9cff",
            "ACCENT_LIGHT": "#84b5ff",
            "ACCENT_DIM": "#3f7ee5",
            "ACCENT_GLOW": "#7fb0ff",
            "SUCCESS": "#5f9cff",
            "SUCCESS_DIM": "#1f3557",
            "WARNING": "#5f9cff",
            "WARNING_DIM": "#1f3557",
            "DANGER": "#5f9cff",
            "DANGER_DIM": "#1f3557",
            "INFO": "#5f9cff",
            "INFO_DIM": "#1f3557",
            "TEXT": "#e9f0fb",
            "TEXT_SEC": "#c2d2e9",
            "TEXT_MUTED": "#9fb1ca",
            "TEXT_DIM": "#6f829e",
            "BORDER": "#2c3b55",
            "BORDER_LIGHT": "#3a4b67",
            "DIVIDER": "#2d3e59",
            "INPUT_BORDER": "#3a4b67",
            "INPUT_FOCUS": "#5f9cff",
            "BUTTON_SECONDARY_BG_HOVER": "#26354d",
            "BUTTON_DANGER_BG_HOVER": "#4a2a31",
            "CARD_SHADOW": "#111a2a",
            "HEADER_BG": "#174FCC",
            "HEADER_BG_HOVER": "#123FA7",
            "HEADER_TEXT": "#FFFFFF",
            "HEADER_TEXT_MUTED": "#C9DBFF",
            "HEADER_SURFACE": "#1F5CD3",
            "HEADER_SURFACE_HOVER": "#2C67D8",
            "HEADER_BORDER": "#2D6BDD",
            "TREE_BG": "#162236",
            "TREE_SELECT": "#2b4669",
            "TREE_STRIPE": "#19273d",
            "SIDEBAR_TEXT": "#e9f0fb",
            "SIDEBAR_TEXT_MUTED": "#9fb1ca",
            "SIDEBAR_TEXT_DIM": "#6f829e",
            "SIDEBAR_HOVER": "#1c2a40",
            "SIDEBAR_ACTIVE": "#203752",
            "SIDEBAR_BORDER": "#2c3b55",
        },
    }

    @classmethod
    def set_mode(cls, mode: str):
        """Set active palette mode and update class attributes in-place."""
        if mode not in cls._PALETTES:
            raise ValueError(f"Unsupported theme mode: {mode}")
        cls._MODE = mode
        for token, value in cls._PALETTES[mode].items():
            setattr(cls, token, value)

    @classmethod
    def toggle_mode(cls) -> str:
        """Toggle between light and dark mode and return the new mode."""
        cls.set_mode("dark" if cls._MODE == "light" else "light")
        return cls._MODE

    @classmethod
    def mode(cls) -> str:
        """Return current theme mode."""
        return cls._MODE


# Default to light mode unless changed by the app controller.
C.set_mode("light")


# ──────────────────────────────────────────────────────────────────────
# Typography
# ──────────────────────────────────────────────────────────────────────
class F:
    """Font presets."""
    FAM   = "Segoe UI"
    MONO  = "Cascadia Mono"

    H1    = (FAM, 18, "bold")
    H2    = (FAM, 15, "bold")
    H3    = (FAM, 12, "bold")
    H4    = (FAM, 11, "bold")
    BODY  = (FAM, 10)
    SMALL = (FAM, 9)
    TINY  = (FAM, 8)
    MONO_S = (MONO, 9)
    ICON_L = (FAM, 22)
    ICON_M = (FAM, 16)
    ICON_S = (FAM, 12)


# ──────────────────────────────────────────────────────────────────────
# RoundedFrame  —  card with subtle border + inner padding
# ──────────────────────────────────────────────────────────────────────
class Card(tk.Frame):
    """
    A styled panel / card with a coloured left-edge accent,
    background fill, and consistent inner padding.
    """

    def __init__(self, parent, accent_color=None, pad=16, **kw):
        bg = kw.pop("bg", C.BG_CARD)
        super().__init__(parent, bg=C.CARD_SHADOW, bd=0, highlightthickness=0, **kw)

        self._card = tk.Frame(
            self,
            bg=bg,
            highlightbackground=C.BORDER,
            highlightthickness=1,
            bd=0,
        )
        self._card.pack(fill="both", expand=True, padx=(0, 1), pady=(0, 1))

        if accent_color:
            bar = tk.Frame(self._card, bg=accent_color, width=scaled(3))
            bar.pack(side="left", fill="y")

        self._inner = tk.Frame(self._card, bg=bg)
        self._inner.pack(fill="both", expand=True, padx=scaled(pad), pady=scaled(pad))

    @property
    def inner(self):
        return self._inner


# ──────────────────────────────────────────────────────────────────────
# MetricTile  —  big number + label (for dashboards)
# ──────────────────────────────────────────────────────────────────────
class MetricTile(tk.Frame):
    """
    Display a single KPI / metric in a compact tile.

        ┌──────────────┐
        │  icon  value  │
        │     label     │
        └──────────────┘
    """

    def __init__(self, parent, icon="", value="—", label="", color=None, bg=None):
        color = color or C.ACCENT
        bg = bg or C.BG_CARD
        super().__init__(parent, bg=bg, highlightbackground=C.BORDER, highlightthickness=1)

        inner = tk.Frame(self, bg=bg)
        inner.pack(padx=14, pady=12)

        top = tk.Frame(inner, bg=bg)
        top.pack(anchor="center")

        if icon:
            tk.Label(top, text=icon, font=scaled_font(16), bg=bg, fg=color).pack(side="left", padx=(0, scaled(6)))

        self._value_lbl = tk.Label(top, text=value, font=scaled_font(18, "bold"), bg=bg, fg=color)
        self._value_lbl.pack(side="left")

        self._label_lbl = tk.Label(inner, text=label, font=F.SMALL, bg=bg, fg=C.TEXT_SEC)
        self._label_lbl.pack(anchor="center", pady=(2, 0))

    def set(self, value, label=None):
        self._value_lbl.configure(text=value)
        if label is not None:
            self._label_lbl.configure(text=label)


# ──────────────────────────────────────────────────────────────────────
# ModernButton
# ──────────────────────────────────────────────────────────────────────
class ModernButton(tk.Canvas):
    """
    A flat button drawn on a Canvas with hover / press colour transitions.
    Supports primary (filled accent), secondary (outline), and danger styles.
    """

    @staticmethod
    def _styles():
        return {
            "primary": {
                "fill": C.ACCENT,
                "fg": "#ffffff",
                "fill_hover": C.ACCENT_LIGHT,
                "fill_press": C.ACCENT_DIM,
                "outline": C.ACCENT,
                "outline_hover": C.ACCENT_LIGHT,
                "outline_press": C.ACCENT_DIM,
            },
            "secondary": {
                "fill": None,
                "fg": C.TEXT,
                "fill_hover": C.BUTTON_SECONDARY_BG_HOVER,
                "fill_press": C.BG_HOVER,
                "outline": C.BORDER_LIGHT,
                "outline_hover": C.BORDER_LIGHT,
                "outline_press": C.BORDER,
            },
            "danger": {
                "fill": None,
                "fg": C.DANGER,
                "fill_hover": C.BUTTON_DANGER_BG_HOVER,
                "fill_press": C.DANGER_DIM,
                "outline": C.DANGER,
                "outline_hover": C.DANGER,
                "outline_press": C.DANGER,
            },
            "success": {
                "fill": C.SUCCESS,
                "fg": C.SUCCESS,
                "fg_hover": "#ffffff",
                "fill_hover": "#0E9F6E",
                "fill_press": "#0A825A",
                "outline": C.SUCCESS,
                "outline_hover": "#0E9F6E",
                "outline_press": "#0A825A",
            },
            "ghost": {
                "fill": None,
                "fg": C.TEXT_SEC,
                "fill_hover": C.BG_HOVER,
                "fill_press": C.BG_CARD,
                "outline": C.BORDER,
                "outline_hover": C.BORDER,
                "outline_press": C.BORDER_LIGHT,
            },
        }

    def __init__(self, parent, text="", style="primary", command=None, width=None, icon="", **kw):
        styles = self._styles()
        s = styles.get(style, styles["primary"])
        self._fg = s["fg"]
        self._fg_hover = s.get("fg_hover", self._fg)
        self._radius = 8
        self._command = command

        display = f"{icon}  {text}".strip() if icon else text
        # Measure text width
        tmp = tk.Label(parent, text=display, font=scaled_font(11, "bold"))
        tw = tmp.winfo_reqwidth()
        tmp.destroy()
        w = width or (tw + 40)
        h = 38

        bg_parent = kw.pop("bg", self._get_parent_bg(parent))
        self._fill = s["fill"] if s["fill"] is not None else bg_parent
        self._fill_hover = s["fill_hover"]
        self._fill_press = s["fill_press"]
        self._outline = s["outline"]
        self._outline_hover = s["outline_hover"]
        self._outline_press = s["outline_press"]
        self._anim_after_id = None
        super().__init__(parent, width=w, height=h, bg=bg_parent,
                         highlightthickness=0, bd=0, cursor="hand2", **kw)

        self._rect = self._draw_rounded_rect(
            1,
            1,
            w - 1,
            h - 1,
            radius=self._radius,
            fill=self._fill,
            outline=self._outline,
            width=1,
        )
        self._text = self.create_text(w // 2, h // 2, text=display, fill=self._fg, font=scaled_font(11, "bold"))

        self.bind("<Enter>", lambda e: self._animate_to(self._fill_hover, self._outline_hover, self._fg_hover, duration_ms=170))
        self.bind("<Leave>", lambda e: self._animate_to(self._fill, self._outline, self._fg, duration_ms=170))
        self.bind("<ButtonPress-1>", lambda e: self._animate_to(self._fill_press, self._outline_press, self._fg_hover, duration_ms=90))
        self.bind("<ButtonRelease-1>", self._on_click)

    def _on_click(self, e):
        self._animate_to(self._fill_hover, self._outline_hover, self._fg_hover, duration_ms=120)
        if self._command:
            self._command()

    def _animate_to(self, fill, outline, text, duration_ms=150, steps=6):
        if self._anim_after_id is not None:
            self.after_cancel(self._anim_after_id)
            self._anim_after_id = None

        start_fill = self.itemcget(self._rect, "fill")
        start_outline = self.itemcget(self._rect, "outline")
        start_text = self.itemcget(self._text, "fill")

        def step(i):
            t = i / steps
            self.itemconfig(self._rect, fill=self._mix_hex(start_fill, fill, t))
            self.itemconfig(self._rect, outline=self._mix_hex(start_outline, outline, t))
            self.itemconfig(self._text, fill=self._mix_hex(start_text, text, t))
            if i < steps:
                self._anim_after_id = self.after(max(1, duration_ms // steps), lambda: step(i + 1))
            else:
                self._anim_after_id = None

        step(1)

    @staticmethod
    def _mix_hex(a, b, t):
        a = a.strip()
        b = b.strip()
        if not (a.startswith("#") and b.startswith("#") and len(a) == 7 and len(b) == 7):
            return b
        ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
        br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
        r = int(ar + (br - ar) * t)
        g = int(ag + (bg - ag) * t)
        bl = int(ab + (bb - ab) * t)
        return f"#{r:02x}{g:02x}{bl:02x}"

    def _draw_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, splinesteps=24, **kwargs)

    @staticmethod
    def _get_parent_bg(widget):
        try:
            return widget.cget("bg")
        except Exception:
            return C.BG_MAIN


# ──────────────────────────────────────────────────────────────────────
# SectionHeader
# ──────────────────────────────────────────────────────────────────────
class SectionHeader(tk.Frame):
    """A section header with icon, title, and optional subtitle."""

    def __init__(self, parent, icon="", title="", subtitle="", bg=None):
        bg = bg or C.BG_MAIN
        super().__init__(parent, bg=bg)

        left = tk.Frame(self, bg=bg)
        left.pack(side="left", fill="x", expand=True)

        top_row = tk.Frame(left, bg=bg)
        top_row.pack(anchor="w")

        if icon:
            tk.Label(top_row, text=icon, font=scaled_font(20), bg=bg, fg=C.ACCENT).pack(side="left", padx=(0, scaled(10)))

        tk.Label(top_row, text=title, font=scaled_font(18, "bold"), bg=bg, fg=C.TEXT).pack(side="left")

        if subtitle:
            tk.Label(left, text=subtitle, font=scaled_font(9), bg=bg, fg=C.TEXT_MUTED).pack(anchor="w", pady=(scaled(2), 0))


# ──────────────────────────────────────────────────────────────────────
# Badge / Chip
# ──────────────────────────────────────────────────────────────────────
class Badge(tk.Frame):
    """Small coloured badge / chip (e.g. "Classification", "OK")."""

    def __init__(self, parent, text="", color=None, bg_parent=None):
        color = color or C.ACCENT
        dim = self._dim(color)
        super().__init__(parent, bg=dim, padx=8, pady=2)
        tk.Label(self, text=text, font=F.TINY, bg=dim, fg=color).pack()

    @staticmethod
    def _dim(hex_color):
        """Generate a very dim version of the colour for the background."""
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        return f"#{r // 5:02x}{g // 5:02x}{b // 5:02x}"


# ──────────────────────────────────────────────────────────────────────
# StyledTreeview  —  modern-looking treeview
# ──────────────────────────────────────────────────────────────────────
class StyledTreeview(tk.Frame):
    """A Treeview wrapped in a frame with scrollbars and modern colours."""

    def __init__(self, parent, columns, col_widths=None, height=10, selectmode="browse", **kw):
        bg = kw.pop("bg", C.BG_MAIN)
        super().__init__(parent, bg=bg)

        style = ttk.Style()
        style_name = f"Custom{id(self)}.Treeview"
        style.configure(style_name,
                        background=C.TREE_BG, foreground=C.TEXT,
                        fieldbackground=C.TREE_BG, font=F.SMALL,
                        rowheight=32, borderwidth=0)
        style.configure(f"{style_name}.Heading",
                        background=C.BG_CARD, foreground=C.TEXT_SEC,
                        font=F.H4, borderwidth=0, relief="flat")
        style.map(style_name,
                  background=[("selected", C.TREE_SELECT)],
                  foreground=[("selected", C.TEXT)])
        style.map(f"{style_name}.Heading",
                  background=[("active", C.BG_HOVER)])

        container = tk.Frame(self, bg=C.BORDER, padx=1, pady=1)
        container.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            container,
            columns=columns,
            show="headings",
            height=height,
            style=style_name,
            selectmode=selectmode,
        )
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        widths = col_widths or {}
        for col in columns:
            w = widths.get(col, 120)
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, width=w, minwidth=60, anchor="center")


# ──────────────────────────────────────────────────────────────────────
# LogPanel  —  styled output / log area
# ──────────────────────────────────────────────────────────────────────
class LogPanel(tk.Frame):
    """Monospace text area styled as a terminal / log panel."""

    def __init__(self, parent, height=6, label="Output", bg_outer=None, scrollbar=False):
        bg_outer = bg_outer or C.BG_MAIN
        super().__init__(parent, bg=bg_outer)

        if label:
            tk.Label(self, text=label, font=F.H4, bg=bg_outer, fg=C.TEXT_SEC).pack(anchor="w", pady=(0, 6))

        border = tk.Frame(self, bg=C.BORDER, padx=1, pady=1)
        border.pack(fill="both", expand=True)

        self._text = tk.Text(
            border, height=height, bg=C.TREE_BG, fg=C.TEXT,
            font=F.MONO_S, relief="flat", padx=12, pady=10,
            wrap="word", insertbackground=C.TEXT, selectbackground=C.TREE_SELECT,
        )

        if scrollbar:
            vsb = ttk.Scrollbar(border, orient="vertical", command=self._text.yview)
            self._text.configure(yscrollcommand=vsb.set)
            self._text.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            border.rowconfigure(0, weight=1)
            border.columnconfigure(0, weight=1)
        else:
            self._text.pack(fill="both", expand=True)

    def set_content(self, content: str):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", content)
        self._text.configure(state="disabled")

    def append(self, line: str):
        self._text.configure(state="normal")
        self._text.insert("end", line + "\n")
        self._text.see("end")
        self._text.configure(state="disabled")


# ──────────────────────────────────────────────────────────────────────
# PipelineStep  —  a labelled step with status dot
# ──────────────────────────────────────────────────────────────────────
class PipelineStep(tk.Frame):
    """A single pipeline step indicator: ● label."""

    def __init__(self, parent, text="", status="pending", bg=None):
        bg = bg or C.BG_CARD
        super().__init__(parent, bg=bg)
        self._status_colors = {
            "pending": C.TEXT_DIM,
            "active": C.ACCENT,
            "done": C.ACCENT,
            "error": C.ACCENT,
        }
        self._dot = tk.Label(self, text="●", font=F.ICON_S, bg=bg,
                              fg=self._status_colors.get(status, C.TEXT_DIM))
        self._dot.pack(side="left", padx=(0, 8))
        self._lbl = tk.Label(self, text=text, font=F.BODY, bg=bg, fg=C.TEXT)
        self._lbl.pack(side="left")

    def set_status(self, status: str):
        self._dot.configure(fg=self._status_colors.get(status, C.TEXT_DIM))


# ──────────────────────────────────────────────────────────────────────
# Tooltip
# ──────────────────────────────────────────────────────────────────────
class Tooltip:
    """Lightweight hover tooltip for Tk widgets."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 450):
        self._widget = widget
        self._text = text
        self._delay_ms = delay_ms
        self._after_id = None
        self._tip = None

        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _=None):
        if not self._text:
            return
        self._cancel()
        self._after_id = self._widget.after(self._delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip is not None:
            return
        x = self._widget.winfo_rootx() + 12
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 8
        self._tip = tk.Toplevel(self._widget)
        self._tip.overrideredirect(True)
        self._tip.attributes("-topmost", True)
        self._tip.configure(bg=C.BG_CARD)
        label = tk.Label(
            self._tip,
            text=self._text,
            bg=C.BG_CARD,
            fg=C.TEXT,
            font=F.TINY,
            padx=8,
            pady=4,
            relief="solid",
            bd=1,
        )
        label.pack()
        self._tip.geometry(f"+{x}+{y}")

    def _hide(self, _=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def bind_mousewheel_to(canvas: tk.Canvas, scope_widget: tk.Widget):
    """Bind mouse wheel scrolling only when cursor is over the view scope."""

    def _on_wheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")

    def _bind(_):
        canvas.bind_all("<MouseWheel>", _on_wheel)
        canvas.bind_all("<Button-4>", _on_wheel)
        canvas.bind_all("<Button-5>", _on_wheel)

    def _unbind(_):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    scope_widget.bind("<Enter>", _bind)
    scope_widget.bind("<Leave>", _unbind)
