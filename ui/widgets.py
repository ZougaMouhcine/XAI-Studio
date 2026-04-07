"""
XAI Studio — Custom Widget Library
=====================================
Modern, visually polished custom Tkinter widgets that go far beyond
the default look. Rounded cards, gradient buttons, badges, metric
tiles, and more.
"""

import tkinter as tk
from tkinter import ttk


# ──────────────────────────────────────────────────────────────────────
# Color Palette  —  Modern AI / Analytics Dashboard
# ──────────────────────────────────────────────────────────────────────
class C:
    """Centralised colour constants."""

    # Backgrounds
    BG_ROOT      = "#0b1120"
    BG_SIDEBAR   = "#070d1a"
    BG_MAIN      = "#0f1729"
    BG_CARD      = "#162032"
    BG_CARD_ALT  = "#1a2742"
    BG_INPUT     = "#1e293b"
    BG_HOVER     = "#1e3a5f"
    BG_SURFACE   = "#1e293b"

    # Accent
    ACCENT       = "#3b82f6"
    ACCENT_LIGHT = "#60a5fa"
    ACCENT_DIM   = "#1d4ed8"
    ACCENT_GLOW  = "#2563eb"

    # Semantic
    SUCCESS      = "#10b981"
    SUCCESS_DIM  = "#064e3b"
    WARNING      = "#f59e0b"
    WARNING_DIM  = "#78350f"
    DANGER       = "#ef4444"
    DANGER_DIM   = "#7f1d1d"
    INFO         = "#06b6d4"
    INFO_DIM     = "#164e63"

    # Text
    TEXT         = "#e2e8f0"
    TEXT_SEC     = "#94a3b8"
    TEXT_MUTED   = "#64748b"
    TEXT_DIM     = "#475569"

    # Borders / Dividers
    BORDER       = "#1e293b"
    BORDER_LIGHT = "#334155"
    DIVIDER      = "#1e293b"

    # Treeview
    TREE_BG      = "#111c2e"
    TREE_SELECT  = "#1e3a5f"
    TREE_STRIPE  = "#0f1729"


# ──────────────────────────────────────────────────────────────────────
# Typography
# ──────────────────────────────────────────────────────────────────────
class F:
    """Font presets."""
    FAM   = "Segoe UI"
    MONO  = "Cascadia Code"

    H1    = (FAM, 22, "bold")
    H2    = (FAM, 16, "bold")
    H3    = (FAM, 13, "bold")
    H4    = (FAM, 11, "bold")
    BODY  = (FAM, 11)
    SMALL = (FAM, 10)
    TINY  = (FAM, 9)
    MONO_S = (MONO, 10)
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
        super().__init__(parent, bg=bg, highlightbackground=C.BORDER,
                         highlightthickness=1, **kw)

        if accent_color:
            bar = tk.Frame(self, bg=accent_color, width=4)
            bar.pack(side="left", fill="y")

        self._inner = tk.Frame(self, bg=bg)
        self._inner.pack(fill="both", expand=True, padx=pad, pady=pad)

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

    def __init__(self, parent, icon="", value="—", label="", color=C.ACCENT, bg=C.BG_CARD):
        super().__init__(parent, bg=bg, highlightbackground=C.BORDER, highlightthickness=1)

        inner = tk.Frame(self, bg=bg)
        inner.pack(padx=14, pady=12)

        top = tk.Frame(inner, bg=bg)
        top.pack(anchor="w")

        if icon:
            tk.Label(top, text=icon, font=F.ICON_M, bg=bg, fg=color).pack(side="left", padx=(0, 6))

        self._value_lbl = tk.Label(top, text=value, font=(F.FAM, 20, "bold"), bg=bg, fg=color)
        self._value_lbl.pack(side="left")

        self._label_lbl = tk.Label(inner, text=label, font=F.SMALL, bg=bg, fg=C.TEXT_SEC)
        self._label_lbl.pack(anchor="w", pady=(2, 0))

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

    STYLES = {
        "primary":   {"bg": C.ACCENT,     "fg": "#ffffff", "hover": C.ACCENT_LIGHT, "press": C.ACCENT_DIM},
        "secondary": {"bg": C.BG_SURFACE, "fg": C.TEXT,    "hover": C.BG_HOVER,     "press": C.BG_CARD},
        "danger":    {"bg": C.DANGER_DIM, "fg": "#fca5a5", "hover": C.DANGER,       "press": "#991b1b"},
        "success":   {"bg": C.SUCCESS_DIM,"fg": "#6ee7b7", "hover": C.SUCCESS,      "press": "#065f46"},
        "ghost":     {"bg": C.BG_MAIN,    "fg": C.TEXT_SEC,"hover": C.BG_CARD,      "press": C.BG_SURFACE},
    }

    def __init__(self, parent, text="", style="primary", command=None, width=None, icon="", **kw):
        s = self.STYLES.get(style, self.STYLES["primary"])
        self._bg = s["bg"]; self._fg = s["fg"]
        self._hover = s["hover"]; self._press = s["press"]
        self._command = command

        display = f"{icon}  {text}".strip() if icon else text
        # Measure text width
        tmp = tk.Label(parent, text=display, font=F.H4)
        tw = tmp.winfo_reqwidth()
        tmp.destroy()
        w = width or (tw + 40)
        h = 38

        bg_parent = kw.pop("bg", self._get_parent_bg(parent))
        super().__init__(parent, width=w, height=h, bg=bg_parent,
                         highlightthickness=0, bd=0, cursor="hand2", **kw)

        self._rect = self.create_rectangle(0, 0, w, h, fill=self._bg, outline="", width=0)
        self._text = self.create_text(w // 2, h // 2, text=display, fill=self._fg, font=F.H4)

        self.bind("<Enter>", lambda e: (self.itemconfig(self._rect, fill=self._hover),
                                         self.itemconfig(self._text, fill="#ffffff")))
        self.bind("<Leave>", lambda e: (self.itemconfig(self._rect, fill=self._bg),
                                         self.itemconfig(self._text, fill=self._fg)))
        self.bind("<ButtonPress-1>", lambda e: self.itemconfig(self._rect, fill=self._press))
        self.bind("<ButtonRelease-1>", self._on_click)

    def _on_click(self, e):
        self.itemconfig(self._rect, fill=self._hover)
        if self._command:
            self._command()

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

    def __init__(self, parent, icon="", title="", subtitle="", bg=C.BG_MAIN):
        super().__init__(parent, bg=bg)

        left = tk.Frame(self, bg=bg)
        left.pack(side="left", fill="x", expand=True)

        top_row = tk.Frame(left, bg=bg)
        top_row.pack(anchor="w")

        if icon:
            tk.Label(top_row, text=icon, font=F.ICON_L, bg=bg, fg=C.ACCENT).pack(side="left", padx=(0, 10))

        tk.Label(top_row, text=title, font=F.H1, bg=bg, fg=C.TEXT).pack(side="left")

        if subtitle:
            tk.Label(left, text=subtitle, font=F.SMALL, bg=bg, fg=C.TEXT_MUTED).pack(anchor="w", pady=(2, 0))


# ──────────────────────────────────────────────────────────────────────
# Badge / Chip
# ──────────────────────────────────────────────────────────────────────
class Badge(tk.Frame):
    """Small coloured badge / chip (e.g. "Classification", "OK")."""

    def __init__(self, parent, text="", color=C.ACCENT, bg_parent=C.BG_CARD):
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

    def __init__(self, parent, columns, col_widths=None, height=10, **kw):
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
                  foreground=[("selected", "#ffffff")])
        style.map(f"{style_name}.Heading",
                  background=[("active", C.BG_HOVER)])

        container = tk.Frame(self, bg=C.BORDER, padx=1, pady=1)
        container.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(container, columns=columns, show="headings",
                                  height=height, style=style_name, selectmode="browse")
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
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, minwidth=60)


# ──────────────────────────────────────────────────────────────────────
# LogPanel  —  styled output / log area
# ──────────────────────────────────────────────────────────────────────
class LogPanel(tk.Frame):
    """Scrollable monospace text area styled as a terminal / log panel."""

    def __init__(self, parent, height=6, label="Output", bg_outer=C.BG_MAIN):
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

    STATUS_COLORS = {"pending": C.TEXT_DIM, "active": C.WARNING, "done": C.SUCCESS, "error": C.DANGER}

    def __init__(self, parent, text="", status="pending", bg=C.BG_CARD):
        super().__init__(parent, bg=bg)
        self._dot = tk.Label(self, text="●", font=F.ICON_S, bg=bg,
                              fg=self.STATUS_COLORS.get(status, C.TEXT_DIM))
        self._dot.pack(side="left", padx=(0, 8))
        self._lbl = tk.Label(self, text=text, font=F.BODY, bg=bg, fg=C.TEXT)
        self._lbl.pack(side="left")

    def set_status(self, status: str):
        self._dot.configure(fg=self.STATUS_COLORS.get(status, C.TEXT_DIM))
