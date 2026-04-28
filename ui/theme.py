"""
XAI Studio — UI Theme
======================
Professional ttk theme for both light and dark modes.
"""

import tkinter as tk
from tkinter import ttk
from ui.widgets import C, F


def apply_theme(root: tk.Tk):
    """Apply ttk styles based on the currently active color palette."""
    root.configure(bg=C.BG_ROOT)

    style = ttk.Style(root)
    style.theme_use("clam")

    # ── Base ──────────────────────────────────────────────────────────
    style.configure(".",
                     background=C.BG_MAIN, foreground=C.TEXT,
                     font=F.BODY, borderwidth=0, relief="flat")

    # ── Frames ────────────────────────────────────────────────────────
    style.configure("TFrame",          background=C.BG_MAIN)
    style.configure("Card.TFrame",     background=C.BG_CARD)
    style.configure("Surface.TFrame",  background=C.BG_SURFACE)
    style.configure("Sidebar.TFrame",  background=C.BG_SIDEBAR)

    # ── Labels ────────────────────────────────────────────────────────
    style.configure("TLabel",          background=C.BG_MAIN, foreground=C.TEXT, font=F.BODY)
    style.configure("Title.TLabel",    font=F.H1, foreground=C.TEXT)
    style.configure("Heading.TLabel",  font=F.H2, foreground=C.TEXT)
    style.configure("Sub.TLabel",      font=F.H3, foreground=C.TEXT)
    style.configure("Muted.TLabel",    font=F.SMALL, foreground=C.TEXT_MUTED)
    style.configure("Accent.TLabel",   font=F.H4, foreground=C.ACCENT)
    style.configure("Success.TLabel",  font=F.H4, foreground=C.SUCCESS)
    style.configure("Warning.TLabel",  font=F.H4, foreground=C.WARNING)
    style.configure("Danger.TLabel",   font=F.H4, foreground=C.DANGER)

    # Card-context labels
    for suffix in ("", "Muted.", "Heading.", "Accent."):
        base = f"Card{suffix}TLabel" if suffix else "Card.TLabel"
        fg_map = {"": C.TEXT, "Muted.": C.TEXT_SEC, "Heading.": C.TEXT, "Accent.": C.ACCENT}
        font_map = {"": F.BODY, "Muted.": F.SMALL, "Heading.": F.H3, "Accent.": F.H4}
        style.configure(base, background=C.BG_CARD,
                         foreground=fg_map.get(suffix, C.TEXT),
                         font=font_map.get(suffix, F.BODY))

    # ── Buttons ───────────────────────────────────────────────────────
    style.configure("TButton",
                background=C.BG_MAIN, foreground=C.TEXT,
                font=F.H4, padding=(16, 8), borderwidth=1,
                bordercolor=C.BORDER_LIGHT, relief="flat")
    style.map("TButton",
            background=[("active", C.BUTTON_SECONDARY_BG_HOVER), ("disabled", C.BG_MAIN)],
            bordercolor=[("active", C.BORDER_LIGHT), ("disabled", C.BORDER)],
              foreground=[("disabled", C.TEXT_DIM)])

    style.configure("Accent.TButton",
                     background=C.ACCENT, foreground="#ffffff",
                font=F.H4, padding=(16, 8), borderwidth=0, relief="flat")
    style.map("Accent.TButton",
            background=[("active", C.ACCENT_LIGHT), ("disabled", C.BORDER_LIGHT)],
              foreground=[("disabled", C.TEXT_DIM)])

    style.configure("Danger.TButton",
                background=C.BG_MAIN, foreground=C.DANGER,
                font=F.H4, padding=(16, 8), borderwidth=1,
                bordercolor=C.DANGER, relief="flat")
    style.map("Danger.TButton",
            background=[("active", C.BUTTON_DANGER_BG_HOVER), ("disabled", C.BG_MAIN)],
            bordercolor=[("active", C.DANGER), ("disabled", C.BORDER)],
            foreground=[("disabled", C.TEXT_DIM)])

    style.configure("Small.TButton",
                     padding=(12, 6), font=F.SMALL,
                background=C.BG_MAIN)
    style.map("Small.TButton",
            background=[("active", C.BUTTON_SECONDARY_BG_HOVER)])

    # ── Entry ─────────────────────────────────────────────────────────
    style.configure("TEntry",
                     fieldbackground=C.BG_INPUT, foreground=C.TEXT,
                     insertcolor=C.ACCENT, padding=8, font=F.BODY,
                borderwidth=1, relief="flat", bordercolor=C.INPUT_BORDER)
    style.map("TEntry",
            fieldbackground=[("focus", C.BG_INPUT)],
            bordercolor=[("focus", C.INPUT_FOCUS)],
              foreground=[("disabled", C.TEXT_DIM)])

    # ── Combobox ──────────────────────────────────────────────────────
    style.configure("TCombobox",
                fieldbackground=C.BG_INPUT, background=C.BG_INPUT,
                     foreground=C.TEXT, padding=8, font=F.BODY,
                arrowcolor=C.TEXT_SEC, borderwidth=1, bordercolor=C.INPUT_BORDER)
    style.map("TCombobox",
            fieldbackground=[("readonly", C.BG_INPUT), ("focus", C.BG_INPUT)],
            bordercolor=[("focus", C.INPUT_FOCUS), ("readonly", C.INPUT_BORDER)],
              foreground=[("readonly", C.TEXT)])

    # ── Treeview ──────────────────────────────────────────────────────
    style.configure("Treeview",
                     background=C.TREE_BG, foreground=C.TEXT,
                     fieldbackground=C.TREE_BG, font=F.SMALL,
                     rowheight=32, borderwidth=0)
    style.configure("Treeview.Heading",
                     background=C.BG_CARD_ALT, foreground=C.TEXT_SEC,
                     font=F.H4, borderwidth=0, relief="flat")
    style.map("Treeview",
              background=[("selected", C.TREE_SELECT)],
              foreground=[("selected", C.TEXT)])
    style.map("Treeview.Heading",
              background=[("active", C.BG_HOVER)])

    # ── Progressbar ───────────────────────────────────────────────────
    style.configure("TProgressbar",
                     troughcolor=C.BG_INPUT, background=C.ACCENT, thickness=6)
    style.configure("Green.Horizontal.TProgressbar",
                     troughcolor=C.BG_INPUT, background=C.SUCCESS)

    # ── Checkbutton ───────────────────────────────────────────────────
    style.configure("TCheckbutton",
                     background=C.BG_MAIN, foreground=C.TEXT, font=F.BODY,
                     indicatorbackground=C.BG_INPUT, indicatorforeground=C.ACCENT)
    style.map("TCheckbutton",
              background=[("active", C.BG_MAIN)],
              indicatorbackground=[("selected", C.ACCENT)])
    style.configure("Card.TCheckbutton", background=C.BG_CARD)
    style.map("Card.TCheckbutton", background=[("active", C.BG_CARD)])

    # ── Labelframe ────────────────────────────────────────────────────
    style.configure("TLabelframe",
                     background=C.BG_MAIN, foreground=C.TEXT,
                     bordercolor=C.BORDER, borderwidth=1)
    style.configure("TLabelframe.Label",
                     background=C.BG_MAIN, foreground=C.ACCENT, font=F.H3)

    # ── Separator ─────────────────────────────────────────────────────
    style.configure("TSeparator", background=C.BORDER)

    # ── Scrollbar ─────────────────────────────────────────────────────
    style.configure("Vertical.TScrollbar",
                     background=C.BG_SURFACE, troughcolor=C.BG_MAIN,
                     arrowcolor=C.TEXT_DIM, borderwidth=0, width=10)
    style.map("Vertical.TScrollbar",
              background=[("active", C.BG_HOVER)])
    style.configure("Horizontal.TScrollbar",
                     background=C.BG_SURFACE, troughcolor=C.BG_MAIN,
                     arrowcolor=C.TEXT_DIM, borderwidth=0, width=10)

    # ── Notebook ──────────────────────────────────────────────────────
    style.configure("TNotebook", background=C.BG_MAIN, borderwidth=0)
    style.configure("TNotebook.Tab",
                background=C.BG_CARD_ALT, foreground=C.TEXT_MUTED,
                     padding=(16, 8), font=F.H4)
    style.map("TNotebook.Tab",
            background=[("selected", C.BG_CARD)],
              foreground=[("selected", C.ACCENT)])

    # Shared option database values for native tk widgets
    root.option_add("*Background", C.BG_MAIN)
    root.option_add("*Foreground", C.TEXT)
    root.option_add("*selectBackground", C.TREE_SELECT)
    root.option_add("*selectForeground", C.TEXT)
