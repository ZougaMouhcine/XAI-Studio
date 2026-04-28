"""Display and DPI helpers for high-quality desktop rendering."""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont


def enable_windows_dpi_awareness() -> None:
    """Enable DPI awareness on Windows before Tk initializes."""
    if not sys.platform.startswith("win"):
        return

    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def detect_ui_scale(root: tk.Tk) -> float:
    """Return a reasonable UI scale factor based on screen DPI."""
    try:
        dpi = float(root.winfo_fpixels("1i"))
    except Exception:
        dpi = 96.0
    scale = max(1.0, min(2.0, dpi / 96.0))
    return round(scale, 2)


def apply_tk_scaling(root: tk.Tk, scale: float) -> None:
    """Apply global Tk scaling and clean font rendering defaults."""
    try:
        root.tk.call("tk", "scaling", scale)
    except Exception:
        pass

    root.option_add("*tearOff", False)

    base_size = max(10, int(round(10 * scale)))
    for font_name, size, weight in (
        ("TkDefaultFont", base_size, None),
        ("TkTextFont", base_size, None),
        ("TkMenuFont", base_size, None),
        ("TkHeadingFont", base_size + 1, "bold"),
        ("TkCaptionFont", base_size - 1, None),
    ):
        try:
            cfg = tkfont.nametofont(font_name)
            cfg.configure(family="Segoe UI", size=size, weight=weight or "normal")
        except Exception:
            pass


def try_configure_customtkinter(scale: float) -> bool:
    """Configure customtkinter if available; return True when successful."""
    try:
        import customtkinter as ctk  # type: ignore

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        ctk.set_widget_scaling(scale)
        ctk.set_window_scaling(scale)
        return True
    except Exception:
        return False


