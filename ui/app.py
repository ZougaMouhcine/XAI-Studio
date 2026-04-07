"""
XAI Studio — Main Application Window (v2 — Modern Dashboard)
===============================================================
Root Tk window with sidebar, view switching, and polished status bar.
"""

import tkinter as tk
from tkinter import ttk

from ui.theme import apply_theme
from ui.widgets import C, F
from ui.components.sidebar import Sidebar
from ui.components.status_bar import StatusBar
from ui.views.data_view import DataView
from ui.views.preprocessing_view import PreprocessingView
from ui.views.training_view import TrainingView
from ui.views.evaluation_view import EvaluationView
from ui.views.models_view import ModelsView
from utils.logger import set_ui_callback
from config.settings import APP_NAME, APP_VERSION, WINDOW_WIDTH, WINDOW_HEIGHT, MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT


class XAIStudioApp:
    """Main application controller."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} — v{APP_VERSION}")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.root.configure(bg=C.BG_ROOT)

        # Set window icon title
        self.root.option_add("*tearOff", False)

        # Apply theme
        apply_theme(self.root)

        # Build UI
        self._build_layout()

        # Connect logger → status bar
        set_ui_callback(self._on_log_message)

        # Start on Data view
        self._show_view("data")

    def _build_layout(self):
        # ── Main horizontal container ────────────────────────────
        main = tk.Frame(self.root, bg=C.BG_ROOT)
        main.pack(fill="both", expand=True)

        # ── Sidebar ──────────────────────────────────────────────
        self._sidebar = Sidebar(main, on_navigate=self._show_view)
        self._sidebar.pack(side="left", fill="y")

        # ── Subtle separator ─────────────────────────────────────
        sep = tk.Frame(main, bg=C.BORDER, width=1)
        sep.pack(side="left", fill="y")

        # ── Content area ─────────────────────────────────────────
        right = tk.Frame(main, bg=C.BG_MAIN)
        right.pack(side="left", fill="both", expand=True)

        # Top accent line
        accent_line = tk.Frame(right, bg=C.ACCENT, height=2)
        accent_line.pack(fill="x")

        self._content = tk.Frame(right, bg=C.BG_MAIN)
        self._content.pack(fill="both", expand=True)

        # ── Status bar ───────────────────────────────────────────
        self._status_bar = StatusBar(self.root)
        self._status_bar.pack(side="bottom", fill="x")

        # ── Create all views ─────────────────────────────────────
        self._views: dict[str, ttk.Frame] = {}
        self._create_views()

    def _create_views(self):
        view_classes = {
            "data": DataView,
            "preprocessing": PreprocessingView,
            "training": TrainingView,
            "evaluation": EvaluationView,
            "models": ModelsView,
        }
        for name, cls in view_classes.items():
            self._views[name] = cls(self._content)

    def _show_view(self, view_name: str):
        for view in self._views.values():
            view.pack_forget()

        view = self._views.get(view_name)
        if view:
            view.pack(fill="both", expand=True)
            self._sidebar.set_active(view_name)
            if hasattr(view, "on_enter"):
                view.on_enter()

    def _on_log_message(self, level: str, message: str):
        self.root.after(0, lambda: self._status_bar.set_message(level, message))

    def run(self):
        self.root.mainloop()
