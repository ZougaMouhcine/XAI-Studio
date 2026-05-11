"""
XAI Studio — Main Application Window (v2 — Modern Dashboard)
===============================================================
Root Tk window with sidebar, view switching, and polished status bar.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

from ui.theme import apply_theme
from ui.widgets import C
from ui.display import detect_ui_scale, apply_tk_scaling, try_configure_customtkinter
from ui.components.top_bar import TopBar
from ui.components.sidebar import Sidebar
from ui.components.status_bar import StatusBar
from ui.components.agent_panel import AgentPanel
from ui.views.data_view import DataView
from ui.views.preprocessing_view import PreprocessingView
from ui.views.training_view import TrainingView
from ui.views.evaluation_view import EvaluationView
from ui.views.prediction_view import PredictionView
from ui.views.models_view import ModelsView
from ui.views.upload_view import UploadView
from ui.views.xai_view import XAIView
from ui.views.visualization_view import VisualizationView
from ui.views.fairness_view import FairnessView
from services.agent_service import AgentService
from ui.components.agent_settings import load_agent_config
from utils.logger import set_ui_callback
from config.settings import APP_NAME, APP_VERSION, WINDOW_WIDTH, WINDOW_HEIGHT, MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT


class XAIStudioApp:
    """Main application controller."""

    def __init__(self):
        C.set_mode("light")
        self.root = tk.Tk()
        self.root.navigate_to = self._show_view
        self._ui_scale = detect_ui_scale(self.root)
        try:
            from ui.widgets import set_ui_scale

            set_ui_scale(self._ui_scale)
        except Exception:
            pass
        apply_tk_scaling(self.root, self._ui_scale)
        try_configure_customtkinter(self._ui_scale)
        self.root.title(f"{APP_NAME} — v{APP_VERSION}")
        width = int(WINDOW_WIDTH * self._ui_scale)
        height = int(WINDOW_HEIGHT * self._ui_scale)
        min_width = int(MIN_WINDOW_WIDTH * self._ui_scale)
        min_height = int(MIN_WINDOW_HEIGHT * self._ui_scale)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min_width, min_height)
        self.root.configure(bg=C.BG_ROOT)
        self._current_view = "data"
        self._main_container = None
        self._top_bar = None
        self._top_bar_divider = None
        self._content = None
        self._sidebar = None
        self._status_bar = None
        self._agent_panel = None
        self._agent_visible = False
        self._views: dict[str, ttk.Frame] = {}

        # ── Initialize Agent Service ─────────────────────────────────
        self._agent_service = AgentService()
        self._configure_agent()

        # Set window icon title
        self.root.option_add("*tearOff", False)

        # Configure named Tk fonts instead of a global string font value.
        base_size = max(10, int(round(10 * self._ui_scale)))
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

        # Apply theme
        apply_theme(self.root)

        # Build UI
        self._build_layout()

        # Connect logger → status bar
        set_ui_callback(self._on_log_message)

        # Start on Data view
        self._show_view(self._current_view)

    def _build_layout(self):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

        if self._main_container is not None:
            self._main_container.destroy()
            self._main_container = None
        if self._top_bar is not None:
            self._top_bar.destroy()
            self._top_bar = None
        if self._top_bar_divider is not None:
            self._top_bar_divider.destroy()
            self._top_bar_divider = None
        if self._status_bar is not None:
            self._status_bar.destroy()
            self._status_bar = None
        self._agent_panel = None

        # ── Top Header ──────────────────────────────────────────
        self._top_bar = TopBar(self.root, on_toggle_agent=self._toggle_agent_panel)
        self._top_bar.pack(side="top", fill="x")
        self._top_bar_divider = tk.Frame(self.root, bg=C.HEADER_BORDER, height=1)
        self._top_bar_divider.pack(side="top", fill="x")

        # ── Main horizontal container ────────────────────────────
        self._main_container = tk.Frame(self.root, bg=C.BG_ROOT)
        self._main_container.pack(fill="both", expand=True)

        # ── Sidebar ──────────────────────────────────────────────
        self._sidebar = Sidebar(
            self._main_container,
            on_navigate=self._show_view,
            on_toggle_theme=self._toggle_theme,
        )
        self._sidebar.pack(side="left", fill="y")

        # ── Subtle separator ─────────────────────────────────────
        sep = tk.Frame(self._main_container, bg=C.BORDER, width=1)
        sep.pack(side="left", fill="y")

        # ── Content area ─────────────────────────────────────────
        right = tk.Frame(self._main_container, bg=C.BG_MAIN)
        right.pack(side="left", fill="both", expand=True)

        # Top divider
        top_divider = tk.Frame(right, bg=C.DIVIDER, height=1)
        top_divider.pack(fill="x")

        self._content = tk.Frame(right, bg=C.BG_MAIN)
        self._content.pack(fill="both", expand=True)

        # ── Agent Panel (right side, togglable) ──────────────────
        self._agent_panel = AgentPanel(
            self._main_container,
            agent_service=self._agent_service,
            on_close=self._toggle_agent_panel,
        )
        if self._agent_visible:
            self._agent_panel.pack(side="right", fill="y")

        # ── Status bar ───────────────────────────────────────────
        self._status_bar = StatusBar(self.root)
        self._status_bar.pack(side="bottom", fill="x")

        # ── Create all views ─────────────────────────────────────
        self._views = {}
        self._create_views()

    def _create_views(self):
        view_classes = {
            "data": DataView,
            "preprocessing": PreprocessingView,
            "training": TrainingView,
            "evaluation": EvaluationView,
            "prediction": PredictionView,
            "models": ModelsView,
            "upload": UploadView,
            "xai": XAIView,
            "visualization": VisualizationView,
            "fairness": FairnessView,
        }
        for name, cls in view_classes.items():
            self._views[name] = cls(self._content)

    def _show_view(self, view_name: str):
        self._current_view = view_name
        for view in self._views.values():
            view.pack_forget()

        view = self._views.get(view_name)
        if view:
            view.pack(fill="both", expand=True)
            self._sidebar.set_active(view_name)
            if hasattr(view, "on_enter"):
                view.on_enter()

    def _toggle_theme(self):
        current = self._current_view
        new_mode = C.toggle_mode()
        apply_theme(self.root)
        self._build_layout()
        self._show_view(current)
        if self._status_bar:
            mode_text = "clair" if new_mode == "light" else "sombre"
            self._status_bar.set_message("INFO", f"Theme: mode {mode_text}")

    # ── Agent Panel ──────────────────────────────────────────────────

    def _configure_agent(self):
        """Load saved API keys and configure the agent service."""
        config = load_agent_config()
        self._agent_service.configure(
            groq_key=config.get("groq_key", ""),
            gemini_key=config.get("gemini_key", ""),
            groq_model=config.get("groq_model", "llama-3.3-70b-versatile"),
            gemini_model=config.get("gemini_model", "gemini-2.5-flash"),
            navigate_fn=self._show_view,
            preferred_provider=config.get("preferred_provider", ""),
        )

    def _toggle_agent_panel(self):
        """Show or hide the AI assistant panel."""
        self._agent_visible = not self._agent_visible
        if self._agent_panel:
            if self._agent_visible:
                self._agent_panel.pack(side="right", fill="y")
            else:
                self._agent_panel.pack_forget()
        # Update top bar button state
        if self._top_bar:
            self._top_bar.set_agent_active(self._agent_visible)

    def _on_log_message(self, level: str, message: str):
        self.root.after(
            0,
            lambda: self._status_bar.set_message(level, message) if self._status_bar else None,
        )

    def run(self):
        # Bind Ctrl+Shift+I to toggle agent panel
        self.root.bind_all("<Control-Shift-i>", lambda e: self._toggle_agent_panel())
        self.root.mainloop()
