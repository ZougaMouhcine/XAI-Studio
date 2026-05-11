"""
XAI Studio -- Agent Settings Dialog
====================================
Modal dialog for entering Groq / Gemini API keys with
inline validation and dynamic model fetching.
"""

import tkinter as tk
from tkinter import ttk
import json
import os
import threading

from ui.widgets import C, F, scaled_font
from ui.components.dialogs import apply_popup_geometry
from config.settings import BASE_DIR
from utils.logger import get_logger
from services.i18n import _

logger = get_logger(__name__)

AGENT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "agent.json")


def load_agent_config() -> dict:
    """Load saved agent config (API keys, model preferences)."""
    try:
        if os.path.exists(AGENT_CONFIG_PATH):
            with open(AGENT_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load agent config: %s", exc)
    return {}


def save_agent_config(config: dict):
    """Persist agent config to disk."""
    try:
        os.makedirs(os.path.dirname(AGENT_CONFIG_PATH), exist_ok=True)
        with open(AGENT_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("Agent config saved to %s", AGENT_CONFIG_PATH)
    except Exception as exc:
        logger.error("Failed to save agent config: %s", exc)


class AgentSettingsDialog(tk.Toplevel):
    """Modal dialog for entering API keys with inline test + dynamic model list."""

    def __init__(self, parent, agent_service, on_save=None):
        super().__init__(parent)
        self._agent = agent_service
        self._on_save = on_save

        self.title(_("settings_title"))
        self.configure(bg=C.BG_CARD)
        self.resizable(False, False)

        apply_popup_geometry(self, parent)

        self.transient(parent)
        self.grab_set()

        # Load existing config
        self._config = load_agent_config()

        self._build()

    def _build(self):
        bg = C.BG_CARD
        pad = 24

        # ── Title ────────────────────────────────────────────────────
        title_frame = tk.Frame(self, bg=bg)
        title_frame.pack(fill="x", padx=pad, pady=(pad, 0))

        tk.Label(
            title_frame, text=_("settings_title"),
            font=scaled_font(14, "bold"), bg=bg, fg=C.TEXT,
        ).pack(side="left")

        # ── Info banner ──────────────────────────────────────────────
        info_frame = tk.Frame(self, bg=C.INFO_DIM)
        info_frame.pack(fill="x", padx=pad, pady=(16, 0))

        tk.Label(
            info_frame,
            text=_("settings_desc"),
            font=(F.FAM, 9), bg=C.INFO_DIM, fg=C.TEXT_SEC,
            justify="left", padx=12, pady=8,
        ).pack(fill="x")

        # ── Groq Section ─────────────────────────────────────────────
        self._groq_status = None
        self._groq_key_var, self._groq_model_var, self._groq_status = (
            self._build_provider_section(
                title="Groq API",
                key_field="groq_key",
                model_field="groq_model",
                default_model="llama-3.3-70b-versatile",
                description="Free tier -- No credit card required",
                test_fn=self._test_groq,
            )
        )

        # ── Gemini Section ───────────────────────────────────────────
        self._gemini_key_var, self._gemini_model_var, self._gemini_status = (
            self._build_provider_section(
                title="Google Gemini API",
                key_field="gemini_key",
                model_field="gemini_model",
                default_model="gemini-2.5-flash",
                description="Free tier -- Via Google AI Studio",
                test_fn=self._test_gemini,
            )
        )

        # ── Active Provider Section ──────────────────────────────────
        pref_section = tk.Frame(self, bg=bg)
        pref_section.pack(fill="x", padx=pad, pady=(12, 0))

        pref_row = tk.Frame(pref_section, bg=bg)
        pref_row.pack(fill="x")

        tk.Label(
            pref_row, text=_("active_provider"), font=scaled_font(11, "bold"),
            bg=bg, fg=C.TEXT,
        ).pack(side="left")

        tk.Label(
            pref_row, text=_("active_provider_desc"), font=(F.FAM, 8),
            bg=bg, fg=C.TEXT_MUTED,
        ).pack(side="right")

        self._preferred_var = tk.StringVar(
            value=self._config.get("preferred_provider", "Groq")
        )
        pref_combo = ttk.Combobox(
            pref_section, textvariable=self._preferred_var,
            values=["Groq", "Gemini"],
            state="readonly", font=(F.FAM, 9),
        )
        pref_combo.pack(fill="x", pady=(6, 0))

        # ── Buttons ──────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=bg)
        btn_frame.pack(fill="x", padx=pad, pady=(16, pad))

        # Status
        self._status_label = tk.Label(
            btn_frame, text="", font=(F.FAM, 9), bg=bg, fg=C.TEXT_SEC,
        )
        self._status_label.pack(side="left")

        # Cancel
        cancel_btn = tk.Label(
            btn_frame, text=_("cancel"), font=scaled_font(10, "bold"),
            bg=C.BG_CARD_ALT, fg=C.TEXT_SEC, padx=16, pady=6, cursor="hand2",
        )
        cancel_btn.pack(side="right", padx=(8, 0))
        cancel_btn.bind("<Button-1>", lambda e: self.destroy())

        # Save
        save_btn = tk.Label(
            btn_frame, text=_("save_apply"), font=scaled_font(10, "bold"),
            bg=C.ACCENT, fg="#ffffff", padx=16, pady=6, cursor="hand2",
        )
        save_btn.pack(side="right")
        save_btn.bind("<Button-1>", lambda e: self._save())
        save_btn.bind("<Enter>", lambda e: save_btn.configure(bg=C.ACCENT_LIGHT))
        save_btn.bind("<Leave>", lambda e: save_btn.configure(bg=C.ACCENT))

    def _build_provider_section(self, title, key_field, model_field,
                                 default_model, description, test_fn):
        """Build a provider section with key input, test button, and model dropdown.

        Returns (key_var, model_var, status_label).
        """
        bg = C.BG_CARD
        pad = 24

        section = tk.Frame(self, bg=bg)
        section.pack(fill="x", padx=pad, pady=(12, 0))

        # Title row
        title_row = tk.Frame(section, bg=bg)
        title_row.pack(fill="x")

        tk.Label(
            title_row, text=title, font=scaled_font(11, "bold"),
            bg=bg, fg=C.TEXT,
        ).pack(side="left")

        tk.Label(
            title_row, text=description, font=(F.FAM, 8),
            bg=bg, fg=C.TEXT_MUTED,
        ).pack(side="right")

        # API Key label
        tk.Label(
            section, text=_("api_key"), font=(F.FAM, 9, "bold"),
            bg=bg, fg=C.TEXT_SEC,
        ).pack(anchor="w", pady=(8, 2))

        # Key input + Test button row
        key_row = tk.Frame(section, bg=bg)
        key_row.pack(fill="x")

        key_var = tk.StringVar(value=self._config.get(key_field, ""))
        key_entry = tk.Entry(
            key_row, textvariable=key_var, font=(F.MONO, 9),
            bg=C.BG_INPUT, fg=C.TEXT, relief="flat",
            insertbackground=C.TEXT, show="*",
            highlightbackground=C.INPUT_BORDER, highlightthickness=1,
        )
        key_entry.pack(side="left", fill="x", expand=True, ipady=6)

        test_btn = tk.Label(
            key_row, text=_("test_btn"), font=(F.FAM, 9, "bold"),
            bg=C.BG_CARD_ALT, fg=C.ACCENT, padx=12, pady=6, cursor="hand2",
        )
        test_btn.pack(side="right", padx=(6, 0))
        test_btn.bind("<Button-1>", lambda e: test_fn())
        test_btn.bind("<Enter>", lambda e: test_btn.configure(bg=C.BG_HOVER))
        test_btn.bind("<Leave>", lambda e: test_btn.configure(bg=C.BG_CARD_ALT))

        # Toggle visibility
        show_var = tk.BooleanVar(value=False)
        def toggle_show(sv=show_var, entry=key_entry):
            entry.configure(show="" if sv.get() else "*")
        show_check = tk.Checkbutton(
            section, text=_("show_key"), variable=show_var, command=toggle_show,
            font=(F.FAM, 8), bg=bg, fg=C.TEXT_MUTED,
            selectcolor=C.BG_INPUT, activebackground=bg,
        )
        show_check.pack(anchor="w")

        # Status label (test result feedback)
        status_lbl = tk.Label(
            section, text="", font=(F.FAM, 8, "bold"), bg=bg, fg=C.TEXT_SEC,
        )
        status_lbl.pack(anchor="w")

        # Model selector
        tk.Label(
            section, text=_("model_label"), font=(F.FAM, 9, "bold"),
            bg=bg, fg=C.TEXT_SEC,
        ).pack(anchor="w", pady=(4, 2))

        model_var = tk.StringVar(value=self._config.get(model_field, default_model))
        model_combo = ttk.Combobox(
            section, textvariable=model_var, values=[default_model],
            state="readonly", font=(F.FAM, 9),
        )
        model_combo.pack(fill="x")

        # Store references for dynamic updates
        setattr(self, f"_{key_field}_var", key_var)
        setattr(self, f"_{model_field}_var", model_var)
        setattr(self, f"_{model_field}_combo", model_combo)

        return key_var, model_var, status_lbl

    # ── Test functions ───────────────────────────────────────────────

    def _test_groq(self):
        """Test Groq API key and fetch models on success."""
        key = self._groq_key_var.get().strip()
        if not key:
            self._groq_status.configure(text="Enter a key first", fg=C.WARNING)
            return
        self._groq_status.configure(text="Testing...", fg=C.TEXT_SEC)
        self.update_idletasks()

        def _run():
            from services.llm_client import groq_test_key, groq_list_models
            success, msg = groq_test_key(key)
            models = groq_list_models(key) if success else []
            self.after(0, lambda: self._on_groq_test_done(success, msg, models))

        threading.Thread(target=_run, daemon=True).start()

    def _on_groq_test_done(self, success: bool, msg: str, models: list[str]):
        if success:
            model_count = len(models)
            self._groq_status.configure(
                text=f"{msg} ({model_count} models)", fg=C.ACCENT,
            )
            if models:
                combo = self._groq_model_combo
                combo.configure(values=models)
                current = self._groq_model_var.get()
                if current not in models:
                    self._groq_model_var.set(models[0])
        else:
            self._groq_status.configure(text=msg, fg=C.DANGER)

    def _test_gemini(self):
        """Test Gemini API key and fetch models on success."""
        key = self._gemini_key_var.get().strip()
        if not key:
            self._gemini_status.configure(text="Enter a key first", fg=C.WARNING)
            return
        self._gemini_status.configure(text="Testing...", fg=C.TEXT_SEC)
        self.update_idletasks()

        def _run():
            from services.llm_client import gemini_test_key, gemini_list_models
            success, msg = gemini_test_key(key)
            models = gemini_list_models(key) if success else []
            self.after(0, lambda: self._on_gemini_test_done(success, msg, models))

        threading.Thread(target=_run, daemon=True).start()

    def _on_gemini_test_done(self, success: bool, msg: str, models: list[str]):
        if success:
            model_count = len(models)
            self._gemini_status.configure(
                text=f"{msg} ({model_count} models)", fg=C.ACCENT,
            )
            if models:
                combo = self._gemini_model_combo
                combo.configure(values=models)
                current = self._gemini_model_var.get()
                if current not in models:
                    self._gemini_model_var.set(models[0])
        else:
            self._gemini_status.configure(text=msg, fg=C.DANGER)

    # ── Save ─────────────────────────────────────────────────────────

    def _save(self):
        """Save config and reconfigure the agent."""
        config = {
            "groq_key": self._groq_key_var.get().strip(),
            "groq_model": self._groq_model_var.get().strip(),
            "gemini_key": self._gemini_key_var.get().strip(),
            "gemini_model": self._gemini_model_var.get().strip(),
            "preferred_provider": self._preferred_var.get().strip(),
        }
        save_agent_config(config)

        # Reconfigure agent
        self._agent.configure(
            groq_key=config["groq_key"],
            gemini_key=config["gemini_key"],
            groq_model=config["groq_model"],
            gemini_model=config["gemini_model"],
            preferred_provider=config["preferred_provider"],
        )

        self._status_label.configure(text="Saved!", fg=C.ACCENT)

        if self._on_save:
            self._on_save()

        self.after(800, self.destroy)
