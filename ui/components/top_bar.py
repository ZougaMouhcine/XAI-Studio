"""
XAI Studio — Top Header Bar
=============================
Docker Desktop-inspired top application bar with AI copilot toggle.
"""

import tkinter as tk
from ui.widgets import C, F


class TopBar(tk.Frame):
    """Top app header with brand and agent toggle."""

    def __init__(self, parent, on_toggle_agent=None):
        super().__init__(parent, bg=C.HEADER_BG, height=64)
        self.pack_propagate(False)
        self._on_toggle_agent = on_toggle_agent
        self._agent_btn = None
        self._agent_active = False
        self._build()

    def _build(self):
        root = tk.Frame(self, bg=C.HEADER_BG)
        root.pack(fill="both", expand=True, padx=16)

        # Left brand area
        left = tk.Frame(root, bg=C.HEADER_BG, width=248)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        brand = tk.Frame(left, bg=C.HEADER_BG)
        brand.pack(side="left", padx=(4, 0), pady=16)

        tk.Label(
            brand,
            text="XAI.STUDIO",
            bg=C.HEADER_BG,
            fg=C.HEADER_TEXT,
            font=(F.FAM, 15, "bold"),
        ).pack(side="left")

        # Spacer keeps brand aligned to the left within the header.
        tk.Frame(root, bg=C.HEADER_BG).pack(side="left", fill="x", expand=True)

        # ── Right side: Agent toggle button ──────────────────────────
        right = tk.Frame(root, bg=C.HEADER_BG)
        right.pack(side="right", fill="y")

        self._agent_btn = tk.Label(
            right,
            text="✦ Copilot",
            font=(F.FAM, 10, "bold"),
            bg=C.HEADER_SURFACE,
            fg=C.HEADER_TEXT,
            padx=12,
            pady=6,
            cursor="hand2",
        )
        self._agent_btn.pack(side="right", padx=(8, 0), pady=16)
        self._agent_btn.bind("<Button-1>", lambda e: self._toggle_agent())
        self._agent_btn.bind("<Enter>", lambda e: self._agent_btn.configure(bg=C.HEADER_SURFACE_HOVER))
        self._agent_btn.bind("<Leave>", lambda e: self._update_agent_btn_bg())

        # Keyboard shortcut hint
        shortcut = tk.Label(
            right,
            text="Ctrl+Shift+I",
            font=(F.FAM, 8),
            bg=C.HEADER_BG,
            fg=C.HEADER_TEXT_MUTED,
        )
        shortcut.pack(side="right", pady=16)

    def _toggle_agent(self):
        if self._on_toggle_agent:
            self._on_toggle_agent()

    def set_agent_active(self, active: bool):
        """Update the button visual state."""
        self._agent_active = active
        self._update_agent_btn_bg()

    def _update_agent_btn_bg(self):
        if self._agent_btn:
            if self._agent_active:
                self._agent_btn.configure(bg=C.HEADER_SURFACE_HOVER, fg="#ffffff")
            else:
                self._agent_btn.configure(bg=C.HEADER_SURFACE, fg=C.HEADER_TEXT)
