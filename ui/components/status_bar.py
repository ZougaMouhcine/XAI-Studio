"""
XAI Studio — Status Bar (v2 — Modern)
=======================================
Sleek bottom bar with animated status dot, latest message, and version tag.
"""

import tkinter as tk
from ui.widgets import C, F


class StatusBar(tk.Frame):
    """Polished status bar pinned to the bottom of the window."""

    def __init__(self, parent):
        super().__init__(parent, bg=C.BG_STATUS, height=34)
        self.pack_propagate(False)

        top_border = tk.Frame(self, bg=C.BORDER, height=1)
        top_border.pack(side="top", fill="x")

        # Right section: branding
        right = tk.Frame(self, bg=C.BG_STATUS)
        right.pack(side="right", fill="y", padx=(0, 16))

        tk.Label(right, text="XAI Studio v1.0.0", font=(F.FAM, 8),
                 bg=C.BG_STATUS, fg=C.TEXT_DIM).pack(side="right", pady=8)

        # Separator dot
        tk.Label(right, text="·", font=(F.FAM, 8),
                 bg=C.BG_STATUS, fg=C.TEXT_DIM).pack(side="right", padx=8, pady=8)

        tk.Label(right, text="Phase 1", font=(F.FAM, 8),
                 bg=C.BG_STATUS, fg=C.TEXT_DIM).pack(side="right", pady=8)

    def set_message(self, level: str, message: str):
        _ = level
        _ = message

    def set_ready(self):
        return
