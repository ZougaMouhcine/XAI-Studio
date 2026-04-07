"""
XAI Studio — Status Bar (v2 — Modern)
=======================================
Sleek bottom bar with animated status dot, latest message, and version tag.
"""

import tkinter as tk
from ui.widgets import C, F


class StatusBar(tk.Frame):
    """Polished status bar pinned to the bottom of the window."""

    LEVEL_COLORS = {
        "INFO":     C.ACCENT,
        "WARNING":  C.WARNING,
        "ERROR":    C.DANGER,
        "DEBUG":    C.TEXT_DIM,
        "CRITICAL": C.DANGER,
    }

    def __init__(self, parent):
        super().__init__(parent, bg="#060c18", height=34)
        self.pack_propagate(False)

        # Left section: status
        left = tk.Frame(self, bg="#060c18")
        left.pack(side="left", fill="y", padx=(16, 0))

        self._dot = tk.Label(left, text="●", font=(F.FAM, 7),
                              bg="#060c18", fg=C.SUCCESS)
        self._dot.pack(side="left", padx=(0, 8), pady=8)

        self._msg = tk.Label(left, text="Ready", font=(F.FAM, 9),
                              bg="#060c18", fg=C.TEXT_MUTED, anchor="w")
        self._msg.pack(side="left", fill="x", expand=True, pady=8)

        # Right section: branding
        right = tk.Frame(self, bg="#060c18")
        right.pack(side="right", fill="y", padx=(0, 16))

        tk.Label(right, text="XAI Studio v1.0.0", font=(F.FAM, 8),
                 bg="#060c18", fg=C.TEXT_DIM).pack(side="right", pady=8)

        # Separator dot
        tk.Label(right, text="·", font=(F.FAM, 8),
                 bg="#060c18", fg=C.TEXT_DIM).pack(side="right", padx=8, pady=8)

        tk.Label(right, text="Phase 1", font=(F.FAM, 8),
                 bg="#060c18", fg=C.TEXT_DIM).pack(side="right", pady=8)

    def set_message(self, level: str, message: str):
        color = self.LEVEL_COLORS.get(level, C.TEXT_MUTED)
        short = message.split("|")[-1].strip() if "|" in message else message
        self._dot.configure(fg=color)
        self._msg.configure(text=short, fg=C.TEXT_SEC)

    def set_ready(self):
        self._dot.configure(fg=C.SUCCESS)
        self._msg.configure(text="Ready", fg=C.TEXT_MUTED)
