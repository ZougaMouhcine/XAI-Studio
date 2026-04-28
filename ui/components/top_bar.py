"""
XAI Studio — Top Header Bar
=============================
Docker Desktop-inspired top application bar.
"""

import tkinter as tk
from ui.widgets import C, F


class TopBar(tk.Frame):
    """Top app header with brand only."""

    def __init__(self, parent):
        super().__init__(parent, bg=C.HEADER_BG, height=64)
        self.pack_propagate(False)
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
