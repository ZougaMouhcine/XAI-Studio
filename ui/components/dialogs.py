"""
XAI Studio — Reusable Dialog Components (v2 — Modern)
=======================================================
Styled file pickers, message boxes, and a polished progress window.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ui.widgets import C, F


def ask_open_csv() -> str | None:
    """Open a file dialog to select a CSV file."""
    path = filedialog.askopenfilename(
        title="Ouvrir un fichier CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    return path if path else None


def show_info(title: str, message: str):
    messagebox.showinfo(title, message)


def show_error(title: str, message: str):
    messagebox.showerror(title, message)


def show_warning(title: str, message: str):
    messagebox.showwarning(title, message)


def ask_confirm(title: str, message: str) -> bool:
    return messagebox.askyesno(title, message)


class ProgressDialog(tk.Toplevel):
    """
    Modal progress window with modern styling.

    Usage
    -----
    dlg = ProgressDialog(parent, "Training…", total=6)
    dlg.update_progress(1, "Logistic Regression")
    dlg.close()
    """

    def __init__(self, parent, title: str, total: int = 100):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=C.BG_CARD)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.overrideredirect(False)

        w, h = 460, 180
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        self._total = total

        # Accent top bar
        tk.Frame(self, bg=C.ACCENT, height=3).pack(fill="x")

        inner = tk.Frame(self, bg=C.BG_CARD)
        inner.pack(fill="both", expand=True, padx=28, pady=20)

        # Title
        self._label = tk.Label(inner, text="Initialisation…",
                                font=F.H4, bg=C.BG_CARD, fg=C.TEXT)
        self._label.pack(anchor="w", pady=(0, 12))

        # Progress bar
        self._progress = ttk.Progressbar(
            inner, orient="horizontal", length=400, mode="determinate",
            maximum=total, style="Green.Horizontal.TProgressbar",
        )
        self._progress.pack(fill="x", pady=(0, 8))

        # Detail
        self._detail = tk.Label(inner, text=f"0 / {total}",
                                 font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_MUTED)
        self._detail.pack(anchor="w")

        self.update_idletasks()

    def update_progress(self, current: int, detail: str = ""):
        self._progress["value"] = current
        self._label.configure(text=detail)
        self._detail.configure(text=f"{current} / {self._total}")
        self.update_idletasks()

    def close(self):
        self.grab_release()
        self.destroy()
