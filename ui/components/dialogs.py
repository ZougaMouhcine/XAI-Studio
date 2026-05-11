"""
XAI Studio — Reusable Dialog Components (v2 — Modern)
=======================================================
Styled file pickers, message boxes, and a polished progress window.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from ui.widgets import C, F, ModernButton, scaled
from services.i18n import _


POPUP_WIDTH = 520
POPUP_HEIGHT = 560
POPUP_MESSAGE_HEIGHT = 220


def apply_popup_geometry(
    win: tk.Toplevel,
    parent: tk.Misc | None = None,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Apply the standard popup size and center it on the parent/screen."""
    w = scaled(width if width is not None else POPUP_WIDTH)
    h = scaled(height if height is not None else POPUP_HEIGHT)
    win.update_idletasks()

    if parent is not None:
        parent.update_idletasks()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        if pw <= 1 or ph <= 1:
            px = 0
            py = 0
            pw = parent.winfo_screenwidth()
            ph = parent.winfo_screenheight()
        x = px + max(0, (pw - w) // 2)
        y = py + max(0, (ph - h) // 2)
    else:
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)

    win.geometry(f"{w}x{h}+{x}+{y}")


def _resolve_parent() -> tk.Misc | None:
    try:
        return tk._get_default_root()
    except Exception:
        return None


class MessageDialog(tk.Toplevel):
    def __init__(self, parent, title: str, message: str, kind: str = "info", confirm: bool = False):
        super().__init__(parent)
        self.result = False
        self.title(title)
        self.configure(bg=C.BG_CARD)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        accent = {
            "info": C.ACCENT,
            "warning": C.WARNING,
            "error": C.DANGER,
            "confirm": C.ACCENT,
        }.get(kind, C.ACCENT)

        apply_popup_geometry(self, parent, height=POPUP_MESSAGE_HEIGHT)

        tk.Frame(self, bg=accent, height=3).pack(fill="x")

        inner = tk.Frame(self, bg=C.BG_CARD)
        inner.pack(fill="both", expand=True, padx=24, pady=18)

        tk.Label(inner, text=title, font=F.H4, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")
        wrap = scaled(POPUP_WIDTH - 96)
        tk.Label(
            inner,
            text=message,
            font=F.BODY,
            bg=C.BG_CARD,
            fg=C.TEXT_SEC,
            justify="left",
            wraplength=wrap,
        ).pack(anchor="w", pady=(10, 0))

        actions = tk.Frame(inner, bg=C.BG_CARD)
        actions.pack(fill="x", pady=(16, 0))

        if confirm:
            ModernButton(
                actions,
                text=_("prep_target_select_cancel"),
                style="ghost",
                command=self._on_cancel,
                bg=C.BG_CARD,
                width=100,
            ).pack(side="right")

        ModernButton(
            actions,
            text="OK",
            style="primary",
            command=self._on_ok,
            bg=C.BG_CARD,
            width=100,
        ).pack(side="right", padx=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._on_cancel if confirm else self._on_ok)

    def _on_ok(self):
        self.result = True
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        self.result = False
        self.grab_release()
        self.destroy()


def ask_open_csv() -> str | None:
    """Open a file dialog to select a CSV/Excel dataset."""
    path = filedialog.askopenfilename(
        title=_("dlg_open_dataset"),
        filetypes=[
            ("Tabular Files", "*.csv *.xlsx *.xls"),
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx *.xls"),
            ("All Files", "*.*"),
        ],
    )
    return path if path else None


def ask_save_pipeline_file() -> str | None:
    path = filedialog.asksaveasfilename(
        title=_("dlg_save_pipeline"),
        defaultextension=".joblib",
        filetypes=[("Joblib", "*.joblib"), ("All Files", "*.*")],
    )
    return path if path else None


def ask_open_pipeline_file() -> str | None:
    path = filedialog.askopenfilename(
        title=_("dlg_load_pipeline"),
        filetypes=[("Joblib", "*.joblib"), ("All Files", "*.*")],
    )
    return path if path else None


def ask_export_python_file() -> str | None:
    path = filedialog.asksaveasfilename(
        title=_("dlg_export_py"),
        defaultextension=".py",
        filetypes=[("Python", "*.py"), ("All Files", "*.*")],
    )
    return path if path else None


def show_info(title: str, message: str):
    parent = _resolve_parent()
    if parent is None:
        messagebox.showinfo(title, message)
        return
    dlg = MessageDialog(parent, title, message, kind="info")
    dlg.wait_window()


def show_error(title: str, message: str):
    parent = _resolve_parent()
    if parent is None:
        messagebox.showerror(title, message)
        return
    dlg = MessageDialog(parent, title, message, kind="error")
    dlg.wait_window()


def show_warning(title: str, message: str):
    parent = _resolve_parent()
    if parent is None:
        messagebox.showwarning(title, message)
        return
    dlg = MessageDialog(parent, title, message, kind="warning")
    dlg.wait_window()


def ask_confirm(title: str, message: str) -> bool:
    parent = _resolve_parent()
    if parent is None:
        return messagebox.askyesno(title, message)
    dlg = MessageDialog(parent, title, message, kind="confirm", confirm=True)
    dlg.wait_window()
    return bool(dlg.result)


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

        apply_popup_geometry(self, parent)

        self._total = total

        # Accent top bar
        tk.Frame(self, bg=C.ACCENT, height=3).pack(fill="x")

        inner = tk.Frame(self, bg=C.BG_CARD)
        inner.pack(fill="both", expand=True, padx=28, pady=20)

        # Title
        self._label = tk.Label(inner, text=_("dlg_init"),
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

        # ETA and Resource usage
        self._eta = tk.Label(inner, text="ETA: —", font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_MUTED)
        self._eta.pack(anchor="w", pady=(6, 0))

        self._resources = tk.Label(inner, text="CPU: —  MEM: —", font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_MUTED)
        self._resources.pack(anchor="w", pady=(2, 0))

        self.update_idletasks()

    def update_progress(self, current: int, detail: str = "", eta: str | None = None, resources: str | None = None):
        self._progress["value"] = current
        self._label.configure(text=detail)
        self._detail.configure(text=f"{current} / {self._total}")
        if eta is not None:
            self._eta.configure(text=f"ETA: {eta}")
        if resources is not None:
            self._resources.configure(text=resources)
        self.update_idletasks()

    def close(self):
        self.grab_release()
        self.destroy()
