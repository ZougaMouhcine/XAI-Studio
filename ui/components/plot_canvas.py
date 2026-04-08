"""
XAI Studio — Plot Canvas Component
=====================================
Reusable FigureCanvasTkAgg wrapper for embedding matplotlib plots
inside Tkinter with dark theme styling.
"""

import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ui.widgets import C, F, ModernButton
from utils.logger import get_logger

logger = get_logger(__name__)

# Dark theme matching the app palette
PLOT_STYLE = {
    "figure.facecolor": C.BG_CARD,
    "axes.facecolor": "#111c2e",
    "axes.edgecolor": C.BORDER_LIGHT,
    "axes.labelcolor": C.TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "text.color": C.TEXT,
    "xtick.color": C.TEXT_SEC,
    "ytick.color": C.TEXT_SEC,
    "grid.color": C.BORDER,
    "grid.alpha": 0.4,
    "legend.facecolor": C.BG_CARD,
    "legend.edgecolor": C.BORDER,
    "legend.fontsize": 9,
    "font.size": 10,
}


def apply_plot_style():
    """Apply the XAI Studio dark theme to matplotlib."""
    plt.rcParams.update(PLOT_STYLE)


def create_styled_figure(figsize=(8, 5), dpi=100) -> Figure:
    """Create a new Figure with the app's dark theme applied."""
    apply_plot_style()
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=C.BG_CARD)
    return fig


class PlotCanvas(tk.Frame):
    """
    Reusable matplotlib canvas for Tkinter.

    Usage
    -----
        canvas = PlotCanvas(parent)
        canvas.pack(fill="both", expand=True)

        fig = create_styled_figure()
        ax = fig.add_subplot(111)
        ax.plot(...)
        canvas.update_figure(fig)
    """

    def __init__(self, parent, bg=C.BG_MAIN, show_toolbar=False):
        super().__init__(parent, bg=bg)

        self._bg = bg
        self._show_toolbar = show_toolbar
        self._figure = None
        self._canvas_widget = None
        self._toolbar = None

        # Placeholder
        self._placeholder = tk.Label(
            self, text="📊  Le graphique apparaîtra ici",
            font=F.BODY, bg=bg, fg=C.TEXT_DIM,
        )
        self._placeholder.pack(fill="both", expand=True, pady=60)

    def update_figure(self, fig: Figure):
        """Replace the current plot with a new figure."""
        self._clear()
        self._figure = fig

        self._canvas_widget = FigureCanvasTkAgg(fig, master=self)
        widget = self._canvas_widget.get_tk_widget()
        widget.configure(bg=self._bg, highlightthickness=0)
        widget.pack(fill="both", expand=True)

        if self._show_toolbar:
            self._toolbar = NavigationToolbar2Tk(self._canvas_widget, self)
            self._toolbar.update()

        self._canvas_widget.draw()
        logger.info("Plot canvas updated")

    def export_png(self, filepath: str | None = None):
        """Save the current figure to a PNG file."""
        if self._figure is None:
            return

        if filepath is None:
            filepath = filedialog.asksaveasfilename(
                title="Exporter le graphique",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            )
        if not filepath:
            return

        self._figure.savefig(filepath, dpi=150, bbox_inches="tight",
                              facecolor=self._figure.get_facecolor())
        logger.info("Plot exported → %s", filepath)

    def _clear(self):
        """Remove current canvas and placeholder."""
        if self._placeholder:
            self._placeholder.pack_forget()
            self._placeholder = None

        if self._toolbar:
            self._toolbar.destroy()
            self._toolbar = None

        if self._canvas_widget:
            self._canvas_widget.get_tk_widget().destroy()
            self._canvas_widget = None

        if self._figure:
            plt.close(self._figure)
            self._figure = None

    def destroy(self):
        self._clear()
        super().destroy()
