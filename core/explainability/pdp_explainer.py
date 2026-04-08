"""
XAI Studio — PDP Explainer
==============================
Partial Dependence Plots (1D and 2D) for model interpretation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_pdp_1d(model, X, feature_idx, feature_name=None, grid_resolution=50):
    """
    Compute and plot a 1D Partial Dependence Plot.

    Parameters
    ----------
    model : fitted estimator
    X : np.ndarray
        Input features (typically X_test).
    feature_idx : int
        Index of the feature to plot.
    feature_name : str, optional
    grid_resolution : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.inspection import partial_dependence
    from ui.components.plot_canvas import apply_plot_style, C
    apply_plot_style()

    fname = feature_name or f"Feature {feature_idx}"

    result = partial_dependence(
        model, X, features=[feature_idx],
        grid_resolution=grid_resolution,
        kind="average",
    )

    pd_values = result["average"][0]
    grid_values = result["grid_values"][0]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=C.BG_CARD)
    ax.set_facecolor("#111c2e")

    ax.plot(grid_values, pd_values, color="#3b82f6", linewidth=2.5)
    ax.fill_between(grid_values, pd_values, alpha=0.15, color="#3b82f6")

    ax.set_xlabel(fname)
    ax.set_ylabel("Partial Dependence")
    ax.set_title(f"PDP — {fname}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    logger.info("1D PDP computed for feature '%s'", fname)

    result_fig = fig
    plt.close("all")
    return result_fig


def compute_pdp_2d(model, X, feature_idxs, feature_names=None, grid_resolution=30):
    """
    Compute and plot a 2D Partial Dependence heatmap.

    Parameters
    ----------
    model : fitted estimator
    X : np.ndarray
    feature_idxs : tuple[int, int]
        Pair of feature indices.
    feature_names : list[str], optional
    grid_resolution : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.inspection import partial_dependence
    from ui.components.plot_canvas import apply_plot_style, C
    apply_plot_style()

    f1, f2 = feature_idxs
    name1 = feature_names[0] if feature_names else f"Feature {f1}"
    name2 = feature_names[1] if feature_names else f"Feature {f2}"

    result = partial_dependence(
        model, X, features=[feature_idxs],
        grid_resolution=grid_resolution,
        kind="average",
    )

    pd_values = result["average"][0]
    grid_1 = result["grid_values"][0]
    grid_2 = result["grid_values"][1]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=C.BG_CARD)
    ax.set_facecolor("#111c2e")

    XX, YY = np.meshgrid(grid_1, grid_2)
    im = ax.contourf(XX, YY, pd_values.T, levels=25, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Partial Dependence", color=C.TEXT)
    cbar.ax.yaxis.set_tick_params(color=C.TEXT_SEC)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=C.TEXT_SEC)

    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    ax.set_title(f"PDP 2D — {name1} × {name2}")

    fig.tight_layout()
    logger.info("2D PDP computed for features '%s' × '%s'", name1, name2)

    result_fig = fig
    plt.close("all")
    return result_fig
