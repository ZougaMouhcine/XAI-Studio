"""
XAI Studio — SHAP Explainer
===============================
Compute SHAP values and generate explanation plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.logger import get_logger
from core.model_loader import TREE_BASED_CLASSES, LINEAR_CLASSES

logger = get_logger(__name__)


def _get_explainer(model, X_background):
    """
    Auto-select the most efficient SHAP explainer for the model.

    Priority: TreeExplainer > LinearExplainer > KernelExplainer
    """
    import shap

    class_name = type(model).__name__

    if class_name in TREE_BASED_CLASSES:
        try:
            logger.info("Using SHAP TreeExplainer for %s", class_name)
            return shap.TreeExplainer(model)
        except Exception:
            logger.warning("TreeExplainer failed, falling back.")

    if class_name in LINEAR_CLASSES:
        try:
            logger.info("Using SHAP LinearExplainer for %s", class_name)
            return shap.LinearExplainer(model, X_background)
        except Exception:
            logger.warning("LinearExplainer failed, falling back.")

    # Fallback: KernelExplainer (slow but universal)
    logger.warning(
        "Using SHAP KernelExplainer (slow) for %s with %d features. "
        "This will be slow. Consider using a tree-based model for faster explanations.",
        class_name, X_background.shape[1]
    )
    
    # Subsample background more aggressively for performance (max 50)
    if len(X_background) > 50:
        bg_idx = np.random.choice(len(X_background), 50, replace=False)
        bg = X_background[bg_idx]
    else:
        bg = X_background

    predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    return shap.KernelExplainer(predict_fn, bg)


def compute_shap_values(model, X_data, feature_names=None):
    """
    Compute SHAP values for the given model and data.

    Parameters
    ----------
    model : fitted estimator
    X_data : np.ndarray
        Data to explain (typically X_test or a subset).
    feature_names : list[str], optional

    Returns
    -------
    shap.Explanation or np.ndarray
    """
    import shap

    explainer = _get_explainer(model, X_data)
    shap_values = explainer(X_data)

    logger.info("SHAP values computed — shape: %s", str(np.shape(shap_values.values)))
    return shap_values


def plot_shap_summary(shap_values, X_data=None, feature_names=None, plot_type="bar"):
    """
    Generate a SHAP summary plot (bar or beeswarm).

    Returns matplotlib Figure.
    """
    import shap
    from ui.components.plot_canvas import apply_plot_style, C
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C.BG_CARD)
    ax.set_facecolor("#111c2e")

    plt.sca(ax)

    vals = shap_values.values
    # Handle multiclass (take mean absolute across classes)
    if vals.ndim == 3:
        vals_2d = np.abs(vals).mean(axis=2)
    else:
        vals_2d = vals

    if plot_type == "bar":
        mean_abs = np.abs(vals_2d).mean(axis=0)
        names = feature_names or [f"Feature {i}" for i in range(len(mean_abs))]
        # Sort
        sorted_idx = np.argsort(mean_abs)[-20:]  # top 20
        ax.barh(
            [names[i] for i in sorted_idx],
            mean_abs[sorted_idx],
            color="#3b82f6",
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance (Bar)")
    else:
        # Beeswarm — use SHAP's built-in with our figure
        shap.summary_plot(
            vals_2d, features=X_data, feature_names=feature_names,
            plot_type="dot", show=False, max_display=20,
        )
        fig = plt.gcf()
        fig.set_facecolor(C.BG_CARD)
        for a in fig.axes:
            a.set_facecolor("#111c2e")

    fig.tight_layout()
    result_fig = fig
    plt.close("all")
    return result_fig


def plot_shap_waterfall(shap_values, instance_idx=0, feature_names=None):
    """
    Generate a SHAP waterfall plot for a specific instance.

    Returns matplotlib Figure.
    """
    import shap
    from ui.components.plot_canvas import apply_plot_style, C
    apply_plot_style()

    sv = shap_values[instance_idx]
    # Handle multiclass: select class with highest mean absolute SHAP value
    if sv.values.ndim > 1:
        # Find the class with highest mean absolute contribution
        if sv.values.shape[1] > 1:
            class_idx = np.argmax(np.abs(sv.values).mean(axis=0))
        else:
            class_idx = 0
        
        sv = shap.Explanation(
            values=sv.values[:, class_idx],
            base_values=(
                sv.base_values[class_idx]
                if isinstance(sv.base_values, (list, np.ndarray)) and len(sv.base_values) > class_idx
                else sv.base_values
            ),
            data=sv.data,
            feature_names=feature_names,
        )

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C.BG_CARD)
    plt.sca(ax)
    shap.plots.waterfall(sv, show=False, max_display=15)
    fig = plt.gcf()
    fig.set_facecolor(C.BG_CARD)
    for a in fig.axes:
        a.set_facecolor("#111c2e")
    fig.tight_layout()
    result_fig = fig
    plt.close("all")
    return result_fig


def plot_shap_force(shap_values, instance_idx=0, feature_names=None):
    """
    Generate a SHAP force-style bar plot for a specific instance.

    Returns matplotlib Figure (uses bar chart as force plots are HTML-based).
    """
    from ui.components.plot_canvas import apply_plot_style, C
    apply_plot_style()

    sv = shap_values[instance_idx]
    vals = sv.values
    if vals.ndim > 1:
        vals = vals[:, 1] if vals.shape[1] > 1 else vals[:, 0]

    names = feature_names or [f"F{i}" for i in range(len(vals))]

    # Top features by absolute contribution
    top_k = min(15, len(vals))
    top_idx = np.argsort(np.abs(vals))[-top_k:]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C.BG_CARD)
    ax.set_facecolor("#111c2e")

    colors = ["#ef4444" if v < 0 else "#10b981" for v in vals[top_idx]]
    ax.barh(
        [names[i] for i in top_idx],
        vals[top_idx],
        color=colors,
    )
    ax.axvline(x=0, color=C.TEXT_DIM, linestyle="--", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(f"SHAP Force Plot — Instance {instance_idx}")
    fig.tight_layout()

    result_fig = fig
    return result_fig
