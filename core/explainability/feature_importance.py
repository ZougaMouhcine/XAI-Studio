"""
XAI Studio — Feature Importance
==================================
Extract and plot feature importance from trained models.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.logger import get_logger
from core.model_loader import TREE_BASED_CLASSES

logger = get_logger(__name__)


def get_feature_importance(model, X_test=None, y_test=None, feature_names=None):
    """
    Extract feature importances from a model.

    Strategy:
      1. Use .feature_importances_ for tree-based models.
      2. Use abs(coef_) for linear models.
      3. Fall back to sklearn permutation_importance.

    Parameters
    ----------
    model : fitted estimator
    X_test : np.ndarray, optional
        Required for permutation importance fallback.
    y_test : np.ndarray, optional
        Required for permutation importance fallback.
    feature_names : list[str], optional

    Returns
    -------
    dict with keys: importances (np.ndarray), feature_names (list)
    """
    class_name = type(model).__name__
    n_features = getattr(model, "n_features_in_", None)

    # Default feature names
    if feature_names is None and n_features is not None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    # --- Tree-based models ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        logger.info("Using built-in feature_importances_ (%s)", class_name)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        return {"importances": importances, "feature_names": feature_names}

    # --- Linear models (coefficients) ---
    if hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)
        logger.info("Using |coef_| as importance (%s)", class_name)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        return {"importances": importances, "feature_names": feature_names}

    # --- Permutation importance fallback ---
    if X_test is not None and y_test is not None:
        from sklearn.inspection import permutation_importance
        logger.info("Computing permutation importance (%s)", class_name)
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1,
        )
        importances = result.importances_mean
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        return {"importances": importances, "feature_names": feature_names}

    raise ValueError(
        "Impossible d'extraire l'importance des features. "
        "Fournissez X_test et y_test pour la méthode de permutation."
    )


def plot_feature_importance(importances, feature_names, top_k=20, title="Feature Importance"):
    """
    Create a ranked horizontal bar chart of feature importances.

    Returns matplotlib Figure.
    """
    from ui.components.plot_canvas import apply_plot_style, C
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C.BG_CARD)
    ax.set_facecolor("#111c2e")

    importances = np.array(importances)
    n = min(top_k, len(importances))
    sorted_idx = np.argsort(importances)[-n:]

    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_values = importances[sorted_idx]

    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n))

    ax.barh(sorted_names, sorted_values, color=colors)
    ax.set_xlabel("Importance")
    ax.set_title(title)

    fig.tight_layout()
    result_fig = fig
    plt.close("all")
    return result_fig
