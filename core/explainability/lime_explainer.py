"""
XAI Studio — LIME Explainer
===============================
Explain individual predictions with LIME tabular explanations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.logger import get_logger

logger = get_logger(__name__)


def create_lime_explainer(X_train, feature_names=None, mode="classification"):
    """
    Create a LimeTabularExplainer.

    Parameters
    ----------
    X_train : np.ndarray
        Training data for computing statistics.
    feature_names : list[str], optional
    mode : str
        "classification" or "regression"

    Returns
    -------
    lime.lime_tabular.LimeTabularExplainer
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode=mode,
        random_state=42,
    )
    logger.info("LIME explainer created (mode=%s, features=%d)",
                mode, X_train.shape[1])
    return explainer


def explain_instance(explainer, model, instance, num_features=10):
    """
    Explain a single prediction with LIME.

    Parameters
    ----------
    explainer : LimeTabularExplainer
    model : fitted estimator
    instance : np.ndarray (1D)
        Single row of feature values.
    num_features : int

    Returns
    -------
    lime.explanation.Explanation
    """
    # Use predict_proba for classifiers, predict for regressors
    # with fallback handling
    predict_fn = None
    if hasattr(model, "predict_proba"):
        try:
            # Test that predict_proba actually works
            test_pred = model.predict_proba(instance.reshape(1, -1))
            predict_fn = model.predict_proba
        except (ValueError, AttributeError, TypeError):
            logger.warning("predict_proba exists but failed on test sample, using predict instead")
    
    if predict_fn is None:
        predict_fn = model.predict

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=num_features,
    )
    logger.info("LIME explanation generated for instance")
    return explanation


def plot_lime_explanation(explanation, title="LIME Explanation"):
    """
    Render a LIME explanation as a horizontal bar chart.

    Returns matplotlib Figure.
    """
    from ui.components.plot_canvas import apply_plot_style, C
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C.BG_CARD)
    ax.set_facecolor("#111c2e")

    exp_list = explanation.as_list()
    if not exp_list:
        ax.text(0.5, 0.5, "Pas de données d'explication",
                ha="center", va="center", color=C.TEXT_MUTED,
                transform=ax.transAxes, fontsize=12)
        return fig

    # Sort by absolute weight
    exp_list.sort(key=lambda x: abs(x[1]))

    labels = [e[0] for e in exp_list]
    values = [e[1] for e in exp_list]
    colors = ["#ef4444" if v < 0 else "#10b981" for v in values]

    ax.barh(labels, values, color=colors)
    ax.axvline(x=0, color=C.TEXT_DIM, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Contribution à la prédiction")
    ax.set_title(title)

    fig.tight_layout()
    result_fig = fig
    plt.close("all")
    return result_fig
