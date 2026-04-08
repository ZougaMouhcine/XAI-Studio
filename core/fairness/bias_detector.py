"""
XAI Studio — Bias Detector
==============================
Compute fairness metrics to detect potential model bias.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds for bias alerts
DISPARATE_IMPACT_LOW = 0.8
DISPARATE_IMPACT_HIGH = 1.25
PARITY_THRESHOLD = 0.1
EQUALIZED_ODDS_THRESHOLD = 0.1


def compute_fairness_metrics(y_true, y_pred, sensitive_values, group_names=None):
    """
    Compute fairness metrics across groups defined by a sensitive attribute.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Model predictions.
    sensitive_values : np.ndarray
        Values of the sensitive attribute (same length as y_true).
    group_names : dict, optional
        Mapping from group value → readable name.

    Returns
    -------
    dict with keys:
        groups (list of dicts per group), summary (dict of overall metrics),
        alerts (list of alert strings), bias_detected (bool)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_values = np.asarray(sensitive_values)

    unique_groups = np.unique(sensitive_values)
    results = {"groups": [], "alerts": [], "bias_detected": False}

    group_metrics = {}

    for g in unique_groups:
        mask = sensitive_values == g
        yt = y_true[mask]
        yp = y_pred[mask]
        n = len(yt)

        name = str(group_names.get(g, g)) if group_names else str(g)

        # Accuracy
        accuracy = float(np.mean(yt == yp)) if n > 0 else 0.0

        # Positive prediction rate (Demographic Parity)
        positive_rate = float(np.mean(yp == 1)) if n > 0 else 0.0

        # TPR and FPR (for Equalized Odds)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        gm = {
            "group": name,
            "group_value": g,
            "count": n,
            "accuracy": round(accuracy, 4),
            "positive_rate": round(positive_rate, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        }
        results["groups"].append(gm)
        group_metrics[name] = gm

    # --- Summary metrics ---
    positive_rates = [g["positive_rate"] for g in results["groups"]]
    tprs = [g["tpr"] for g in results["groups"]]
    fprs = [g["fpr"] for g in results["groups"]]
    accuracies = [g["accuracy"] for g in results["groups"]]

    # Demographic Parity Difference
    dp_diff = max(positive_rates) - min(positive_rates) if positive_rates else 0
    # Equalized Odds
    eq_odds_tpr = max(tprs) - min(tprs) if tprs else 0
    eq_odds_fpr = max(fprs) - min(fprs) if fprs else 0
    # Disparate Impact Ratio
    max_pr = max(positive_rates) if positive_rates else 1
    min_pr = min(positive_rates) if positive_rates else 0
    di_ratio = (min_pr / max_pr) if max_pr > 0 else 0.0

    results["summary"] = {
        "demographic_parity_diff": round(dp_diff, 4),
        "equalized_odds_tpr_diff": round(eq_odds_tpr, 4),
        "equalized_odds_fpr_diff": round(eq_odds_fpr, 4),
        "disparate_impact_ratio": round(di_ratio, 4),
        "accuracy_spread": round(max(accuracies) - min(accuracies), 4) if accuracies else 0,
    }

    # --- Alerts ---
    if dp_diff > PARITY_THRESHOLD:
        results["alerts"].append(
            f"⚠ Démographic Parity: différence de {dp_diff:.2%} entre groupes (seuil: {PARITY_THRESHOLD:.0%})"
        )
    if di_ratio < DISPARATE_IMPACT_LOW:
        results["alerts"].append(
            f"🔴 Disparate Impact Ratio = {di_ratio:.3f} (< {DISPARATE_IMPACT_LOW}) — Biais significatif détecté"
        )
    elif di_ratio > DISPARATE_IMPACT_HIGH:
        results["alerts"].append(
            f"🔴 Disparate Impact Ratio = {di_ratio:.3f} (> {DISPARATE_IMPACT_HIGH}) — Biais inversé détecté"
        )
    if eq_odds_tpr > EQUALIZED_ODDS_THRESHOLD:
        results["alerts"].append(
            f"⚠ Equalized Odds (TPR): différence de {eq_odds_tpr:.2%} entre groupes"
        )
    if eq_odds_fpr > EQUALIZED_ODDS_THRESHOLD:
        results["alerts"].append(
            f"⚠ Equalized Odds (FPR): différence de {eq_odds_fpr:.2%} entre groupes"
        )

    results["bias_detected"] = len(results["alerts"]) > 0

    logger.info(
        "Fairness metrics computed — %d groups, %d alerts",
        len(results["groups"]), len(results["alerts"]),
    )
    return results


def plot_fairness_comparison(metrics_result, metric_key="accuracy", title=None):
    """
    Plot a bar chart comparing a metric across groups.

    Parameters
    ----------
    metrics_result : dict
        Output of compute_fairness_metrics.
    metric_key : str
        One of: accuracy, positive_rate, tpr, fpr
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from ui.components.plot_canvas import apply_plot_style, C
    apply_plot_style()

    groups = metrics_result["groups"]
    names = [g["group"] for g in groups]
    values = [g.get(metric_key, 0) for g in groups]

    metric_labels = {
        "accuracy": "Exactitude",
        "positive_rate": "Taux de prédiction positive",
        "tpr": "Taux de vrais positifs (TPR)",
        "fpr": "Taux de faux positifs (FPR)",
    }

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=C.BG_CARD)
    ax.set_facecolor("#111c2e")

    # Color by fairness level
    colors = []
    for v in values:
        mean_v = np.mean(values)
        diff = abs(v - mean_v)
        if diff < 0.05:
            colors.append("#10b981")  # green — fair
        elif diff < 0.1:
            colors.append("#f59e0b")  # orange — borderline
        else:
            colors.append("#ef4444")  # red — biased

    bars = ax.bar(names, values, color=colors, width=0.6)

    ax.set_ylabel(metric_labels.get(metric_key, metric_key))
    ax.set_title(title or f"Comparaison — {metric_labels.get(metric_key, metric_key)}")
    ax.set_ylim(0, 1.05)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", color=C.TEXT, fontsize=9,
        )

    fig.tight_layout()
    result_fig = fig
    plt.close("all")
    return result_fig
