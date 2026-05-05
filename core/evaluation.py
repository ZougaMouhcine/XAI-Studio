"""
XAI Studio — Model Evaluation
===============================
Compute performance metrics for classification and regression models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    silhouette_score,
    davies_bouldin_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True labels / values.
    task_type : str
        'classification' or 'regression'.
    model_name : str
        Name for logging purposes.

    Returns
    -------
    dict
        Dictionary of metric names → values.
    """
    if model is None:
        logger.warning("Skipping evaluation for '%s' — model is None", model_name)
        return {"error": "Model training failed"}

    y_pred = model.predict(X_test)

    if task_type == "classification":
        metrics = _evaluate_classification(y_test, y_pred)
        # ROC AUC if possible
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
                if y_score.ndim == 2 and y_score.shape[1] > 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
                else:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_score[:, 1])
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
                metrics["roc_auc"] = roc_auc_score(y_test, y_score)
        except Exception:
            metrics["roc_auc"] = None
    elif task_type == "regression":
        metrics = _evaluate_regression(y_test, y_pred)
    elif task_type == "clustering":
        metrics = _evaluate_clustering(model, X_test)
    else:
        raise ValueError(f"Unknown task type: '{task_type}'")

    logger.info("Evaluated '%s' — %s", model_name, 
                ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                          for k, v in metrics.items() 
                          if k not in ("confusion_matrix", "classification_report")))
    return metrics


def _evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    average = "weighted" if len(np.unique(y_true)) > 2 else "binary"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    return metrics


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def _evaluate_clustering(model, X: np.ndarray) -> dict:
    """Compute clustering metrics using predicted labels."""
    if X is None:
        return {"error": "No data for clustering evaluation"}

    try:
        if hasattr(model, "predict"):
            labels = model.predict(X)
        else:
            labels = model.fit_predict(X)
    except Exception as exc:
        return {"error": str(exc)}

    metrics = {}
    try:
        metrics["silhouette"] = silhouette_score(X, labels)
    except Exception:
        metrics["silhouette"] = None
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
    except Exception:
        metrics["davies_bouldin"] = None
    return metrics


def compare_models(
    evaluation_results: dict[str, dict],
    task_type: str,
) -> pd.DataFrame:
    """
    Create a comparison table across all evaluated models.

    Parameters
    ----------
    evaluation_results : dict
        model_name → metrics dict (output of evaluate_model).
    task_type : str
        'classification' or 'regression'.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by the primary metric (descending).
    """
    if task_type == "classification":
        keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        sort_by = "accuracy"
    elif task_type == "regression":
        keys = ["mae", "mse", "rmse", "r2", "mape"]
        sort_by = "r2"
    else:
        keys = ["silhouette", "davies_bouldin"]
        sort_by = "silhouette"

    rows = []
    for name, metrics in evaluation_results.items():
        if "error" in metrics:
            row = {"Model": name, **{k: None for k in keys}, "Status": "Failed"}
        else:
            row = {"Model": name}
            for k in keys:
                val = metrics.get(k)
                row[k] = round(val, 4) if isinstance(val, float) else val
            row["Status"] = "OK"
        rows.append(row)

    df = pd.DataFrame(rows)
    ascending = sort_by in ("mae", "mse", "rmse", "davies_bouldin")
    df = df.sort_values(sort_by, ascending=ascending, na_position="last").reset_index(drop=True)

    logger.info("Model comparison table generated (%d models)", len(df))
    return df
