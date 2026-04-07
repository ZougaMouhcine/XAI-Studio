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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    elif task_type == "regression":
        metrics = _evaluate_regression(y_test, y_pred)
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

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


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
        keys = ["accuracy", "precision", "recall", "f1_score"]
        sort_by = "accuracy"
    else:
        keys = ["mae", "mse", "rmse", "r2"]
        sort_by = "r2"

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
    ascending = sort_by in ("mae", "mse", "rmse")
    df = df.sort_values(sort_by, ascending=ascending, na_position="last").reset_index(drop=True)

    logger.info("Model comparison table generated (%d models)", len(df))
    return df
