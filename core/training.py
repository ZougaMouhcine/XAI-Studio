"""
XAI Studio — Model Training
=============================
Dynamic model instantiation and training for classification & regression tasks.
"""

import importlib
import time
from typing import Any

import numpy as np

from config.settings import CLASSIFICATION_MODELS, REGRESSION_MODELS
from utils.logger import get_logger

logger = get_logger(__name__)


def get_available_models(task_type: str) -> dict:
    """
    Return the registry of supported models for the given task type.

    Parameters
    ----------
    task_type : str
        Either 'classification' or 'regression'.

    Returns
    -------
    dict
        Model name → {module, class, default_params}.
    """
    if task_type == "classification":
        return CLASSIFICATION_MODELS
    elif task_type == "regression":
        return REGRESSION_MODELS
    else:
        raise ValueError(f"Unknown task type: '{task_type}'. Expected 'classification' or 'regression'.")


def _instantiate_model(model_info: dict, custom_params: dict | None = None):
    """Dynamically import and instantiate a scikit-learn model."""
    module = importlib.import_module(model_info["module"])
    cls = getattr(module, model_info["class"])

    params = {**model_info["default_params"]}
    if custom_params:
        params.update(custom_params)

    return cls(**params)


def train_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    custom_params: dict | None = None,
) -> tuple[Any, float]:
    """
    Train a single model by name.

    Parameters
    ----------
    name : str
        Model name (must exist in the registry).
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    task_type : str
        'classification' or 'regression'.
    custom_params : dict, optional
        Override default hyperparameters.

    Returns
    -------
    tuple[model, training_time_seconds]
    """
    registry = get_available_models(task_type)
    if name not in registry:
        raise ValueError(
            f"Model '{name}' not found in {task_type} registry. "
            f"Available: {list(registry.keys())}"
        )

    model_info = registry[name]
    model = _instantiate_model(model_info, custom_params)

    logger.info("Training '%s' (%s)…", name, model_info["class"])
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info("'%s' trained in %.3f s", name, elapsed)

    return model, elapsed


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    selected_models: list[str] | None = None,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Train multiple models sequentially.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    task_type : str
        'classification' or 'regression'.
    selected_models : list[str], optional
        Subset of model names to train. If None, trains all.
    progress_callback : callable, optional
        Called with (current_index, total, model_name) after each model.

    Returns
    -------
    dict
        model_name → {"model": fitted_model, "training_time": float}
    """
    registry = get_available_models(task_type)
    names = selected_models if selected_models else list(registry.keys())

    results = {}
    total = len(names)
    for idx, name in enumerate(names, 1):
        try:
            model, elapsed = train_model(name, X_train, y_train, task_type)
            results[name] = {"model": model, "training_time": elapsed}
        except Exception as exc:
            logger.error("Failed to train '%s': %s", name, exc)
            results[name] = {"model": None, "training_time": 0, "error": str(exc)}

        if progress_callback:
            progress_callback(idx, total, name)

    logger.info("Training complete — %d/%d models succeeded", 
                sum(1 for v in results.values() if v["model"] is not None), total)
    return results
