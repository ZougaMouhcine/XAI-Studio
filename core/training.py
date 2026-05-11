"""
XAI Studio — Model Training
=============================
Dynamic model instantiation and training for classification & regression tasks.
"""

import importlib
import time
from typing import Any

import numpy as np

from config.settings import CLASSIFICATION_MODELS, REGRESSION_MODELS, CLUSTERING_MODELS
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from utils.logger import get_logger

logger = get_logger(__name__)


def _is_model_available(model_info: dict) -> bool:
    try:
        module = importlib.import_module(model_info["module"])
        _ = getattr(module, model_info["class"])
        return True
    except Exception:
        return False


def get_available_models(task_type: str, include_unavailable: bool = False) -> dict:
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
        registry = CLASSIFICATION_MODELS
    elif task_type == "regression":
        registry = REGRESSION_MODELS
    elif task_type == "clustering":
        registry = CLUSTERING_MODELS
    else:
        raise ValueError(
            f"Unknown task type: '{task_type}'. Expected 'classification', 'regression', or 'clustering'."
        )

    if include_unavailable:
        return registry

    return {name: info for name, info in registry.items() if _is_model_available(info)}


def _instantiate_model(model_info: dict, custom_params: dict | None = None):
    """Dynamically import and instantiate a scikit-learn model."""
    module = importlib.import_module(model_info["module"])
    cls = getattr(module, model_info["class"])

    params = {}
    if custom_params:
        params.update(custom_params)

    return cls(**params)


def _infer_param_type(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (list, tuple, set)):
        return "choice"
    return "string"


def _wrap_multi_output(model, task_type: str):
    if task_type == "classification":
        if isinstance(model, MultiOutputClassifier):
            return model
        return MultiOutputClassifier(model)
    if task_type == "regression":
        if isinstance(model, MultiOutputRegressor):
            return model
        return MultiOutputRegressor(model)
    return model


def get_model_param_schema(
    task_type: str,
    model_name: str,
    include_all: bool = False,
) -> list[dict]:
    """Return parameter metadata for a model, including inferred defaults."""
    registry = get_available_models(task_type, include_unavailable=True)
    info = registry.get(model_name)
    if info is None:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    param_meta = info.get("param_meta", {}) or {}
    default_params = info.get("default_params", {}) or {}

    defaults = {}
    try:
        module = importlib.import_module(info["module"])
        cls = getattr(module, info["class"])
        defaults = cls().get_params()
    except Exception:
        defaults = {}

    schema = []
    used = set()
    for name, meta in param_meta.items():
        default_val = meta.get("default", default_params.get(name, defaults.get(name)))
        entry = {
            "name": name,
            "type": meta.get("type") or _infer_param_type(default_val),
            "default": default_val,
            "min": meta.get("min"),
            "max": meta.get("max"),
            "choices": meta.get("choices"),
            "desc": meta.get("desc", ""),
            "source": "meta",
        }
        schema.append(entry)
        used.add(name)

    if include_all:
        for name, val in defaults.items():
            if name in used:
                continue
            schema.append({
                "name": name,
                "type": _infer_param_type(val),
                "default": val,
                "min": None,
                "max": None,
                "choices": None,
                "desc": "",
                "source": "auto",
            })

    # Ensure deterministic ordering
    return sorted(schema, key=lambda e: (e.get("source") != "meta", e.get("name", "")))


def train_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    custom_params: dict | None = None,
    multi_target: bool = False,
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
    if multi_target and task_type != "clustering":
        if y_train is not None and hasattr(y_train, "ndim"):
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                model = _wrap_multi_output(model, task_type)

    logger.info("Training '%s' (%s)…", name, model_info["class"])
    start = time.perf_counter()
    if task_type == "clustering":
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info("'%s' trained in %.3f s", name, elapsed)

    return model, elapsed


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    selected_models: list[str] | None = None,
    model_params_map: dict[str, dict] | None = None,
    multi_target: bool = False,
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
    model_params_map = model_params_map or {}

    results = {}
    total = len(names)
    for idx, name in enumerate(names, 1):
        elapsed = None
        try:
            custom_params = model_params_map.get(name)
            model, elapsed = train_model(
                name,
                X_train,
                y_train,
                task_type,
                custom_params=custom_params,
                multi_target=multi_target,
            )
            results[name] = {"model": model, "training_time": elapsed}
        except Exception as exc:
            logger.error("Failed to train '%s': %s", name, exc)
            results[name] = {"model": None, "training_time": 0, "error": str(exc)}

        if progress_callback:
            progress_callback(idx, total, name, elapsed)

    logger.info("Training complete — %d/%d models succeeded", 
                sum(1 for v in results.values() if v["model"] is not None), total)
    return results
