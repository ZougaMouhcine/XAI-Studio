"""
XAI Studio — Model Persistence
================================
Save and load trained models as .pkl files with embedded metadata.
"""

import os
import json
from datetime import datetime

import joblib

from config.settings import MODELS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def save_model(
    model,
    metadata: dict,
    filepath: str | None = None,
    model_name: str = "model",
) -> str:
    """
    Serialize a trained model and its metadata to disk.

    The saved artifact is a dict:
        {"model": fitted_model, "metadata": {...}}

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn model.
    metadata : dict
        Arbitrary metadata (metrics, columns, task_type, etc.).
    filepath : str, optional
        Full path. If None, auto-generates inside MODELS_DIR.
    model_name : str
        Used for the auto-generated filename.

    Returns
    -------
    str
        Absolute path to the saved .pkl file.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace(" ", "_").lower()
        filename = f"{safe_name}_{timestamp}.pkl"
        filepath = os.path.join(MODELS_DIR, filename)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Enrich metadata
    metadata = {
        **metadata,
        "saved_at": datetime.now().isoformat(),
        "model_class": type(model).__name__,
    }

    artifact = {"model": model, "metadata": metadata}
    joblib.dump(artifact, filepath)
    logger.info("Model saved → %s (%.1f KB)", filepath, os.path.getsize(filepath) / 1024)
    return filepath


def load_model(filepath: str) -> tuple:
    """
    Load a model and its metadata from a .pkl file.

    Returns
    -------
    tuple[model, metadata_dict]
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    artifact = joblib.load(filepath)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(
            "Invalid model file format. Expected dict with 'model' key."
        )

    logger.info("Model loaded ← %s (%s)", filepath, artifact["metadata"].get("model_class", "unknown"))
    return artifact["model"], artifact.get("metadata", {})


def list_saved_models(directory: str | None = None) -> list[dict]:
    """
    List all saved .pkl models in a directory.

    Returns
    -------
    list[dict]
        Each dict: {filename, filepath, size_kb, saved_at, model_class, ...}
    """
    directory = directory or MODELS_DIR
    if not os.path.isdir(directory):
        return []

    models = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(directory, fname)
        try:
            _, metadata = load_model(fpath)
            models.append({
                "filename": fname,
                "filepath": fpath,
                "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                **metadata,
            })
        except Exception as exc:
            logger.warning("Could not read model '%s': %s", fname, exc)
            models.append({
                "filename": fname,
                "filepath": fpath,
                "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                "error": str(exc),
            })

    logger.info("Found %d saved model(s) in %s", len(models), directory)
    return models


def delete_model(filepath: str) -> bool:
    """
    Delete a saved model file.

    Returns
    -------
    bool
        True if successfully deleted.
    """
    if not os.path.isfile(filepath):
        logger.warning("Cannot delete — file not found: %s", filepath)
        return False

    os.remove(filepath)
    logger.info("Deleted model: %s", filepath)
    return True
