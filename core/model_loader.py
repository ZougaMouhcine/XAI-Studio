"""
XAI Studio — Model Loader
============================
Load externally saved model files (.pkl / .joblib) and auto-detect
their type, algorithm, and metadata.
"""

import os
import joblib

from utils.logger import get_logger

logger = get_logger(__name__)

# Model families for classification / detection heuristics
_CLASSIFIER_KEYWORDS = {
    "Classifier", "SVC", "NuSVC", "LinearSVC",
    "LogisticRegression", "SGDClassifier", "Perceptron",
    "BernoulliNB", "GaussianNB", "MultinomialNB", "ComplementNB",
}
_REGRESSOR_KEYWORDS = {
    "Regressor", "SVR", "NuSVR", "LinearSVR",
    "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    "SGDRegressor",
}

TREE_BASED_CLASSES = {
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "AdaBoostClassifier", "AdaBoostRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
}

LINEAR_CLASSES = {
    "LogisticRegression", "LinearRegression",
    "Ridge", "Lasso", "ElasticNet",
    "SGDClassifier", "SGDRegressor",
    "LinearSVC", "LinearSVR",
}


def load_model_file(filepath: str):
    """
    Load a model from a .pkl or .joblib file.

    Supports two formats:
      1. Phase 1 dict: {"model": estimator, "metadata": {...}}
      2. Raw sklearn estimator

    Returns
    -------
    tuple[model, metadata_dict]
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in (".pkl", ".joblib"):
        raise ValueError(
            f"Format non supporté : '{ext}'. "
            "Seuls les fichiers .pkl et .joblib sont acceptés."
        )

    try:
        artifact = joblib.load(filepath)
    except Exception as exc:
        raise ValueError(f"Impossible de charger le fichier : {exc}") from exc

    # Phase 1 format: {"model": ..., "metadata": {...}}
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        metadata = artifact.get("metadata", {})
        logger.info("Loaded Phase 1 model artifact from %s", filepath)
        return model, metadata

    # Raw estimator
    if hasattr(artifact, "predict"):
        logger.info("Loaded raw estimator from %s", filepath)
        return artifact, {}

    raise ValueError(
        "Le fichier ne contient ni un estimateur scikit-learn "
        "ni un artefact XAI Studio valide."
    )


def detect_model_info(model, existing_metadata: dict | None = None) -> dict:
    """
    Inspect a loaded model and extract descriptive metadata.

    Returns
    -------
    dict with keys:
        algorithm, model_class, task_type, n_features,
        feature_names, params, is_tree_based, is_linear
    """
    meta = dict(existing_metadata or {})
    class_name = type(model).__name__

    meta["model_class"] = class_name
    meta["algorithm"] = _pretty_algorithm_name(class_name)

    # --- Task type detection ------------------------------------------------
    if "task_type" not in meta or not meta["task_type"]:
        meta["task_type"] = _detect_task_type(model, class_name)

    # --- Feature count ------------------------------------------------------
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        meta["n_features"] = int(n_features)
    elif "feature_names" in meta:
        meta["n_features"] = len(meta["feature_names"])

    # --- Feature names ------------------------------------------------------
    feat_names = getattr(model, "feature_names_in_", None)
    if feat_names is not None:
        meta.setdefault("feature_names", list(feat_names))

    # --- Hyperparameters ----------------------------------------------------
    try:
        meta["params"] = model.get_params()
    except Exception:
        meta["params"] = {}

    # --- Model family flags -------------------------------------------------
    meta["is_tree_based"] = class_name in TREE_BASED_CLASSES
    meta["is_linear"] = class_name in LINEAR_CLASSES

    return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_task_type(model, class_name: str) -> str:
    """Heuristic to determine if a model is a classifier or regressor."""
    # Check sklearn attribute
    if hasattr(model, "_estimator_type"):
        etype = getattr(model, "_estimator_type")
        if etype == "classifier":
            return "classification"
        elif etype == "regressor":
            return "regression"

    # Keyword matching
    if any(kw in class_name for kw in _CLASSIFIER_KEYWORDS):
        return "classification"
    if any(kw in class_name for kw in _REGRESSOR_KEYWORDS):
        return "regression"

    # Has predict_proba → likely classifier
    if hasattr(model, "predict_proba"):
        return "classification"

    return "unknown"


def _pretty_algorithm_name(class_name: str) -> str:
    """Convert sklearn class name to a readable algorithm label."""
    mapping = {
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest (Classifier)",
        "RandomForestRegressor": "Random Forest (Regressor)",
        "GradientBoostingClassifier": "Gradient Boosting (Classifier)",
        "GradientBoostingRegressor": "Gradient Boosting (Regressor)",
        "DecisionTreeClassifier": "Decision Tree (Classifier)",
        "DecisionTreeRegressor": "Decision Tree (Regressor)",
        "SVC": "SVM (Classifier)",
        "SVR": "SVR (Regressor)",
        "KNeighborsClassifier": "KNN (Classifier)",
        "KNeighborsRegressor": "KNN (Regressor)",
        "LinearRegression": "Linear Regression",
        "XGBClassifier": "XGBoost (Classifier)",
        "XGBRegressor": "XGBoost (Regressor)",
    }
    return mapping.get(class_name, class_name)
