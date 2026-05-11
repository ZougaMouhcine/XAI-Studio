"""
XAI Studio — Preprocessing Pipeline
=====================================
Automatic preprocessing: type detection, missing value imputation,
encoding, scaling, and train/test splitting.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from config.settings import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Container for all preprocessing outputs."""

    X_train: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None
    feature_names: list[str] = field(default_factory=list)
    input_feature_names: list[str] = field(default_factory=list)
    numeric_feature_names: list[str] = field(default_factory=list)
    categorical_feature_names: list[str] = field(default_factory=list)
    feature_schema: list[dict] = field(default_factory=list)
    target_names: list[str] = field(default_factory=list)
    task_type: str = ""  # "classification" or "regression"
    label_encoder: LabelEncoder | None = None
    encoders: dict[str, Any] = field(default_factory=dict)
    scaler: StandardScaler | None = None
    summary: dict = field(default_factory=dict)


def _detect_task_type(y: pd.Series | pd.DataFrame) -> str:
    """Determine if the target represents a classification or regression task."""
    if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
        # For simplicity, if multi-target, assume classification if any column looks categorical
        for col in y.columns:
            if y[col].dtype == "object" or y[col].dtype.name == "category" or y[col].nunique() <= 20:
                return "classification"
        return "regression"

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 20 and n_unique / len(y) < 0.05:
        return "classification"
    return "regression"


def preprocess_data(
    df: pd.DataFrame,
    target_columns: list[str] | None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    options: dict[str, Any] | None = None,
) -> PreprocessingResult:
    """
    Run the full preprocessing pipeline on a DataFrame.

    Steps
    -----
    1. Separate features (X) and target (y).
    2. Detect task type (classification / regression).
    3. Impute missing values (median for numeric, mode for categorical).
    4. Encode the target if classification (LabelEncoder).
    5. One-hot encode categorical features.
    6. Scale numeric features (StandardScaler).
    7. Split into train / test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame.
    target_columns : list[str] | None
        List of target column names.
    test_size : float
        Proportion of the data reserved for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    PreprocessingResult
        Dataclass with all the outputs needed for training.
    """
    result = PreprocessingResult()
    result.target_names = target_columns or []
    options = options or {}

    apply_imputation = options.get("apply_imputation", True)
    apply_encoding = options.get("apply_encoding", True)
    apply_scaling = options.get("apply_scaling", True)

    numeric_impute_strategy = options.get("numeric_impute_strategy", "median")
    categorical_impute_strategy = options.get("categorical_impute_strategy", "most_frequent")
    if categorical_impute_strategy == "majority_voting":
        categorical_impute_strategy = "most_frequent"
    numeric_fill_value = options.get("numeric_fill_value", 0)
    categorical_fill_value = options.get("categorical_fill_value", "missing")

    if target_columns:
        for col in target_columns:
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found in DataFrame.")

    # ------------------------------------------------------------------
    # 1. Separate X / y
    # ------------------------------------------------------------------
    if target_columns:
        X = df.drop(columns=target_columns).copy()
        y = df[target_columns].copy()
        if len(target_columns) == 1:
            y = y.iloc[:, 0]
    else:
        X = df.copy()
        y = None

    # ------------------------------------------------------------------
    # 2. Detect task type
    # ------------------------------------------------------------------
    if y is None:
        result.task_type = "clustering"
    else:
        result.task_type = _detect_task_type(y)
    logger.info("Detected task type: %s", result.task_type)

    # ------------------------------------------------------------------
    # 3. Impute missing values
    # ------------------------------------------------------------------
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    result.input_feature_names = X.columns.tolist()
    result.numeric_feature_names = numeric_cols
    result.categorical_feature_names = categorical_cols

    if apply_imputation:
        if numeric_cols:
            num_kwargs = {"strategy": numeric_impute_strategy}
            if numeric_impute_strategy == "constant":
                num_kwargs["fill_value"] = numeric_fill_value
            num_imputer = SimpleImputer(**num_kwargs)
            X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
            result.encoders["num_imputer"] = num_imputer
            logger.info("Imputed %d numeric columns (%s strategy)", len(numeric_cols), numeric_impute_strategy)

        if categorical_cols:
            cat_kwargs = {"strategy": categorical_impute_strategy}
            if categorical_impute_strategy == "constant":
                cat_kwargs["fill_value"] = categorical_fill_value
            cat_imputer = SimpleImputer(**cat_kwargs)
            X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
            result.encoders["cat_imputer"] = cat_imputer
            logger.info("Imputed %d categorical columns (%s strategy)", len(categorical_cols), categorical_impute_strategy)

    # Handle missing target values
    if y is not None:
        if isinstance(y, pd.DataFrame):
            missing_mask = y.isnull().any(axis=1)
            if missing_mask.any():
                n_missing = int(missing_mask.sum())
                X = X.loc[~missing_mask]
                y = y.loc[~missing_mask]
                logger.warning("Dropped %d rows with missing target values", n_missing)
        else:
            if y.isnull().any():
                n_missing = int(y.isnull().sum())
                mask = y.notnull()
                X = X[mask]
                y = y[mask]
                logger.warning("Dropped %d rows with missing target values", n_missing)

    # ------------------------------------------------------------------
    # 4. Encode target (classification only)
    # ------------------------------------------------------------------
    if result.task_type == "classification" and y is not None:
        if isinstance(y, pd.Series):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_columns[0], index=y.index)
            result.label_encoder = le
            logger.info(
                "Encoded target: %d classes → %s",
                len(le.classes_), list(le.classes_),
            )
        else:
            logger.info("Multi-column target detected, skipping LabelEncoder.")

    # ------------------------------------------------------------------
    # 5. One-hot encode categorical features
    # ------------------------------------------------------------------
    if categorical_cols and apply_encoding:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(X[categorical_cols])
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
        encoded_df = pd.DataFrame(encoded, columns=ohe_feature_names, index=X.index)

        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, encoded_df], axis=1)
        result.encoders["one_hot_encoder"] = ohe
        logger.info(
            "One-hot encoded %d categorical columns → %d new features",
            len(categorical_cols), len(ohe_feature_names),
        )
    elif categorical_cols and not apply_encoding:
        raise ValueError(
            "Des colonnes catégorielles existent. Activez l'encodage ou supprimez les colonnes catégorielles."
        )

    # ------------------------------------------------------------------
    # 6. Scale numeric features
    # ------------------------------------------------------------------
    if apply_scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        result.scaler = scaler
        logger.info("Scaled %d features with StandardScaler", X_scaled.shape[1])
    else:
        X_scaled = X.values
        logger.info("Skipped feature scaling")
    result.feature_names = X.columns.tolist()
    result.feature_schema = []
    for col in numeric_cols:
        result.feature_schema.append({"name": col, "type": "numeric", "default": None, "choices": None})
    if categorical_cols:
        choices_map = {}
        ohe = result.encoders.get("one_hot_encoder")
        if ohe is not None and hasattr(ohe, "categories_"):
            for col, choices in zip(categorical_cols, ohe.categories_):
                choices_map[col] = [str(choice) for choice in list(choices)]
        for col in categorical_cols:
            choices = choices_map.get(col, [])
            result.feature_schema.append({
                "name": col,
                "type": "choice" if choices else "categorical",
                "default": choices[0] if choices else None,
                "choices": choices,
            })

    # ------------------------------------------------------------------
    # 7. Train / test split
    # ------------------------------------------------------------------
    if result.task_type == "clustering":
        result.X_train = X_scaled
        result.X_test = X_scaled
        result.y_train = None
        result.y_test = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values, test_size=test_size, random_state=random_state,
        )
        result.X_train = X_train
        result.X_test = X_test
        result.y_train = y_train
        result.y_test = y_test

    result.summary = {
        "original_shape": df.shape,
        "features_count": len(result.feature_names),
        "numeric_original": len(numeric_cols),
        "categorical_original": len(categorical_cols),
        "train_size": len(result.X_train),
        "test_size": len(result.X_test),
        "task_type": result.task_type,
        "options": {
            "apply_imputation": apply_imputation,
            "apply_encoding": apply_encoding,
            "apply_scaling": apply_scaling,
            "numeric_impute_strategy": numeric_impute_strategy,
            "categorical_impute_strategy": categorical_impute_strategy,
        },
    }

    logger.info(
        "Preprocessing complete — train: %d, test: %d, features: %d",
        len(result.X_train), len(result.X_test), len(result.feature_names),
    )
    return result
