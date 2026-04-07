"""
XAI Studio — Global Configuration & Settings
=============================================
Centralized constants, paths, and default parameters for the entire application.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")

# Ensure directories exist at import time
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_NAME = "XAI Studio"
APP_VERSION = "1.0.0"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
MIN_WINDOW_WIDTH = 960
MIN_WINDOW_HEIGHT = 620

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
MAX_PREVIEW_ROWS = 100

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Supported Models
# ---------------------------------------------------------------------------
CLASSIFICATION_MODELS = {
    "Logistic Regression": {
        "module": "sklearn.linear_model",
        "class": "LogisticRegression",
        "default_params": {"max_iter": 1000, "random_state": DEFAULT_RANDOM_STATE},
    },
    "Random Forest": {
        "module": "sklearn.ensemble",
        "class": "RandomForestClassifier",
        "default_params": {"n_estimators": 100, "random_state": DEFAULT_RANDOM_STATE},
    },
    "SVM": {
        "module": "sklearn.svm",
        "class": "SVC",
        "default_params": {"kernel": "rbf", "random_state": DEFAULT_RANDOM_STATE},
    },
    "KNN": {
        "module": "sklearn.neighbors",
        "class": "KNeighborsClassifier",
        "default_params": {"n_neighbors": 5},
    },
    "Decision Tree": {
        "module": "sklearn.tree",
        "class": "DecisionTreeClassifier",
        "default_params": {"random_state": DEFAULT_RANDOM_STATE},
    },
    "Gradient Boosting": {
        "module": "sklearn.ensemble",
        "class": "GradientBoostingClassifier",
        "default_params": {"n_estimators": 100, "random_state": DEFAULT_RANDOM_STATE},
    },
}

REGRESSION_MODELS = {
    "Linear Regression": {
        "module": "sklearn.linear_model",
        "class": "LinearRegression",
        "default_params": {},
    },
    "Random Forest Regressor": {
        "module": "sklearn.ensemble",
        "class": "RandomForestRegressor",
        "default_params": {"n_estimators": 100, "random_state": DEFAULT_RANDOM_STATE},
    },
    "SVR": {
        "module": "sklearn.svm",
        "class": "SVR",
        "default_params": {"kernel": "rbf"},
    },
    "KNN Regressor": {
        "module": "sklearn.neighbors",
        "class": "KNeighborsRegressor",
        "default_params": {"n_neighbors": 5},
    },
    "Decision Tree Regressor": {
        "module": "sklearn.tree",
        "class": "DecisionTreeRegressor",
        "default_params": {"random_state": DEFAULT_RANDOM_STATE},
    },
    "Gradient Boosting Regressor": {
        "module": "sklearn.ensemble",
        "class": "GradientBoostingRegressor",
        "default_params": {"n_estimators": 100, "random_state": DEFAULT_RANDOM_STATE},
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
