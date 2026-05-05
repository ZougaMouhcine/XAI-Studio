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
        "param_grid": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
        "param_meta": {
            "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "desc": "Regularization strength"},
            "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 5000, "desc": "Maximum iterations"},
            "solver": {"type": "choice", "choices": ["lbfgs", "liblinear", "saga"], "default": "lbfgs", "desc": "Optimization solver"},
        },
    },
    "SVM": {
        "module": "sklearn.svm",
        "class": "SVC",
        "default_params": {"kernel": "rbf", "probability": True},
        "param_grid": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        "param_meta": {
            "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "desc": "Penalty parameter"},
            "kernel": {"type": "choice", "choices": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf", "desc": "Kernel type"},
            "gamma": {"type": "choice", "choices": ["scale", "auto"], "default": "scale", "desc": "Kernel coefficient"},
        },
    },
    "KNN": {
        "module": "sklearn.neighbors",
        "class": "KNeighborsClassifier",
        "default_params": {"n_neighbors": 5},
        "param_grid": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
        "param_meta": {
            "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 30, "desc": "Number of neighbors"},
            "weights": {"type": "choice", "choices": ["uniform", "distance"], "default": "uniform", "desc": "Weighting method"},
        },
    },
    "Decision Tree": {
        "module": "sklearn.tree",
        "class": "DecisionTreeClassifier",
        "default_params": {"random_state": DEFAULT_RANDOM_STATE},
        "param_grid": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        "param_meta": {
            "max_depth": {"type": "int", "default": None, "min": 1, "max": 50, "desc": "Maximum tree depth"},
            "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "desc": "Min samples to split"},
        },
    },
    "Random Forest": {
        "module": "sklearn.ensemble",
        "class": "RandomForestClassifier",
        "default_params": {"n_estimators": 100, "random_state": DEFAULT_RANDOM_STATE},
        "param_grid": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 500, "desc": "Number of trees"},
            "max_depth": {"type": "int", "default": None, "min": 1, "max": 50, "desc": "Maximum depth"},
        },
    },
    "Gradient Boosting": {
        "module": "sklearn.ensemble",
        "class": "GradientBoostingClassifier",
        "default_params": {"n_estimators": 100, "random_state": DEFAULT_RANDOM_STATE},
        "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 500, "desc": "Boosting stages"},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "desc": "Learning rate"},
        },
    },
    "XGBoost": {
        "module": "xgboost",
        "class": "XGBClassifier",
        "default_params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
        "param_grid": {"n_estimators": [200, 400], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 200, "min": 50, "max": 800, "desc": "Number of trees"},
            "max_depth": {"type": "int", "default": 6, "min": 2, "max": 12, "desc": "Tree depth"},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "desc": "Learning rate"},
        },
    },
    "LightGBM": {
        "module": "lightgbm",
        "class": "LGBMClassifier",
        "default_params": {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31},
        "param_grid": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 200, "min": 50, "max": 800, "desc": "Number of trees"},
            "num_leaves": {"type": "int", "default": 31, "min": 8, "max": 128, "desc": "Number of leaves"},
        },
    },
    "CatBoost": {
        "module": "catboost",
        "class": "CatBoostClassifier",
        "default_params": {"iterations": 300, "learning_rate": 0.1, "depth": 6, "verbose": False},
        "param_grid": {"iterations": [200, 300], "depth": [4, 6, 8]},
        "param_meta": {
            "iterations": {"type": "int", "default": 300, "min": 50, "max": 800, "desc": "Iterations"},
            "depth": {"type": "int", "default": 6, "min": 2, "max": 10, "desc": "Tree depth"},
        },
    },
    "Naive Bayes": {
        "module": "sklearn.naive_bayes",
        "class": "GaussianNB",
        "default_params": {},
        "param_grid": {},
        "param_meta": {"var_smoothing": {"type": "float", "default": 1e-9, "min": 1e-12, "max": 1e-6, "desc": "Smoothing"}},
    },
    "AdaBoost": {
        "module": "sklearn.ensemble",
        "class": "AdaBoostClassifier",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1},
        "param_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 400, "desc": "Number of estimators"},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.5, "desc": "Learning rate"},
        },
    },
    "Extra Trees": {
        "module": "sklearn.ensemble",
        "class": "ExtraTreesClassifier",
        "default_params": {"n_estimators": 200, "random_state": DEFAULT_RANDOM_STATE},
        "param_grid": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 200, "min": 50, "max": 500, "desc": "Number of trees"},
            "max_depth": {"type": "int", "default": None, "min": 1, "max": 50, "desc": "Maximum depth"},
        },
    },
}

REGRESSION_MODELS = {
    "Linear Regression": {
        "module": "sklearn.linear_model",
        "class": "LinearRegression",
        "default_params": {},
        "param_grid": {},
        "param_meta": {
            "fit_intercept": {"type": "bool", "default": True, "desc": "Fit intercept"},
        },
    },
    "Ridge": {
        "module": "sklearn.linear_model",
        "class": "Ridge",
        "default_params": {"alpha": 1.0},
        "param_grid": {"alpha": [0.1, 1.0, 10.0]},
        "param_meta": {"alpha": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "desc": "Regularization strength"}},
    },
    "Lasso": {
        "module": "sklearn.linear_model",
        "class": "Lasso",
        "default_params": {"alpha": 1.0, "max_iter": 1000},
        "param_grid": {"alpha": [0.1, 1.0, 10.0]},
        "param_meta": {"alpha": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "desc": "Regularization strength"}},
    },
    "ElasticNet": {
        "module": "sklearn.linear_model",
        "class": "ElasticNet",
        "default_params": {"alpha": 1.0, "l1_ratio": 0.5},
        "param_grid": {"alpha": [0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.8]},
        "param_meta": {
            "alpha": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "desc": "Regularization strength"},
            "l1_ratio": {"type": "float", "default": 0.5, "min": 0.1, "max": 0.9, "desc": "Elastic mixing"},
        },
    },
    "SVR": {
        "module": "sklearn.svm",
        "class": "SVR",
        "default_params": {"kernel": "rbf"},
        "param_grid": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
        "param_meta": {
            "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "desc": "Penalty parameter"},
            "kernel": {"type": "choice", "choices": ["rbf", "linear", "poly"], "default": "rbf", "desc": "Kernel type"},
        },
    },
    "Random Forest Regressor": {
        "module": "sklearn.ensemble",
        "class": "RandomForestRegressor",
        "default_params": {"n_estimators": 100, "random_state": DEFAULT_RANDOM_STATE},
        "param_grid": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 500, "desc": "Number of trees"},
            "max_depth": {"type": "int", "default": None, "min": 1, "max": 50, "desc": "Maximum depth"},
        },
    },
    "Gradient Boosting Regressor": {
        "module": "sklearn.ensemble",
        "class": "GradientBoostingRegressor",
        "default_params": {"n_estimators": 100, "random_state": DEFAULT_RANDOM_STATE},
        "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 100, "min": 50, "max": 500, "desc": "Boosting stages"},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 0.3, "desc": "Learning rate"},
        },
    },
    "XGBoost Regressor": {
        "module": "xgboost",
        "class": "XGBRegressor",
        "default_params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
        "param_grid": {"n_estimators": [200, 400], "max_depth": [4, 6, 8]},
        "param_meta": {
            "n_estimators": {"type": "int", "default": 200, "min": 50, "max": 800, "desc": "Number of trees"},
            "max_depth": {"type": "int", "default": 6, "min": 2, "max": 12, "desc": "Tree depth"},
        },
    },
    "Decision Tree Regressor": {
        "module": "sklearn.tree",
        "class": "DecisionTreeRegressor",
        "default_params": {"random_state": DEFAULT_RANDOM_STATE},
        "param_grid": {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
        "param_meta": {
            "max_depth": {"type": "int", "default": None, "min": 1, "max": 50, "desc": "Maximum depth"},
            "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20, "desc": "Min samples to split"},
        },
    },
}

CLUSTERING_MODELS = {
    "KMeans": {
        "module": "sklearn.cluster",
        "class": "KMeans",
        "default_params": {"n_clusters": 3, "n_init": 10},
        "param_grid": {"n_clusters": [2, 3, 4, 5]},
        "param_meta": {
            "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 20, "desc": "Number of clusters"},
            "n_init": {"type": "int", "default": 10, "min": 1, "max": 20, "desc": "Initializations"},
        },
    },
    "DBSCAN": {
        "module": "sklearn.cluster",
        "class": "DBSCAN",
        "default_params": {"eps": 0.5, "min_samples": 5},
        "param_grid": {"eps": [0.3, 0.5, 0.8], "min_samples": [3, 5, 10]},
        "param_meta": {
            "eps": {"type": "float", "default": 0.5, "min": 0.1, "max": 2.0, "desc": "Neighborhood radius"},
            "min_samples": {"type": "int", "default": 5, "min": 2, "max": 20, "desc": "Min samples"},
        },
    },
    "Agglomerative Clustering": {
        "module": "sklearn.cluster",
        "class": "AgglomerativeClustering",
        "default_params": {"n_clusters": 3, "linkage": "ward"},
        "param_grid": {"n_clusters": [2, 3, 4], "linkage": ["ward", "average", "complete"]},
        "param_meta": {
            "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 20, "desc": "Number of clusters"},
            "linkage": {"type": "choice", "choices": ["ward", "average", "complete", "single"], "default": "ward", "desc": "Linkage method"},
        },
    },
    "Gaussian Mixture": {
        "module": "sklearn.mixture",
        "class": "GaussianMixture",
        "default_params": {"n_components": 3, "covariance_type": "full"},
        "param_grid": {"n_components": [2, 3, 4], "covariance_type": ["full", "tied"]},
        "param_meta": {
            "n_components": {"type": "int", "default": 3, "min": 2, "max": 20, "desc": "Components"},
            "covariance_type": {"type": "choice", "choices": ["full", "tied", "diag", "spherical"], "default": "full", "desc": "Covariance"},
        },
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
