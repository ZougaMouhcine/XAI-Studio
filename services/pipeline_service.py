"""
XAI Studio — Pipeline Service (Business Logic Orchestrator)
=============================================================
Central service that maintains pipeline state and coordinates
all core operations.  The UI calls *only* this service — never
the core modules directly.
"""

import pandas as pd
import numpy as np

from core.data_loader import load_csv, get_summary, detect_target_column
from core.preprocessing import preprocess_data, PreprocessingResult
from core.training import train_model, train_all_models, get_available_models
from core.evaluation import evaluate_model, compare_models
from core.persistence import save_model, load_model, list_saved_models, delete_model
from core.model_loader import load_model_file, detect_model_info
from utils.logger import get_logger

logger = get_logger(__name__)


class PipelineService:
    """
    Singleton-style orchestrator that holds the current pipeline state.

    Attributes
    ----------
    dataframe : pd.DataFrame | None
    data_summary : dict | None
    filepath : str | None
    target_column : str | None
    preprocessing_result : PreprocessingResult | None
    trained_models : dict[str, dict] | None
    evaluation_results : dict[str, dict] | None
    comparison_df : pd.DataFrame | None
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset()

    def reset(self):
        """Clear all pipeline state."""
        self.dataframe: pd.DataFrame | None = None
        self.data_summary: dict | None = None
        self.filepath: str | None = None
        self.target_column: str | None = None
        self.preprocessing_result: PreprocessingResult | None = None
        self.trained_models: dict | None = None
        self.evaluation_results: dict | None = None
        self.comparison_df: pd.DataFrame | None = None
        # Phase 2 — Uploaded model state
        self.loaded_model = None
        self.loaded_model_metadata: dict | None = None
        logger.info("Pipeline state reset")

    # ------------------------------------------------------------------
    # Step 1: Data Loading
    # ------------------------------------------------------------------
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file and compute its summary."""
        self.filepath = filepath
        self.dataframe = load_csv(filepath)
        self.data_summary = get_summary(self.dataframe)
        self.target_column = detect_target_column(self.dataframe)

        # Clear downstream state
        self.preprocessing_result = None
        self.trained_models = None
        self.evaluation_results = None
        self.comparison_df = None

        return self.dataframe

    def get_data_summary(self) -> dict | None:
        return self.data_summary

    def get_columns(self) -> list[str]:
        if self.dataframe is not None:
            return self.dataframe.columns.tolist()
        return []

    def set_target_column(self, column: str):
        self.target_column = column
        logger.info("Target column set to: '%s'", column)

    # ------------------------------------------------------------------
    # Step 2: Preprocessing
    # ------------------------------------------------------------------
    def run_preprocessing(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> PreprocessingResult:
        """Run the preprocessing pipeline on the loaded data."""
        if self.dataframe is None:
            raise RuntimeError("No data loaded. Please load a CSV file first.")
        if not self.target_column:
            raise RuntimeError("No target column selected.")

        self.preprocessing_result = preprocess_data(
            self.dataframe,
            target_column=self.target_column,
            test_size=test_size,
            random_state=random_state,
        )

        # Clear downstream
        self.trained_models = None
        self.evaluation_results = None
        self.comparison_df = None

        return self.preprocessing_result

    # ------------------------------------------------------------------
    # Step 3: Training
    # ------------------------------------------------------------------
    def get_model_names(self) -> list[str]:
        """Return available model names for the current task type."""
        if self.preprocessing_result is None:
            return []
        return list(get_available_models(self.preprocessing_result.task_type).keys())

    def run_training(
        self,
        selected_models: list[str] | None = None,
        progress_callback=None,
    ) -> dict:
        """Train models on the preprocessed data."""
        if self.preprocessing_result is None:
            raise RuntimeError("Data not preprocessed yet.")

        pr = self.preprocessing_result
        self.trained_models = train_all_models(
            pr.X_train, pr.y_train, pr.task_type,
            selected_models=selected_models,
            progress_callback=progress_callback,
        )

        # Clear downstream
        self.evaluation_results = None
        self.comparison_df = None

        return self.trained_models

    # ------------------------------------------------------------------
    # Step 4: Evaluation
    # ------------------------------------------------------------------
    def run_evaluation(self) -> dict:
        """Evaluate all trained models on the test set."""
        if self.trained_models is None:
            raise RuntimeError("No models trained yet.")
        if self.preprocessing_result is None:
            raise RuntimeError("No preprocessing result available.")

        pr = self.preprocessing_result
        self.evaluation_results = {}

        for name, entry in self.trained_models.items():
            model = entry.get("model")
            if model is None:
                self.evaluation_results[name] = {"error": entry.get("error", "Training failed")}
                continue
            self.evaluation_results[name] = evaluate_model(
                model, pr.X_test, pr.y_test, pr.task_type, model_name=name,
            )

        self.comparison_df = compare_models(self.evaluation_results, pr.task_type)
        return self.evaluation_results

    def get_comparison_table(self) -> pd.DataFrame | None:
        return self.comparison_df

    # ------------------------------------------------------------------
    # Step 5: Persistence
    # ------------------------------------------------------------------
    def save_trained_model(self, model_name: str) -> str:
        """Save a specific trained model to disk."""
        if self.trained_models is None or model_name not in self.trained_models:
            raise RuntimeError(f"Model '{model_name}' not found in trained models.")

        entry = self.trained_models[model_name]
        if entry.get("model") is None:
            raise RuntimeError(f"Model '{model_name}' failed training — cannot save.")

        metadata = {
            "model_name": model_name,
            "task_type": self.preprocessing_result.task_type if self.preprocessing_result else "unknown",
            "target_column": self.target_column,
            "feature_names": (
                self.preprocessing_result.feature_names
                if self.preprocessing_result else []
            ),
            "training_time": entry.get("training_time", 0),
        }

        # Attach evaluation metrics if available
        if self.evaluation_results and model_name in self.evaluation_results:
            eval_metrics = {
                k: v for k, v in self.evaluation_results[model_name].items()
                if k not in ("confusion_matrix", "classification_report")
            }
            metadata["metrics"] = eval_metrics

        return save_model(entry["model"], metadata, model_name=model_name)

    def save_all_trained_models(self) -> list[str]:
        """Save all successfully trained models."""
        paths = []
        if not self.trained_models:
            return paths
        for name, entry in self.trained_models.items():
            if entry.get("model") is not None:
                path = self.save_trained_model(name)
                paths.append(path)
        return paths

    @staticmethod
    def load_saved_model(filepath: str):
        return load_model(filepath)

    @staticmethod
    def get_saved_models():
        return list_saved_models()

    @staticmethod
    def delete_saved_model(filepath: str):
        return delete_model(filepath)

    # ------------------------------------------------------------------
    # Phase 2: External Model Loading
    # ------------------------------------------------------------------
    def load_external_model(self, filepath: str):
        """Load an external .pkl/.joblib model and store it in state."""
        model, raw_meta = load_model_file(filepath)
        self.loaded_model = model
        self.loaded_model_metadata = detect_model_info(model, raw_meta)
        logger.info("External model loaded: %s", self.loaded_model_metadata.get("algorithm", "unknown"))
        return model, self.loaded_model_metadata

    def get_active_model(self):
        """
        Return the currently active model for XAI analysis.

        Priority: uploaded model > first successfully trained model.

        Returns
        -------
        tuple[model, metadata_dict]
        """
        if self.loaded_model is not None:
            return self.loaded_model, self.loaded_model_metadata or {}

        if self.trained_models:
            for name, entry in self.trained_models.items():
                if entry.get("model") is not None:
                    meta = {
                        "model_name": name,
                        "task_type": self.preprocessing_result.task_type if self.preprocessing_result else "unknown",
                        "feature_names": self.preprocessing_result.feature_names if self.preprocessing_result else [],
                    }
                    meta = detect_model_info(entry["model"], meta)
                    return entry["model"], meta

        return None, {}
