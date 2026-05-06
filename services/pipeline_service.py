"""
XAI Studio — Pipeline Service (Business Logic Orchestrator)
=============================================================
Central service that maintains pipeline state and coordinates
all core operations.  The UI calls *only* this service — never
the core modules directly.
"""

from dataclasses import asdict

import pandas as pd
import numpy as np

from core.data_loader import load_tabular, get_summary, detect_target_column
from core.preprocessing import preprocess_data, PreprocessingResult
from core.preprocessing_module import PreprocessingWorkspace, StepSpec, prettify_pipeline_help
from core.training import train_model, train_all_models, get_available_models, get_model_param_schema
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import random
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
        self.preprocessing_workspace = PreprocessingWorkspace()
        self.trained_models: dict | None = None
        self.evaluation_results: dict | None = None
        self.comparison_df: pd.DataFrame | None = None
        self.training_log: list[str] = []
        # Phase 2 — Uploaded model state
        self.loaded_model = None
        self.loaded_model_metadata: dict | None = None
        logger.info("Pipeline state reset")

    # ------------------------------------------------------------------
    # Step 1: Data Loading
    # ------------------------------------------------------------------
    def load_data(self, filepath: str, has_header: bool = True) -> pd.DataFrame:
        """Load a CSV file and compute its summary."""
        self.filepath = filepath
        self.dataframe = load_tabular(filepath, has_header=has_header)
        self.data_summary = get_summary(self.dataframe)
        self.target_column = detect_target_column(self.dataframe)
        self.preprocessing_workspace.reset(self.dataframe)

        # Clear downstream state
        self.preprocessing_result = None
        self.trained_models = None
        self.evaluation_results = None
        self.comparison_df = None

        return self.dataframe

    def get_data_summary(self) -> dict | None:
        return self.data_summary

    def get_preprocessing_workspace(self) -> PreprocessingWorkspace:
        return self.preprocessing_workspace

    def get_preprocessing_catalog(self) -> dict:
        return self.preprocessing_workspace.CATALOG

    def get_preprocessing_help(self) -> str:
        return prettify_pipeline_help()

    def add_preprocessing_step(self, category: str, method: str, columns: list[str], options: dict):
        self.preprocessing_workspace.add_step(StepSpec(category=category, method=method, columns=columns, options=options))

    def remove_preprocessing_step(self, index: int):
        self.preprocessing_workspace.remove_step(index)

    def move_preprocessing_step(self, index: int, direction: str):
        self.preprocessing_workspace.move_step(index, direction)

    def run_preprocessing_step(self, category: str, method: str, columns: list[str], options: dict) -> dict:
        out = self.preprocessing_workspace.run_step(StepSpec(category=category, method=method, columns=columns, options=options))
        if out.ok:
            self.dataframe = self.preprocessing_workspace.current_df.copy(deep=True)
            self.data_summary = get_summary(self.dataframe)
        return {"ok": out.ok, "message": out.message, "details": out.details}

    def run_preprocessing_pipeline_advanced(self) -> list[dict]:
        outcomes = self.preprocessing_workspace.run_pipeline()
        if outcomes:
            self.dataframe = self.preprocessing_workspace.current_df.copy(deep=True)
            self.data_summary = get_summary(self.dataframe)
            if self.target_column not in self.dataframe.columns:
                self.target_column = detect_target_column(self.dataframe)
        return [{"ok": o.ok, "message": o.message, "details": o.details} for o in outcomes]

    def preprocessing_undo(self) -> bool:
        ok = self.preprocessing_workspace.undo()
        if ok:
            self.dataframe = self.preprocessing_workspace.current_df.copy(deep=True)
            self.data_summary = get_summary(self.dataframe)
        return ok

    def preprocessing_redo(self) -> bool:
        ok = self.preprocessing_workspace.redo()
        if ok:
            self.dataframe = self.preprocessing_workspace.current_df.copy(deep=True)
            self.data_summary = get_summary(self.dataframe)
        return ok

    def save_preprocessing_pipeline(self, filepath: str):
        self.preprocessing_workspace.save_pipeline(filepath)

    def load_preprocessing_pipeline(self, filepath: str):
        self.preprocessing_workspace.load_pipeline(filepath)

    def export_preprocessing_pipeline_code(self, filepath: str):
        self.preprocessing_workspace.export_pipeline_code(filepath)

    def get_preprocessing_recommendations(self) -> list[str]:
        return self.preprocessing_workspace.recommend_steps(self.target_column)

    def get_preprocessing_issues(self) -> dict:
        return self.preprocessing_workspace.dataset_issues()

    def get_preprocessing_pipeline(self) -> list[dict]:
        return [asdict(s) for s in self.preprocessing_workspace.pipeline]

    def get_preprocessing_logs(self) -> list[str]:
        return self.preprocessing_workspace.operation_log

    def get_current_dataframe_preview(self, n: int = 100) -> pd.DataFrame:
        if self.preprocessing_workspace.current_df is None:
            return pd.DataFrame()
        return self.preprocessing_workspace.current_df.head(n)

    def get_columns(self) -> list[str]:
        if self.dataframe is not None:
            return self.dataframe.columns.tolist()
        return []

    def set_target_column(self, column: str | None):
        self.target_column = column
        logger.info("Target column set to: '%s'", column)

    # ------------------------------------------------------------------
    # Step 2: Preprocessing
    # ------------------------------------------------------------------
    def run_preprocessing(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        options: dict | None = None,
    ) -> PreprocessingResult:
        """Run the preprocessing pipeline on the loaded data."""
        if self.dataframe is None:
            raise RuntimeError("No data loaded. Please load a CSV file first.")
        if self.target_column is None:
            logger.info("No target selected — switching to clustering mode")

        self.preprocessing_result = preprocess_data(
            self.dataframe,
            target_column=self.target_column,
            test_size=test_size,
            random_state=random_state,
            options=options,
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

    def get_model_registry(self) -> dict:
        """Return model metadata registry for current task type."""
        if self.preprocessing_result is None:
            return {}
        return get_available_models(self.preprocessing_result.task_type)

    def get_model_param_schema(self, model_name: str, include_all: bool = False) -> list[dict]:
        """Return parameter schema for a model name."""
        if self.preprocessing_result is None:
            return []
        return get_model_param_schema(self.preprocessing_result.task_type, model_name, include_all=include_all)

    def get_training_logs(self) -> list[str]:
        return list(self.training_log)

    def run_training(
        self,
        selected_models: list[str] | None = None,
        model_params_map: dict[str, dict] | None = None,
        progress_callback=None,
    ) -> dict:
        """Train models on the preprocessed data."""
        if self.preprocessing_result is None:
            raise RuntimeError("Data not preprocessed yet.")

        pr = self.preprocessing_result
        self.training_log = ["Démarrage entraînement..."]
        self.trained_models = train_all_models(
            pr.X_train, pr.y_train, pr.task_type,
            selected_models=selected_models,
            model_params_map=model_params_map,
            progress_callback=progress_callback,
        )

        for name, entry in self.trained_models.items():
            if entry.get("model") is not None:
                self.training_log.append(f"✅ {name} — {entry['training_time']:.3f}s")
            else:
                self.training_log.append(f"❌ {name} — {entry.get('error', 'error')}")

        # Clear downstream
        self.evaluation_results = None
        self.comparison_df = None

        return self.trained_models

    def run_automl(
        self,
        candidate_models: list[str] | None = None,
        search: str = "grid",
        param_grids: dict | None = None,
        n_iter: int = 20,
        cv: int = 3,
    ) -> dict:
        """Run a simple AutoML search across candidate models.

        Returns a dict with keys: best_model, best_score, best_params, model_name
        """
        if self.preprocessing_result is None:
            raise RuntimeError("Data not preprocessed yet.")

        registry = get_available_models(self.preprocessing_result.task_type)
        names = candidate_models if candidate_models else list(registry.keys())
        X = self.preprocessing_result.X_train
        y = self.preprocessing_result.y_train

        best_overall = {"score": -float("inf"), "model": None, "params": None, "name": None}

        for name in names:
            info = registry.get(name)
            if not info:
                continue
            cls_module = info["module"]
            cls_name = info["class"]
            module = __import__(cls_module, fromlist=[cls_name])
            cls = getattr(module, cls_name)

            grid = (param_grids or {}).get(name) or info.get("param_grid") or {}

            if not grid:
                # If no grid provided, skip complex search and do single fit with defaults
                model = cls(**info.get("default_params", {}))
                model.fit(X, y)
                try:
                    score = model.score(X, y)
                except Exception:
                    score = 0
                if score > best_overall["score"]:
                    best_overall.update({"score": score, "model": model, "params": {}, "name": name})
                continue

            estimator = cls()
            if search == "random":
                searcher = RandomizedSearchCV(estimator, grid, n_iter=n_iter, cv=cv, n_jobs=1)
            elif search in ("grid", "basic"):
                searcher = GridSearchCV(estimator, grid, cv=cv, n_jobs=1)
            else:
                searcher = GridSearchCV(estimator, grid, cv=cv, n_jobs=1)

            searcher.fit(X, y)
            if hasattr(searcher, "best_score_") and searcher.best_score_ > best_overall["score"]:
                best_overall.update({
                    "score": float(searcher.best_score_),
                    "model": searcher.best_estimator_,
                    "params": dict(searcher.best_params_),
                    "name": name,
                })

        # Optional Optuna / Bayesian optimization
        if search in ("optuna", "bayes"):
            try:
                import optuna
                from sklearn.model_selection import cross_val_score
            except Exception as exc:
                raise RuntimeError("Optuna is required for Bayesian optimization.") from exc

            def _suggest(trial, param_name, values):
                if all(isinstance(v, int) for v in values):
                    return trial.suggest_int(param_name, min(values), max(values))
                if all(isinstance(v, float) for v in values):
                    return trial.suggest_float(param_name, min(values), max(values))
                return trial.suggest_categorical(param_name, values)

            def objective(trial):
                model_name = trial.suggest_categorical("model", names)
                info = registry.get(model_name)
                if not info:
                    return -1
                grid = (param_grids or {}).get(model_name) or info.get("param_grid") or {}
                params = {k: _suggest(trial, k, v) for k, v in grid.items() if isinstance(v, list) and v}

                module = __import__(info["module"], fromlist=[info["class"]])
                cls = getattr(module, info["class"])
                est = cls(**params)
                scores = cross_val_score(est, X, y, cv=cv)
                return float(scores.mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_iter)
            best_params = dict(study.best_params)
            best_name = best_params.pop("model", names[0] if names else None)
            best_info = registry.get(best_name)
            if best_info:
                module = __import__(best_info["module"], fromlist=[best_info["class"]])
                cls = getattr(module, best_info["class"])
                model = cls(**best_params)
                model.fit(X, y)
                best_overall.update({"score": float(study.best_value), "model": model, "params": best_params, "name": best_name})

        return best_overall

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
            "input_feature_names": (
                self.preprocessing_result.input_feature_names
                if self.preprocessing_result else []
            ),
            "numeric_feature_names": (
                self.preprocessing_result.numeric_feature_names
                if self.preprocessing_result else []
            ),
            "categorical_feature_names": (
                self.preprocessing_result.categorical_feature_names
                if self.preprocessing_result else []
            ),
            "feature_schema": (
                self.preprocessing_result.feature_schema
                if self.preprocessing_result else []
            ),
            "training_time": entry.get("training_time", 0),
            "preprocessing_artifacts": (
                {
                    "num_imputer": self.preprocessing_result.encoders.get("num_imputer") if self.preprocessing_result else None,
                    "cat_imputer": self.preprocessing_result.encoders.get("cat_imputer") if self.preprocessing_result else None,
                    "one_hot_encoder": self.preprocessing_result.encoders.get("one_hot_encoder") if self.preprocessing_result else None,
                    "scaler": self.preprocessing_result.scaler if self.preprocessing_result else None,
                }
                if self.preprocessing_result else {}
            ),
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
