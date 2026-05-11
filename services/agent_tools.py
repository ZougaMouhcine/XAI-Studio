"""
XAI Studio — Agent Tool Registry
==================================
Defines tools the AI agent can call.  Each tool is a thin wrapper
around an existing PipelineService method — no new ML logic.
"""

import json
import traceback
import pandas as pd

from services.pipeline_service import PipelineService
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Tool Definitions (OpenAI function-calling schema) ────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_pipeline_status",
            "description": (
                "Get the current status of the ML pipeline: whether data is loaded, "
                "target column set, preprocessing done, models trained, evaluation done."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_summary",
            "description": (
                "Get a summary of the currently loaded dataset: shape, columns, "
                "data types, missing values, basic statistics."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the list of column names in the current dataset.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_target_column",
            "description": "Set the target column for supervised learning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to use as the prediction target.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_preprocessing_recommendations",
            "description": (
                "Analyze the dataset and return preprocessing recommendations "
                "(missing values, encoding needs, scaling, etc.)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_preprocessing_issues",
            "description": "Detect data quality issues in the current dataset.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_preprocessing",
            "description": (
                "Run the preprocessing pipeline (imputation, encoding, scaling, "
                "train/test split) on the loaded data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_size": {
                        "type": "number",
                        "description": "Fraction of data for testing (default 0.2).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_preprocessing_catalog",
            "description": "Get the available preprocessing categories, methods, and their required options for the dynamic pipeline.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_preprocessing_step",
            "description": "Add a preprocessing step to the dynamic pipeline workspace. Use get_preprocessing_catalog first to see valid categories and methods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The preprocessing category (e.g., 'data_cleaning', 'encoding').",
                    },
                    "method": {
                        "type": "string",
                        "description": "The specific method name (e.g., 'drop_missing', 'onehot').",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The list of column names to apply this step to.",
                    },
                    "options": {
                        "type": "object",
                        "description": "Additional options for the method (if any).",
                    }
                },
                "required": ["category", "method", "columns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_models",
            "description": "List the available ML models for the current task type (classification, regression, clustering).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_training",
            "description": "Train one or more ML models on the preprocessed data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of model names to train. If empty, all available models are trained."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_evaluation",
            "description": "Evaluate all trained models on the test set and return metrics.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_comparison_table",
            "description": "Get the model comparison table showing metrics for all evaluated models.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": (
                "Navigate the application to a specific section/view. "
                "Valid views: data, preprocessing, training, evaluation, prediction, models."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "enum": ["data", "preprocessing", "training", "evaluation", "prediction", "models"],
                        "description": "The view/section to navigate to.",
                    }
                },
                "required": ["view"],
            },
        },
    },
]


# ── Tool Executors ───────────────────────────────────────────────────

class ToolExecutor:
    """Executes agent tools by dispatching to PipelineService methods."""

    def __init__(self, pipeline: PipelineService, navigate_fn=None):
        self.pipeline = pipeline
        self.navigate_fn = navigate_fn

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return a string result for the LLM."""
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return f"Error: Unknown tool '{tool_name}'"
            result = handler(**arguments)
            logger.info("Tool '%s' executed successfully", tool_name)
            return result
        except Exception as exc:
            logger.error("Tool '%s' failed: %s", tool_name, exc)
            return f"Error executing {tool_name}: {exc}"

    # ── individual tool handlers ─────────────────────────────────────

    def _tool_get_pipeline_status(self) -> str:
        p = self.pipeline
        status = {
            "data_loaded": p.dataframe is not None,
            "file": p.filepath or "None",
            "rows": len(p.dataframe) if p.dataframe is not None else 0,
            "columns": len(p.dataframe.columns) if p.dataframe is not None else 0,
            "target_column": p.target_column or "Not set",
            "preprocessing_done": p.preprocessing_result is not None,
            "task_type": (
                p.preprocessing_result.task_type
                if p.preprocessing_result else "Not determined"
            ),
            "models_trained": (
                list(p.trained_models.keys()) if p.trained_models else []
            ),
            "evaluation_done": p.evaluation_results is not None,
        }
        return json.dumps(status, indent=2, default=str)

    def _tool_get_data_summary(self) -> str:
        summary = self.pipeline.get_data_summary()
        if summary is None:
            return "No data loaded. Please load a dataset first."
        # Make it readable
        result = []
        for key, value in summary.items():
            if isinstance(value, pd.DataFrame):
                result.append(f"\n{key}:\n{value.to_string()}")
            elif isinstance(value, dict):
                result.append(f"\n{key}:")
                for k, v in value.items():
                    result.append(f"  {k}: {v}")
            else:
                result.append(f"{key}: {value}")
        return "\n".join(result)

    def _tool_get_columns(self) -> str:
        cols = self.pipeline.get_columns()
        if not cols:
            return "No data loaded."
        return json.dumps(cols)

    def _tool_set_target_column(self, column: str) -> str:
        cols = self.pipeline.get_columns()
        if column not in cols:
            return f"Column '{column}' not found. Available: {cols}"
        self.pipeline.set_target_column(column)
        return f"Target column set to '{column}'."

    def _tool_get_preprocessing_recommendations(self) -> str:
        try:
            recs = self.pipeline.get_preprocessing_recommendations()
            if not recs:
                return "No specific recommendations. The data looks clean."
            return "\n".join(f"• {r}" for r in recs)
        except Exception as exc:
            return f"Cannot generate recommendations: {exc}"

    def _tool_get_preprocessing_issues(self) -> str:
        try:
            issues = self.pipeline.get_preprocessing_issues()
            if not issues:
                return "No issues detected."
            return json.dumps(issues, indent=2, default=str)
        except Exception as exc:
            return f"Cannot detect issues: {exc}"

    def _tool_run_preprocessing(self, test_size: float = 0.2) -> str:
        try:
            result = self.pipeline.run_preprocessing(test_size=test_size)
            return (
                f"Preprocessing complete.\n"
                f"Task type: {result.task_type}\n"
                f"Train set: {result.X_train.shape[0]} samples, {result.X_train.shape[1]} features\n"
                f"Test set: {result.X_test.shape[0]} samples\n"
                f"Features: {result.feature_names[:10]}{'...' if len(result.feature_names) > 10 else ''}"
            )
        except Exception as exc:
            return f"Preprocessing failed: {exc}"

    def _tool_get_preprocessing_catalog(self) -> str:
        try:
            catalog = self.pipeline.get_preprocessing_catalog()
            return json.dumps(catalog, indent=2, ensure_ascii=False)
        except Exception as exc:
            return f"Failed to get catalog: {exc}"

    def _tool_add_preprocessing_step(
        self, category: str, method: str, columns: list[str], options: dict | None = None
    ) -> str:
        try:
            if options is None:
                options = {}
            self.pipeline.add_preprocessing_step(category, method, columns, options)
            
            # If we have a navigate function, we can try to refresh the preprocessing view
            if self.navigate_fn:
                # Trigger a refresh by navigating to the same view
                self.navigate_fn("preprocessing")
                
            return f"[OK] Step added to dynamic pipeline: {category} -> {method} on {len(columns)} columns."
        except Exception as exc:
            return f"[FAIL] Cannot add step: {exc}"

    def _tool_get_available_models(self) -> str:
        names = self.pipeline.get_model_names()
        if not names:
            return "No models available. Run preprocessing first to determine the task type."
        return f"Available models ({len(names)}): " + ", ".join(names)

    def _tool_run_training(self, models: list[str] | None = None) -> str:
        try:
            selected = models if models else None
            trained = self.pipeline.run_training(selected_models=selected)
            results = []
            for name, entry in trained.items():
                if entry.get("model") is not None:
                    results.append(f"[OK] {name} -- trained in {entry['training_time']:.3f}s")
                else:
                    results.append(f"[FAIL] {name} -- {entry.get('error', 'failed')}")
            return "\n".join(results)
        except Exception as exc:
            return f"Training failed: {exc}"

    def _tool_run_evaluation(self) -> str:
        try:
            results = self.pipeline.run_evaluation()
            lines = []
            for name, metrics in results.items():
                if "error" in metrics:
                    lines.append(f"[FAIL] {name}: {metrics['error']}")
                else:
                    metric_str = ", ".join(
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in metrics.items()
                        if k not in ("confusion_matrix", "classification_report", "y_pred")
                    )
                    lines.append(f"[RESULT] {name}: {metric_str}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Evaluation failed: {exc}"

    def _tool_get_comparison_table(self) -> str:
        table = self.pipeline.get_comparison_table()
        if table is None:
            return "No comparison table. Run evaluation first."
        return table.to_string()

    def _tool_navigate_to(self, view: str) -> str:
        valid = ["data", "preprocessing", "training", "evaluation", "prediction", "models"]
        if view not in valid:
            return f"Invalid view '{view}'. Valid views: {valid}"
        if self.navigate_fn:
            self.navigate_fn(view)
            return f"Navigated to '{view}' view."
        return "Navigation function not available."
