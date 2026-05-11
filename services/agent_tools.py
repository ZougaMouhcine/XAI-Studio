"""
XAI Studio — Agent Tool Registry
==================================
Defines tools the AI agent can call.  Each tool is a thin wrapper
around an existing PipelineService method — no new ML logic.
"""

import json
import threading
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
            "name": "set_target_columns",
            "description": "Set the target columns for supervised learning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target column names (e.g. ['label'])"
                    }
                },
                "required": ["columns"],
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
            "description": (
                "CRITICAL: Use this tool to visually construct the pipeline in the UI. "
                "Adds a single preprocessing step to the dynamic pipeline workspace. "
                "You MUST call this tool for EACH step you want to add. "
                "Use get_preprocessing_catalog first to see valid categories and methods."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The preprocessing category (e.g., 'data_cleaning', 'encoding').",
                    },
                    "method": {
                        "type": "string",
                        "description": "The specific method name (e.g., 'impute', 'one_hot_encoding').",
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

    def __init__(self, pipeline: PipelineService, navigate_fn=None, get_preprocessing_view=None):
        self.pipeline = pipeline
        self.navigate_fn = navigate_fn
        self._get_preprocessing_view = get_preprocessing_view

    def execute(self, tool_name: str, arguments: dict | None) -> str:
        """Execute a tool and return a string result for the LLM."""
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return f"Error: Unknown tool '{tool_name}'"
            
            # Gemini sometimes returns None for empty arguments
            args = arguments or {}
            result = handler(**args)
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
            "target_columns": ", ".join(p.target_columns) if p.target_columns else "Not set",
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

    def _tool_set_target_columns(self, columns: list[str]) -> str:
        avail_cols = self.pipeline.get_columns()
        for col in columns:
            if col not in avail_cols:
                return f"Column '{col}' not found. Available: {avail_cols}"
        self.pipeline.set_target_columns(columns)
        return f"Target columns set to {columns}."

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

            # Try visual UI automation first
            view = self._get_preprocessing_view() if self._get_preprocessing_view else None

            if view is not None and self.navigate_fn:
                # Navigate to the preprocessing view so the user can watch
                self.navigate_fn("preprocessing")

                # Thread-safe synchronisation: schedule on Tk main thread,
                # then wait for the on_done callback to fire.
                done_event = threading.Event()
                error_holder: list[str] = []

                def _on_done():
                    done_event.set()

                def _schedule():
                    try:
                        view.agent_add_step(
                            category=category,
                            method=method,
                            columns=columns,
                            options=options,
                            delay_ms=300,
                            on_done=_on_done,
                        )
                    except Exception as exc:
                        error_holder.append(str(exc))
                        done_event.set()

                # Schedule on the Tk main thread
                view.after(0, _schedule)

                # Wait up to 15 seconds for the UI animation to finish
                done_event.wait(timeout=15)

                if error_holder:
                    return f"[FAIL] Cannot add step visually: {error_holder[0]}"

                return (
                    f"[OK] Step visually added to dynamic pipeline: "
                    f"{category} -> {method} on {columns}."
                )

            # Fallback: no view available — add silently
            self.pipeline.add_preprocessing_step(category, method, columns, options)
            if self.navigate_fn:
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
