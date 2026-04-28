"""
Professional preprocessing workspace for interactive ML desktop workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import tempfile
import textwrap

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA, KernelPCA

from utils.logger import get_logger

logger = get_logger(__name__)


try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler

    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

try:
    import umap  # type: ignore

    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    VIZ_AVAILABLE = True
except Exception:
    VIZ_AVAILABLE = False


@dataclass
class StepSpec:
    category: str
    method: str
    columns: list[str]
    options: dict[str, Any]


@dataclass
class StepOutcome:
    ok: bool
    message: str
    details: dict[str, Any]


class PreprocessingWorkspace:
    """Interactive preprocessing workspace with undo/redo and dynamic steps."""

    CATALOG: dict[str, list[str]] = {
        "data_cleaning": [
            "detect_missing",
            "drop_nan_rows",
            "drop_nan_columns",
            "impute",
            "impute_knn",
            "remove_duplicates",
            "auto_correct_types",
            "convert_dates",
            "clean_categories",
        ],
        "outlier_handling": [
            "detect_iqr",
            "detect_zscore",
            "detect_isolation_forest",
            "handle_outliers",
        ],
        "feature_transformation": [
            "log_transform",
            "sqrt_transform",
            "power_transform",
            "quantile_transform",
            "binning",
            "polynomial_features",
        ],
        "encoding": [
            "label_encoding",
            "one_hot_encoding",
            "ordinal_encoding",
            "target_encoding",
            "binary_encoding",
        ],
        "scaling": [
            "standard_scaler",
            "minmax_scaler",
            "robust_scaler",
            "maxabs_scaler",
            "normalizer",
        ],
        "feature_selection": [
            "correlation_selection",
            "chi_square",
            "anova",
            "mutual_information",
            "rfe",
            "select_k_best",
            "lasso_selection",
            "rf_importance",
            "xgboost_importance",
        ],
        "feature_reduction": [
            "pca",
            "kernel_pca",
            "tsne",
            "umap",
        ],
        "data_balancing": [
            "random_oversampling",
            "random_undersampling",
            "smote",
            "adasyn",
            "class_weight",
        ],
        "data_splitting": [
            "train_test_split",
            "validation_split",
            "kfold",
            "stratified_kfold",
        ],
        "data_visualization": [
            "histogram",
            "boxplot",
            "correlation_heatmap",
            "scatter",
            "pairplot",
            "missing_heatmap",
            "distribution",
        ],
    }

    def __init__(self, df: pd.DataFrame | None = None):
        self.original_df = df.copy(deep=True) if df is not None else pd.DataFrame()
        self.current_df = df.copy(deep=True) if df is not None else pd.DataFrame()
        self.pipeline: list[StepSpec] = []
        self.history: list[pd.DataFrame] = [self.current_df.copy(deep=True)]
        self.future: list[pd.DataFrame] = []
        self.operation_log: list[str] = []
        self.last_details: dict[str, Any] = {}
        self.splits: dict[str, Any] = {}

    def reset(self, df: pd.DataFrame):
        self.original_df = df.copy(deep=True)
        self.current_df = df.copy(deep=True)
        self.pipeline.clear()
        self.history = [self.current_df.copy(deep=True)]
        self.future = []
        self.operation_log.clear()
        self.last_details = {}
        self.splits = {}

    def snapshot(self):
        self.history.append(self.current_df.copy(deep=True))
        if len(self.history) > 30:
            self.history = self.history[-30:]
        self.future.clear()

    def undo(self) -> bool:
        if len(self.history) <= 1:
            return False
        self.future.append(self.history.pop())
        self.current_df = self.history[-1].copy(deep=True)
        self._log("Undo")
        return True

    def redo(self) -> bool:
        if not self.future:
            return False
        nxt = self.future.pop()
        self.history.append(nxt.copy(deep=True))
        self.current_df = nxt.copy(deep=True)
        self._log("Redo")
        return True

    def add_step(self, step: StepSpec):
        self.pipeline.append(step)
        self._log(f"Step added: {step.category}/{step.method}")

    def remove_step(self, index: int):
        self.pipeline.pop(index)
        self._log(f"Step removed: #{index}")

    def move_step(self, index: int, direction: str):
        if direction == "up" and index > 0:
            self.pipeline[index - 1], self.pipeline[index] = self.pipeline[index], self.pipeline[index - 1]
        elif direction == "down" and index < len(self.pipeline) - 1:
            self.pipeline[index + 1], self.pipeline[index] = self.pipeline[index], self.pipeline[index + 1]
        self._log(f"Step moved: #{index} {direction}")

    def run_step(self, step: StepSpec) -> StepOutcome:
        try:
            outcome = self._apply_step(step)
            if outcome.ok and step.category != "data_visualization":
                self.snapshot()
            self.last_details = outcome.details
            self._log(f"{step.category}/{step.method}: {outcome.message}")
            return outcome
        except Exception as exc:
            logger.exception("Error in preprocessing step")
            msg = f"Erreur: {exc}"
            self._log(f"{step.category}/{step.method}: {msg}")
            return StepOutcome(False, msg, {"error": str(exc)})

    def run_pipeline(self) -> list[StepOutcome]:
        results: list[StepOutcome] = []
        for step in self.pipeline:
            results.append(self.run_step(step))
        return results

    def save_pipeline(self, filepath: str):
        payload = [asdict(s) for s in self.pipeline]
        joblib.dump(payload, filepath)
        self._log(f"Pipeline saved: {filepath}")

    def load_pipeline(self, filepath: str):
        payload = joblib.load(filepath)
        self.pipeline = [StepSpec(**p) for p in payload]
        self._log(f"Pipeline loaded: {filepath}")

    def export_pipeline_code(self, filepath: str):
        lines = [
            "import pandas as pd",
            "from sklearn.impute import SimpleImputer, KNNImputer",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer",
            "# Generated from XAI Studio preprocessing pipeline",
            "",
            "def run_pipeline(df):",
            "    df = df.copy()",
        ]
        for step in self.pipeline:
            lines.append(
                f"    # {step.category}/{step.method} columns={step.columns} options={step.options}"
            )
        lines.append("    return df")
        Path(filepath).write_text("\n".join(lines), encoding="utf-8")
        self._log(f"Pipeline exported: {filepath}")

    def recommend_steps(self, target_column: str | None = None) -> list[str]:
        df = self.current_df
        recs: list[str] = []
        if df.empty:
            return recs

        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        if (missing_pct > 0).any():
            recs.append("Data Cleaning: impute or drop missing values")
        if df.duplicated().any():
            recs.append("Data Cleaning: remove duplicates")

        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            recs.append("Data Cleaning: convert_dates + clean_categories")
            recs.append("Encoding: one_hot_encoding or target_encoding")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            recs.append("Outlier Handling: detect_iqr then handle_outliers")
            recs.append("Feature Scaling: robust_scaler")

        if target_column and target_column in df.columns:
            y = df[target_column]
            if y.nunique() <= 30 and y.dtype != float:
                cls_dist = y.value_counts(normalize=True)
                if len(cls_dist) > 1 and cls_dist.max() > 0.65:
                    recs.append("Data Balancing: smote or random_oversampling")
            recs.append("Feature Selection: mutual_information")
            recs.append("Feature Reduction: pca")

        recs.append("Data Splitting: train_test_split or stratified_kfold")
        return recs

    def dataset_issues(self) -> dict[str, Any]:
        df = self.current_df
        if df.empty:
            return {}
        issues = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing_total": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "object_cols": df.select_dtypes(include=["object"]).columns.tolist(),
            "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        }
        return issues

    def _apply_step(self, step: StepSpec) -> StepOutcome:
        cat = step.category
        method = step.method
        columns = step.columns or []
        opts = step.options or {}

        if cat == "data_cleaning":
            return self._step_data_cleaning(method, columns, opts)
        if cat == "outlier_handling":
            return self._step_outliers(method, columns, opts)
        if cat == "feature_transformation":
            return self._step_transform(method, columns, opts)
        if cat == "encoding":
            return self._step_encoding(method, columns, opts)
        if cat == "scaling":
            return self._step_scaling(method, columns, opts)
        if cat == "feature_selection":
            return self._step_feature_selection(method, columns, opts)
        if cat == "feature_reduction":
            return self._step_feature_reduction(method, columns, opts)
        if cat == "data_balancing":
            return self._step_balancing(method, columns, opts)
        if cat == "data_splitting":
            return self._step_splitting(method, columns, opts)
        if cat == "data_visualization":
            return self._step_visualization(method, columns, opts)
        raise ValueError(f"Unknown category: {cat}")

    def _resolve_columns(self, columns: list[str], numeric_only: bool = False) -> list[str]:
        df = self.current_df
        cols = [c for c in columns if c in df.columns] if columns else df.columns.tolist()
        if numeric_only:
            cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        return cols

    def _step_data_cleaning(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        df = self.current_df
        cols = self._resolve_columns(columns)

        if method == "detect_missing":
            missing_pct = (df[cols].isna().mean() * 100).round(2).to_dict()
            return StepOutcome(True, "Missing values detected", {"missing_pct": missing_pct})

        if method == "drop_nan_rows":
            before = len(df)
            self.current_df = df.dropna(subset=cols).reset_index(drop=True)
            return StepOutcome(True, "Rows with NaN removed", {"removed": before - len(self.current_df)})

        if method == "drop_nan_columns":
            before_cols = set(df.columns)
            self.current_df = df.dropna(axis=1)
            removed = sorted(before_cols - set(self.current_df.columns))
            return StepOutcome(True, "Columns with NaN removed", {"removed_columns": removed})

        if method == "impute":
            strategy = opts.get("strategy", "mean")
            fill_value = opts.get("constant_value", 0)
            target_cols = cols
            if strategy == "mode":
                strategy = "most_frequent"
            kwargs = {"strategy": strategy}
            if strategy == "constant":
                kwargs["fill_value"] = fill_value
            imputer = SimpleImputer(**kwargs)
            self.current_df[target_cols] = imputer.fit_transform(self.current_df[target_cols])
            return StepOutcome(True, "Imputation applied", {"strategy": strategy, "columns": target_cols})

        if method == "impute_knn":
            k = int(opts.get("n_neighbors", 5))
            num_cols = self._resolve_columns(cols, numeric_only=True)
            imputer = KNNImputer(n_neighbors=k)
            self.current_df[num_cols] = imputer.fit_transform(self.current_df[num_cols])
            return StepOutcome(True, "KNN imputer applied", {"columns": num_cols, "n_neighbors": k})

        if method == "remove_duplicates":
            before = len(df)
            self.current_df = df.drop_duplicates().reset_index(drop=True)
            return StepOutcome(True, "Duplicates removed", {"removed": before - len(self.current_df)})

        if method == "auto_correct_types":
            converted = []
            for c in cols:
                s = self.current_df[c]
                if s.dtype == object:
                    num = pd.to_numeric(s, errors="coerce")
                    if num.notna().mean() > 0.8:
                        self.current_df[c] = num
                        converted.append((c, "numeric"))
                        continue
                    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                    if dt.notna().mean() > 0.8:
                        self.current_df[c] = dt
                        converted.append((c, "datetime"))
            return StepOutcome(True, "Types corrected", {"converted": converted})

        if method == "convert_dates":
            converted = []
            for c in cols:
                if self.current_df[c].dtype == object:
                    dt = pd.to_datetime(self.current_df[c], errors="coerce", infer_datetime_format=True)
                    if dt.notna().mean() > 0.6:
                        self.current_df[c] = dt
                        converted.append(c)
            return StepOutcome(True, "Date conversion done", {"converted_columns": converted})

        if method == "clean_categories":
            cleaned = []
            for c in cols:
                if self.current_df[c].dtype == object:
                    self.current_df[c] = self.current_df[c].astype(str).str.strip().str.lower()
                    cleaned.append(c)
            return StepOutcome(True, "Categories cleaned", {"columns": cleaned})

        raise ValueError(f"Unsupported data_cleaning method: {method}")

    def _detect_outliers_mask(self, df: pd.DataFrame, cols: list[str], method: str) -> pd.Series:
        if not cols:
            return pd.Series([False] * len(df), index=df.index)

        if method == "detect_iqr":
            q1 = df[cols].quantile(0.25)
            q3 = df[cols].quantile(0.75)
            iqr = q3 - q1
            mask = ((df[cols] < (q1 - 1.5 * iqr)) | (df[cols] > (q3 + 1.5 * iqr))).any(axis=1)
            return mask

        if method == "detect_zscore":
            z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0)
            mask = (z.abs() > 3).any(axis=1)
            return mask

        if method == "detect_isolation_forest":
            model = IsolationForest(random_state=42, contamination="auto")
            pred = model.fit_predict(df[cols].fillna(df[cols].median()))
            return pd.Series(pred == -1, index=df.index)

        raise ValueError(f"Unknown outlier detector: {method}")

    def _step_outliers(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        cols = self._resolve_columns(columns, numeric_only=True)
        df = self.current_df

        if method in {"detect_iqr", "detect_zscore", "detect_isolation_forest"}:
            mask = self._detect_outliers_mask(df, cols, method)
            return StepOutcome(True, "Outliers detected", {"count": int(mask.sum()), "ratio": float(mask.mean())})

        if method == "handle_outliers":
            detector = opts.get("detector", "detect_iqr")
            action = opts.get("action", "remove")
            mask = self._detect_outliers_mask(df, cols, detector)

            if action == "remove":
                self.current_df = df.loc[~mask].reset_index(drop=True)
            elif action == "replace_median":
                for c in cols:
                    med = df[c].median()
                    self.current_df.loc[mask, c] = med
            elif action == "cap_floor":
                lower_q = float(opts.get("lower_q", 0.01))
                upper_q = float(opts.get("upper_q", 0.99))
                for c in cols:
                    lo = df[c].quantile(lower_q)
                    hi = df[c].quantile(upper_q)
                    self.current_df[c] = self.current_df[c].clip(lo, hi)
            else:
                raise ValueError(f"Unknown action: {action}")

            return StepOutcome(True, "Outlier handling applied", {"detector": detector, "action": action, "affected_rows": int(mask.sum())})

        raise ValueError(f"Unsupported outlier method: {method}")

    def _step_transform(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        cols = self._resolve_columns(columns, numeric_only=True)
        df = self.current_df
        before = df[cols].describe().to_dict() if cols else {}

        if method == "log_transform":
            for c in cols:
                min_v = df[c].min()
                shift = abs(min_v) + 1 if min_v <= 0 else 0
                self.current_df[c] = np.log1p(df[c] + shift)

        elif method == "sqrt_transform":
            for c in cols:
                min_v = df[c].min()
                shift = abs(min_v) if min_v < 0 else 0
                self.current_df[c] = np.sqrt(df[c] + shift)

        elif method == "power_transform":
            p_method = opts.get("power_method", "yeo-johnson")
            pt = PowerTransformer(method=p_method)
            self.current_df[cols] = pt.fit_transform(df[cols])

        elif method == "quantile_transform":
            n_quantiles = int(opts.get("n_quantiles", min(1000, max(10, len(df)))))
            output_distribution = opts.get("output_distribution", "normal")
            qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=42)
            self.current_df[cols] = qt.fit_transform(df[cols])

        elif method == "binning":
            bins = int(opts.get("bins", 5))
            for c in cols:
                self.current_df[f"{c}_bin"] = pd.cut(df[c], bins=bins, labels=False, duplicates="drop")

        elif method == "polynomial_features":
            degree = int(opts.get("degree", 2))
            include_bias = bool(opts.get("include_bias", False))
            pf = PolynomialFeatures(degree=degree, include_bias=include_bias)
            arr = pf.fit_transform(df[cols])
            names = pf.get_feature_names_out(cols)
            poly_df = pd.DataFrame(arr, columns=names, index=df.index)
            drop_original = bool(opts.get("drop_original", False))
            if drop_original:
                base = df.drop(columns=cols)
            else:
                base = df.copy()
            self.current_df = pd.concat([base, poly_df], axis=1)

        else:
            raise ValueError(f"Unsupported feature transformation: {method}")

        after_cols = self._resolve_columns(columns, numeric_only=True)
        after = self.current_df[after_cols].describe().to_dict() if after_cols else {}
        return StepOutcome(True, "Feature transformation applied", {"before": before, "after": after})

    def _step_encoding(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        df = self.current_df
        cols = self._resolve_columns(columns)
        cols = [c for c in cols if df[c].dtype == object or str(df[c].dtype).startswith("category")]

        if method == "label_encoding":
            enc_map = {}
            for c in cols:
                le = LabelEncoder()
                self.current_df[c] = le.fit_transform(df[c].astype(str))
                enc_map[c] = list(le.classes_)
            return StepOutcome(True, "Label encoding applied", {"columns": cols, "classes": enc_map})

        if method == "one_hot_encoding":
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            arr = ohe.fit_transform(df[cols].astype(str))
            names = ohe.get_feature_names_out(cols)
            out = pd.DataFrame(arr, columns=names, index=df.index)
            self.current_df = pd.concat([df.drop(columns=cols), out], axis=1)
            return StepOutcome(True, "One-hot encoding applied", {"new_columns": len(names)})

        if method == "ordinal_encoding":
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.current_df[cols] = oe.fit_transform(df[cols].astype(str))
            return StepOutcome(True, "Ordinal encoding applied", {"columns": cols})

        if method == "target_encoding":
            target = opts.get("target_column")
            if not target or target not in df.columns:
                raise ValueError("target_column required for target encoding")
            for c in cols:
                means = df.groupby(c)[target].mean()
                self.current_df[c] = df[c].map(means)
            return StepOutcome(True, "Target encoding applied", {"columns": cols, "target": target})

        if method == "binary_encoding":
            created = []
            for c in cols:
                cats = df[c].astype("category")
                codes = cats.cat.codes.clip(lower=0)
                max_bits = max(1, int(np.ceil(np.log2(max(1, codes.max() + 1)))))
                for bit in range(max_bits):
                    name = f"{c}_bin_{bit}"
                    self.current_df[name] = ((codes >> bit) & 1).astype(int)
                    created.append(name)
                self.current_df = self.current_df.drop(columns=[c])
            return StepOutcome(True, "Binary encoding applied", {"created_columns": created})

        raise ValueError(f"Unsupported encoding method: {method}")

    def _step_scaling(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        cols = self._resolve_columns(columns, numeric_only=True)
        if not cols:
            return StepOutcome(True, "No numeric columns to scale", {})

        if method == "standard_scaler":
            scaler = StandardScaler()
        elif method == "minmax_scaler":
            scaler = MinMaxScaler()
        elif method == "robust_scaler":
            scaler = RobustScaler()
        elif method == "maxabs_scaler":
            scaler = MaxAbsScaler()
        elif method == "normalizer":
            scaler = Normalizer()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        before = self.current_df[cols].describe().to_dict()
        self.current_df[cols] = scaler.fit_transform(self.current_df[cols])
        after = self.current_df[cols].describe().to_dict()
        return StepOutcome(True, "Scaling applied", {"columns": cols, "before": before, "after": after})

    def _prepare_xy(self, target: str, dropna_target: bool = True) -> tuple[pd.DataFrame, pd.Series, str]:
        df = self.current_df
        if target not in df.columns:
            raise ValueError("target_column not found")
        y = df[target]
        X = df.drop(columns=[target])
        if dropna_target:
            mask = y.notna()
            X = X.loc[mask]
            y = y.loc[mask]
        task = "classification" if (y.dtype == object or y.nunique() <= 20) else "regression"
        X = pd.get_dummies(X, drop_first=False)
        X = X.fillna(0)
        return X, y, task

    def _step_feature_selection(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        target = opts.get("target_column")
        X, y, task = self._prepare_xy(target)
        k = int(opts.get("k", min(10, X.shape[1])))
        details: dict[str, Any] = {}

        if method == "correlation_selection":
            threshold = float(opts.get("threshold", 0.95))
            corr = X.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
            keep = [c for c in X.columns if c not in drop_cols]
            self.current_df = pd.concat([self.current_df[[target]], X[keep]], axis=1)
            details = {"dropped": drop_cols, "kept": keep}

        elif method == "chi_square":
            y_enc = y.astype("category").cat.codes if task == "classification" else y
            X_pos = X - X.min() + 1e-9
            sel = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
            sel.fit(X_pos, y_enc)
            scores = pd.Series(sel.scores_, index=X.columns).sort_values(ascending=False)
            details = {"ranking": scores.head(50).to_dict()}

        elif method == "anova":
            score_fn = f_classif if task == "classification" else f_regression
            sel = SelectKBest(score_func=score_fn, k=min(k, X.shape[1]))
            sel.fit(X, y)
            scores = pd.Series(sel.scores_, index=X.columns).sort_values(ascending=False)
            details = {"ranking": scores.head(50).to_dict()}

        elif method == "mutual_information":
            score_fn = mutual_info_classif if task == "classification" else mutual_info_regression
            scores = score_fn(X, y)
            series = pd.Series(scores, index=X.columns).sort_values(ascending=False)
            details = {"ranking": series.head(50).to_dict()}

        elif method == "rfe":
            estimator = RandomForestClassifier(random_state=42) if task == "classification" else RandomForestRegressor(random_state=42)
            n_features = int(opts.get("n_features", min(k, X.shape[1])))
            rfe = RFE(estimator, n_features_to_select=n_features)
            rfe.fit(X, y)
            ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
            details = {"ranking": ranking.head(50).to_dict()}

        elif method == "select_k_best":
            score_fn = f_classif if task == "classification" else f_regression
            sel = SelectKBest(score_func=score_fn, k=min(k, X.shape[1]))
            sel.fit(X, y)
            scores = pd.Series(sel.scores_, index=X.columns).sort_values(ascending=False)
            selected = scores.head(min(k, len(scores))).index.tolist()
            self.current_df = pd.concat([self.current_df[[target]], X[selected]], axis=1)
            details = {"selected": selected, "ranking": scores.head(50).to_dict()}

        elif method == "lasso_selection":
            alpha = float(opts.get("alpha", 0.01))
            if task == "classification":
                y_work = y.astype("category").cat.codes
            else:
                y_work = y
            model = Lasso(alpha=alpha, random_state=42, max_iter=5000)
            model.fit(X, y_work)
            coefs = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=False)
            selected = coefs[coefs > 0].index.tolist()
            details = {"selected": selected[:k], "ranking": coefs.head(50).to_dict()}

        elif method in {"rf_importance", "xgboost_importance"}:
            if method == "xgboost_importance":
                try:
                    from xgboost import XGBClassifier, XGBRegressor  # type: ignore

                    model = XGBClassifier(random_state=42) if task == "classification" else XGBRegressor(random_state=42)
                except Exception:
                    model = RandomForestClassifier(random_state=42) if task == "classification" else RandomForestRegressor(random_state=42)
            else:
                model = RandomForestClassifier(random_state=42) if task == "classification" else RandomForestRegressor(random_state=42)
            model.fit(X, y)
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            details = {"ranking": imp.head(100).to_dict()}

        else:
            raise ValueError(f"Unsupported feature selection method: {method}")

        return StepOutcome(True, "Feature selection executed", details)

    def _step_feature_reduction(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        target = opts.get("target_column")
        X, y, _ = self._prepare_xy(target)
        n_components = int(opts.get("n_components", 2))
        replace_features = bool(opts.get("replace_features", False))

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            arr = reducer.fit_transform(X)
            details = {"explained_variance_ratio": reducer.explained_variance_ratio_.tolist()}

        elif method == "kernel_pca":
            kernel = opts.get("kernel", "rbf")
            reducer = KernelPCA(n_components=n_components, kernel=kernel, random_state=42)
            arr = reducer.fit_transform(X)
            details = {"kernel": kernel}

        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, max(5, len(X) // 10)))
            arr = reducer.fit_transform(X)
            details = {"algorithm": "t-SNE"}

        elif method == "umap":
            if not UMAP_AVAILABLE:
                raise RuntimeError("UMAP is not installed")
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            arr = reducer.fit_transform(X)
            details = {"algorithm": "UMAP"}

        else:
            raise ValueError(f"Unsupported feature reduction method: {method}")

        comp_cols = [f"{method}_comp_{i+1}" for i in range(arr.shape[1])]
        comp_df = pd.DataFrame(arr, columns=comp_cols, index=self.current_df.index)

        if replace_features:
            self.current_df = pd.concat([self.current_df[[target]], comp_df], axis=1)
        else:
            self.current_df = pd.concat([self.current_df, comp_df], axis=1)

        details["projection_columns"] = comp_cols
        return StepOutcome(True, "Feature reduction executed", details)

    def _step_balancing(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        if not IMBLEARN_AVAILABLE and method in {"random_oversampling", "random_undersampling", "smote", "adasyn"}:
            raise RuntimeError("imbalanced-learn is not installed")

        target = opts.get("target_column")
        if not target or target not in self.current_df.columns:
            raise ValueError("target_column required for balancing")

        X, y, _ = self._prepare_xy(target)
        before = pd.Series(y).value_counts().to_dict()

        if method == "random_oversampling":
            sampler = RandomOverSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
        elif method == "random_undersampling":
            sampler = RandomUnderSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
        elif method == "smote":
            sampler = SMOTE(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
        elif method == "adasyn":
            sampler = ADASYN(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
        elif method == "class_weight":
            counts = pd.Series(y).value_counts()
            total = len(y)
            weights = {str(cls): float(total / (len(counts) * cnt)) for cls, cnt in counts.items()}
            return StepOutcome(True, "Class weights computed", {"class_weights": weights, "before": before})
        else:
            raise ValueError(f"Unsupported balancing method: {method}")

        out_df = pd.DataFrame(X_res, columns=X.columns)
        out_df[target] = y_res
        self.current_df = out_df.reset_index(drop=True)
        after = pd.Series(y_res).value_counts().to_dict()
        return StepOutcome(True, "Balancing applied", {"before": before, "after": after})

    def _step_splitting(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        target = opts.get("target_column")
        X, y, task = self._prepare_xy(target)
        random_state = int(opts.get("random_state", 42))

        if method == "train_test_split":
            test_size = float(opts.get("test_size", 0.2))
            stratify = y if (task == "classification" and bool(opts.get("stratify", True))) else None
            split = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
            self.splits["train_test"] = split
            return StepOutcome(True, "Train/Test split created", {"test_size": test_size, "train": len(split[0]), "test": len(split[1])})

        if method == "validation_split":
            val_size = float(opts.get("val_size", 0.2))
            split = train_test_split(X, y, test_size=val_size, random_state=random_state)
            self.splits["validation"] = split
            return StepOutcome(True, "Validation split created", {"val_size": val_size, "train": len(split[0]), "val": len(split[1])})

        if method == "kfold":
            n_splits = int(opts.get("n_splits", 5))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = list(kf.split(X))
            self.splits["kfold"] = folds
            return StepOutcome(True, "KFold created", {"n_splits": n_splits})

        if method == "stratified_kfold":
            n_splits = int(opts.get("n_splits", 5))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = list(skf.split(X, y))
            self.splits["stratified_kfold"] = folds
            return StepOutcome(True, "Stratified KFold created", {"n_splits": n_splits})

        raise ValueError(f"Unsupported split method: {method}")

    def _step_visualization(self, method: str, columns: list[str], opts: dict[str, Any]) -> StepOutcome:
        if not VIZ_AVAILABLE:
            raise RuntimeError("matplotlib/seaborn are not installed")

        df = self.current_df.copy()
        cols = self._resolve_columns(columns)
        if not cols:
            cols = df.columns.tolist()[:4]

        fig, ax = plt.subplots(figsize=(7, 4))

        if method in {"histogram", "distribution"}:
            sns.histplot(df[cols[0]].dropna(), kde=True, ax=ax)
        elif method == "boxplot":
            sns.boxplot(data=df[cols], ax=ax)
        elif method == "correlation_heatmap":
            num = df.select_dtypes(include=[np.number])
            sns.heatmap(num.corr(), cmap="Blues", ax=ax)
        elif method == "scatter":
            if len(cols) < 2:
                raise ValueError("Scatter needs at least 2 columns")
            sns.scatterplot(data=df, x=cols[0], y=cols[1], ax=ax)
        elif method == "pairplot":
            pp = sns.pairplot(df[cols].dropna().head(500))
            path = Path(tempfile.gettempdir()) / f"xai_pairplot_{np.random.randint(1_000_000)}.png"
            pp.savefig(path)
            plt.close("all")
            return StepOutcome(True, "Pairplot generated", {"image_path": str(path)})
        elif method == "missing_heatmap":
            sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
        else:
            raise ValueError(f"Unsupported visualization method: {method}")

        path = Path(tempfile.gettempdir()) / f"xai_viz_{np.random.randint(1_000_000)}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return StepOutcome(True, "Visualization generated", {"image_path": str(path)})

    def _log(self, message: str):
        self.operation_log.append(message)
        logger.info("[PreprocessingWorkspace] %s", message)


def prettify_pipeline_help() -> str:
    return textwrap.dedent(
        """
        Format options JSON examples:
        {"strategy": "median"}
        {"target_column": "label", "k": 10}
        {"detector": "detect_iqr", "action": "cap_floor", "lower_q": 0.01, "upper_q": 0.99}
        """
    ).strip()
