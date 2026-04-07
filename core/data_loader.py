"""
XAI Studio — Data Loader
=========================
Responsible for loading CSV files, validating their content,
and providing summary statistics about the dataset.
"""

import pandas as pd
import numpy as np
from utils.logger import get_logger
from config.settings import SUPPORTED_ENCODINGS, MAX_PREVIEW_ROWS

logger = get_logger(__name__)


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Tries multiple encodings automatically if the default fails.

    Parameters
    ----------
    filepath : str
        Absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed with any supported encoding.
    """
    import os
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if not filepath.lower().endswith(".csv"):
        raise ValueError(f"Unsupported file format. Expected .csv, got: {filepath}")

    last_error = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            if df.empty:
                raise ValueError("The CSV file is empty (no rows).")
            if df.columns.size == 0:
                raise ValueError("The CSV file has no columns.")
            logger.info(
                "Loaded '%s' (%d rows × %d cols, encoding=%s)",
                os.path.basename(filepath), len(df), len(df.columns), encoding,
            )
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            raise ValueError(f"Failed to parse CSV: {exc}") from exc

    raise ValueError(
        f"Could not decode file with any supported encoding "
        f"({', '.join(SUPPORTED_ENCODINGS)}). Last error: {last_error}"
    )


def get_summary(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive summary of the DataFrame.

    Returns
    -------
    dict
        Keys: shape, columns, dtypes, missing_values, missing_pct,
              numeric_stats, categorical_stats, sample_rows.
    """
    n_rows, n_cols = df.shape
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    summary = {
        "shape": (n_rows, n_cols),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_stats": (
            df[numeric_cols].describe().to_dict() if numeric_cols else {}
        ),
        "categorical_stats": {
            col: {
                "unique": df[col].nunique(),
                "top": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "freq": df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0,
            }
            for col in categorical_cols
        },
        "sample_rows": df.head(MAX_PREVIEW_ROWS),
    }
    logger.info(
        "Summary generated: %d numeric cols, %d categorical cols, %.1f%% total missing",
        len(numeric_cols), len(categorical_cols),
        missing.sum() / (n_rows * n_cols) * 100 if n_rows * n_cols > 0 else 0,
    )
    return summary


def detect_target_column(df: pd.DataFrame) -> str | None:
    """
    Heuristic to detect the most likely target column.

    Priority:
    1. Column named 'target', 'label', 'class', 'y' (case-insensitive)
    2. Last column in the DataFrame

    Returns
    -------
    str or None
        Name of the detected target column, or None if DataFrame is empty.
    """
    if df.columns.size == 0:
        return None

    known_targets = ["target", "label", "class", "y", "output", "result"]
    for col in df.columns:
        if col.strip().lower() in known_targets:
            logger.info("Auto-detected target column: '%s' (name match)", col)
            return col

    # Default: last column
    target = df.columns[-1]
    logger.info("Auto-detected target column: '%s' (last column fallback)", target)
    return target
