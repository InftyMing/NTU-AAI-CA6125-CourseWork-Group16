"""Metrics used by the official competition and local experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feedback_ell.constants import TARGET_COLUMNS


def _to_numpy(values: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def columnwise_rmse(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, float]:
    """Return RMSE for each target column."""

    target_names = columns or TARGET_COLUMNS
    true_arr = _to_numpy(y_true)
    pred_arr = _to_numpy(y_pred)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={true_arr.shape}, y_pred={pred_arr.shape}")
    if true_arr.ndim != 2:
        raise ValueError("MCRMSE expects a 2D target matrix.")
    if len(target_names) != true_arr.shape[1]:
        raise ValueError("Number of target names must match target width.")

    rmse = np.sqrt(np.mean((true_arr - pred_arr) ** 2, axis=0))
    return {name: float(value) for name, value in zip(target_names, rmse)}


def mcrmse(
    y_true: np.ndarray | pd.DataFrame,
    y_pred: np.ndarray | pd.DataFrame,
    columns: list[str] | None = None,
) -> float:
    """Mean column-wise root mean squared error."""

    scores = columnwise_rmse(y_true, y_pred, columns=columns)
    return float(np.mean(list(scores.values())))


def clip_scores(predictions: np.ndarray) -> np.ndarray:
    """Clip predictions to the competition score range."""

    return np.clip(predictions, 1.0, 5.0)
