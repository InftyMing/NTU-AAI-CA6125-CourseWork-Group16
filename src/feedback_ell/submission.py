"""Submission selection and ensemble utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feedback_ell.baseline import save_submission
from feedback_ell.constants import TARGET_COLUMNS
from feedback_ell.metrics import mcrmse
from feedback_ell.utils import read_json, write_json


def build_weighted_ensemble(
    sample_submission_path: str | Path,
    train_targets: pd.DataFrame,
    candidate_predictions: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Choose simple non-negative weights by grid search over available OOF predictions."""

    names = list(candidate_predictions)
    if len(names) < 2:
        raise ValueError("At least two candidate prediction sets are required for an ensemble.")

    y = train_targets[TARGET_COLUMNS].to_numpy(dtype=float)
    best = {"score": float("inf"), "weights": None}
    grid = np.linspace(0, 1, 11)
    if len(names) == 2:
        weight_grid = [(w, 1 - w) for w in grid]
    else:
        weight_grid = []
        for w0 in grid:
            for w1 in grid:
                if w0 + w1 <= 1:
                    weight_grid.append((w0, w1, 1 - w0 - w1))

    for weights in weight_grid:
        oof = sum(
            float(weight) * np.asarray(candidate_predictions[name]["oof"], dtype=float)
            for name, weight in zip(names, weights)
        )
        score = mcrmse(y, oof)
        if score < best["score"]:
            best = {"score": score, "weights": dict(zip(names, map(float, weights)))}

    test_pred = sum(
        best["weights"][name] * np.asarray(candidate_predictions[name]["test"], dtype=float)
        for name in names
    )
    sample = pd.read_csv(sample_submission_path)
    save_submission(sample, test_pred, output_path)
    result = {
        "name": "weighted_ensemble",
        "cv_mcrmse": float(best["score"]),
        "weights": best["weights"],
        "submission_path": str(output_path),
    }
    write_json(result, Path(output_path).with_suffix(".metrics.json"))
    return result


def choose_best_submission(
    metrics_paths: list[str | Path],
    output_path: str | Path = "experiments/artifacts/final_selection.json",
) -> dict[str, Any]:
    candidates = []
    for path in metrics_paths:
        payload = read_json(path, default=None)
        if payload is None:
            continue
        if isinstance(payload, list):
            candidates.extend(payload)
        elif isinstance(payload, dict):
            if "components" in payload:
                if isinstance(payload.get("components"), list):
                    candidates.extend(payload["components"])
                if isinstance(payload.get("ensemble"), dict):
                    candidates.append(payload["ensemble"])
            else:
                candidates.append(payload)
    candidates = [
        item
        for item in candidates
        if item.get("submission_path") and item.get("cv_mcrmse") is not None
    ]
    best = min(candidates, key=lambda item: item["cv_mcrmse"]) if candidates else {}
    write_json({"best": best, "candidates": candidates}, output_path)
    return best
