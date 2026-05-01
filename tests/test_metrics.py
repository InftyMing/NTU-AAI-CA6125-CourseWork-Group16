from __future__ import annotations

import numpy as np

from feedback_ell.metrics import clip_scores, columnwise_rmse, mcrmse


def test_mcrmse_matches_manual_calculation() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [4.0, 6.0]])
    scores = columnwise_rmse(y_true, y_pred, columns=["a", "b"])
    assert round(scores["a"], 6) == round(np.sqrt(0.5), 6)
    assert round(scores["b"], 6) == round(np.sqrt(2.0), 6)
    assert round(mcrmse(y_true, y_pred, columns=["a", "b"]), 6) == round(
        (np.sqrt(0.5) + np.sqrt(2.0)) / 2,
        6,
    )


def test_clip_scores_limits_competition_range() -> None:
    clipped = clip_scores(np.array([[0.2, 6.1, 3.0]]))
    assert clipped.tolist() == [[1.0, 5.0, 3.0]]
