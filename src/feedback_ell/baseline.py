"""Traditional baselines and lightweight ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency in smoke tests
    LGBMRegressor = None

from feedback_ell.constants import ID_COLUMN, TARGET_COLUMNS
from feedback_ell.data import load_sample_submission, load_test, load_train, make_folds
from feedback_ell.features import (
    build_tfidf_vectorizer,
    handcrafted_matrix,
    transform_tfidf,
)
from feedback_ell.metrics import clip_scores, columnwise_rmse, mcrmse
from feedback_ell.utils import ensure_dir, set_seed, write_json


@dataclass
class ExperimentResult:
    name: str
    cv_mcrmse: float
    column_rmse: dict[str, float]
    fold_scores: list[dict[str, Any]]
    submission_path: str | None = None


def save_submission(
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
    output_path: str | Path,
) -> None:
    sub = sample_submission.copy()
    sub[TARGET_COLUMNS] = clip_scores(predictions)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output, index=False)


def _evaluate_fold(y_valid: np.ndarray, pred: np.ndarray, fold: int) -> dict[str, Any]:
    return {
        "fold": int(fold),
        "mcrmse": mcrmse(y_valid, pred),
        "column_rmse": columnwise_rmse(y_valid, pred),
    }


def run_mean_baseline(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, output_dir: Path):
    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    means = y.mean(axis=0)
    oof = np.tile(means, (len(train), 1))
    test_pred = np.tile(means, (len(test), 1))
    submission_path = output_dir / "submission_mean_baseline.csv"
    save_submission(sample, test_pred, submission_path)
    return ExperimentResult(
        name="mean_baseline",
        cv_mcrmse=mcrmse(y, oof),
        column_rmse=columnwise_rmse(y, oof),
        fold_scores=[],
        submission_path=str(submission_path),
    ), oof, test_pred


def run_ridge_tfidf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    model_dir: Path,
) -> tuple[ExperimentResult, np.ndarray, np.ndarray]:
    params = config["models"]["ridge_tfidf"]
    vectorizer = build_tfidf_vectorizer(params["word_max_features"], params["char_max_features"])
    all_features = vectorizer.fit_transform(pd.concat([train, test], axis=0)["full_text"].astype(str))
    x_train = all_features[: len(train)]
    x_test = all_features[len(train) :]
    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    oof = np.zeros_like(y)
    test_pred = np.zeros((len(test), len(TARGET_COLUMNS)), dtype=float)
    fold_scores = []
    folds = sorted(train["fold"].unique())

    for fold in folds:
        trn_idx = train.index[train["fold"] != fold].to_numpy()
        val_idx = train.index[train["fold"] == fold].to_numpy()
        model = Ridge(alpha=float(params["ridge_alpha"]), random_state=config["seed"])
        model.fit(x_train[trn_idx], y[trn_idx])
        pred = model.predict(x_train[val_idx])
        oof[val_idx] = pred
        test_pred += model.predict(x_test) / len(folds)
        fold_scores.append(_evaluate_fold(y[val_idx], pred, int(fold)))
        joblib.dump(model, model_dir / f"ridge_tfidf_fold{fold}.joblib")

    joblib.dump(vectorizer, model_dir / "ridge_tfidf_vectorizer.joblib")
    submission_path = output_dir / "submission_ridge_tfidf.csv"
    save_submission(sample, test_pred, submission_path)
    result = ExperimentResult(
        name="ridge_tfidf",
        cv_mcrmse=mcrmse(y, oof),
        column_rmse=columnwise_rmse(y, oof),
        fold_scores=fold_scores,
        submission_path=str(submission_path),
    )
    return result, oof, test_pred


def run_svr_tfidf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[ExperimentResult, np.ndarray, np.ndarray]:
    params = config["models"]["svr_tfidf"]
    vectorizer = build_tfidf_vectorizer(30000, 30000)
    x_train = vectorizer.fit_transform(train["full_text"].astype(str))
    x_test = transform_tfidf(vectorizer, test)
    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    oof = np.zeros_like(y)
    test_pred = np.zeros((len(test), len(TARGET_COLUMNS)), dtype=float)
    fold_scores = []
    folds = sorted(train["fold"].unique())

    for fold in folds:
        trn_idx = train.index[train["fold"] != fold].to_numpy()
        val_idx = train.index[train["fold"] == fold].to_numpy()
        model = MultiOutputRegressor(SVR(C=float(params["C"]), epsilon=float(params["epsilon"])))
        model.fit(x_train[trn_idx], y[trn_idx])
        pred = model.predict(x_train[val_idx])
        oof[val_idx] = pred
        test_pred += model.predict(x_test) / len(folds)
        fold_scores.append(_evaluate_fold(y[val_idx], pred, int(fold)))

    submission_path = output_dir / "submission_svr_tfidf.csv"
    save_submission(sample, test_pred, submission_path)
    result = ExperimentResult(
        name="svr_tfidf",
        cv_mcrmse=mcrmse(y, oof),
        column_rmse=columnwise_rmse(y, oof),
        fold_scores=fold_scores,
        submission_path=str(submission_path),
    )
    return result, oof, test_pred


def run_lgbm_text_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[ExperimentResult, np.ndarray, np.ndarray]:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed.")
    params = config["models"]["lightgbm_features"]
    x_train = handcrafted_matrix(train)
    x_test = handcrafted_matrix(test)
    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    oof = np.zeros_like(y)
    test_pred = np.zeros((len(test), len(TARGET_COLUMNS)), dtype=float)
    fold_scores = []
    folds = sorted(train["fold"].unique())

    for fold in folds:
        trn_idx = train.index[train["fold"] != fold].to_numpy()
        val_idx = train.index[train["fold"] == fold].to_numpy()
        fold_pred = np.zeros((len(val_idx), len(TARGET_COLUMNS)), dtype=float)
        fold_test = np.zeros((len(test), len(TARGET_COLUMNS)), dtype=float)
        for col_idx, target in enumerate(TARGET_COLUMNS):
            model = LGBMRegressor(
                n_estimators=int(params["n_estimators"]),
                learning_rate=float(params["learning_rate"]),
                num_leaves=int(params["num_leaves"]),
                random_state=int(config["seed"]) + col_idx,
                verbosity=-1,
            )
            model.fit(x_train[trn_idx], y[trn_idx, col_idx])
            fold_pred[:, col_idx] = model.predict(x_train[val_idx])
            fold_test[:, col_idx] = model.predict(x_test)
        oof[val_idx] = fold_pred
        test_pred += fold_test / len(folds)
        fold_scores.append(_evaluate_fold(y[val_idx], fold_pred, int(fold)))

    submission_path = output_dir / "submission_lgbm_text_features.csv"
    save_submission(sample, test_pred, submission_path)
    result = ExperimentResult(
        name="lgbm_text_features",
        cv_mcrmse=mcrmse(y, oof),
        column_rmse=columnwise_rmse(y, oof),
        fold_scores=fold_scores,
        submission_path=str(submission_path),
    )
    return result, oof, test_pred


def run_baselines(config: dict[str, Any]) -> list[ExperimentResult]:
    set_seed(int(config["seed"]))
    output_dir = ensure_dir(config["output"]["submissions_dir"])
    artifact_dir = ensure_dir(config["output"]["artifacts_dir"])
    model_dir = ensure_dir("experiments/models")
    train = load_train(config["data"]["train_path"])
    test = load_test(config["data"]["test_path"])
    sample = load_sample_submission(config["data"]["sample_submission_path"])

    if config.get("debug_rows"):
        train = train.head(int(config["debug_rows"])).copy()
        test = test.head(min(len(test), 16)).copy()
        sample = sample.head(len(test)).copy()

    train = make_folds(train, int(config["n_splits"]), int(config["seed"]))
    results: list[ExperimentResult] = []
    oof_payload: dict[str, Any] = {}

    if config["models"]["mean"]["enabled"]:
        result, oof, test_pred = run_mean_baseline(train, test, sample, output_dir)
        results.append(result)
        oof_payload[result.name] = {"oof": oof.tolist(), "test": test_pred.tolist()}

    if config["models"]["ridge_tfidf"]["enabled"]:
        result, oof, test_pred = run_ridge_tfidf(train, test, sample, config, output_dir, model_dir)
        results.append(result)
        oof_payload[result.name] = {"oof": oof.tolist(), "test": test_pred.tolist()}

    if config["models"]["svr_tfidf"]["enabled"] and len(train) <= int(
        config["models"]["svr_tfidf"]["max_train_rows"]
    ):
        result, oof, test_pred = run_svr_tfidf(train, test, sample, config, output_dir)
        results.append(result)
        oof_payload[result.name] = {"oof": oof.tolist(), "test": test_pred.tolist()}

    if config["models"]["lightgbm_features"]["enabled"]:
        result, oof, test_pred = run_lgbm_text_features(train, test, sample, config, output_dir)
        results.append(result)
        oof_payload[result.name] = {"oof": oof.tolist(), "test": test_pred.tolist()}

    summary = [
        {
            "name": item.name,
            "cv_mcrmse": item.cv_mcrmse,
            "column_rmse": item.column_rmse,
            "fold_scores": item.fold_scores,
            "submission_path": item.submission_path,
        }
        for item in sorted(results, key=lambda x: x.cv_mcrmse)
    ]
    write_json(summary, artifact_dir / "baseline_metrics.json")
    write_json(oof_payload, artifact_dir / "baseline_predictions.json")
    return results
