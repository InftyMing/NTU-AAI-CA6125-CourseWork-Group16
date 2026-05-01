"""Enhanced experiments: per-target Ridge tuning, fused features, stacked ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

from feedback_ell.baseline import save_submission
from feedback_ell.constants import TARGET_COLUMNS
from feedback_ell.data import load_sample_submission, load_test, load_train, make_folds
from feedback_ell.features import (
    TEXT_FEATURE_COLUMNS,
    build_tfidf_vectorizer,
    handcrafted_matrix,
)
from feedback_ell.metrics import clip_scores, columnwise_rmse, mcrmse
from feedback_ell.utils import ensure_dir, set_seed, write_json


@dataclass
class ModelOutputs:
    name: str
    oof: np.ndarray
    test_pred: np.ndarray
    cv_mcrmse: float
    column_rmse: dict[str, float]
    fold_scores: list[dict[str, Any]]
    submission_path: str | None = None
    extras: dict[str, Any] | None = None


def _fold_record(y_valid: np.ndarray, pred: np.ndarray, fold: int) -> dict[str, Any]:
    return {
        "fold": int(fold),
        "mcrmse": mcrmse(y_valid, pred),
        "column_rmse": columnwise_rmse(y_valid, pred),
    }


def _scale_features(train_arr: np.ndarray, *other_arrays: np.ndarray):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_arr)
    scaled_others = [scaler.transform(arr) for arr in other_arrays]
    return scaled_train, scaled_others, scaler


def _ridge_per_target(
    x_train: sparse.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    x_valid: sparse.csr_matrix | np.ndarray,
    x_test: sparse.csr_matrix | np.ndarray,
    alphas: list[float],
    seed: int,
    inner_train_for_search: sparse.csr_matrix | np.ndarray | None = None,
    inner_y_for_search: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train one Ridge per target and select alpha via a small validation split."""

    rng = np.random.default_rng(seed)
    chosen: list[float] = []
    val_pred = np.zeros((x_valid.shape[0], len(TARGET_COLUMNS)), dtype=float)
    test_pred = np.zeros((x_test.shape[0], len(TARGET_COLUMNS)), dtype=float)

    if inner_train_for_search is None or inner_y_for_search is None:
        # Carve out a 10% inner split on the training fold for alpha selection.
        n = x_train.shape[0]
        inner_idx = rng.permutation(n)
        cut = max(1, int(n * 0.1))
        search_val_idx = inner_idx[:cut]
        search_trn_idx = inner_idx[cut:]
        inner_train = x_train[search_trn_idx]
        inner_train_y = y_train[search_trn_idx]
        inner_val = x_train[search_val_idx]
        inner_val_y = y_train[search_val_idx]
    else:
        inner_train = inner_train_for_search
        inner_train_y = inner_y_for_search
        inner_val = x_train
        inner_val_y = y_train

    for col_idx, _name in enumerate(TARGET_COLUMNS):
        best_alpha = alphas[0]
        best_score = float("inf")
        for alpha in alphas:
            model = Ridge(alpha=float(alpha), random_state=seed)
            model.fit(inner_train, inner_train_y[:, col_idx])
            pred = model.predict(inner_val)
            score = float(np.sqrt(np.mean((inner_val_y[:, col_idx] - pred) ** 2)))
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
        chosen.append(best_alpha)
        final = Ridge(alpha=best_alpha, random_state=seed)
        final.fit(x_train, y_train[:, col_idx])
        val_pred[:, col_idx] = final.predict(x_valid)
        test_pred[:, col_idx] = final.predict(x_test)

    return val_pred, test_pred, chosen


def run_ridge_per_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> ModelOutputs:
    vectorizer = build_tfidf_vectorizer(60000, 80000)
    all_text = pd.concat([train, test], axis=0)["full_text"].astype(str)
    feats = vectorizer.fit_transform(all_text)
    x_full = feats[: len(train)]
    x_test = feats[len(train) :]
    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    folds = sorted(train["fold"].unique())

    oof = np.zeros_like(y)
    test_pred = np.zeros((x_test.shape[0], len(TARGET_COLUMNS)), dtype=float)
    fold_scores = []
    chosen_alphas_per_fold = []
    alphas = [0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 16.0, 24.0]

    for fold in folds:
        trn_idx = train.index[train["fold"] != fold].to_numpy()
        val_idx = train.index[train["fold"] == fold].to_numpy()
        val_pred, fold_test, chosen = _ridge_per_target(
            x_full[trn_idx], y[trn_idx], x_full[val_idx], x_test, alphas, int(config["seed"]) + fold
        )
        oof[val_idx] = val_pred
        test_pred += fold_test / len(folds)
        fold_scores.append(_fold_record(y[val_idx], val_pred, int(fold)))
        chosen_alphas_per_fold.append({"fold": int(fold), "alphas": chosen})

    return ModelOutputs(
        name="ridge_tfidf_per_target",
        oof=oof,
        test_pred=test_pred,
        cv_mcrmse=mcrmse(y, oof),
        column_rmse=columnwise_rmse(y, oof),
        fold_scores=fold_scores,
        extras={"chosen_alphas_per_fold": chosen_alphas_per_fold, "alpha_grid": alphas},
    )


def run_ridge_fused(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> ModelOutputs:
    vectorizer = build_tfidf_vectorizer(60000, 80000)
    all_text = pd.concat([train, test], axis=0)["full_text"].astype(str)
    tfidf = vectorizer.fit_transform(all_text)
    train_tfidf = tfidf[: len(train)]
    test_tfidf = tfidf[len(train) :]

    train_stats = handcrafted_matrix(train).toarray()
    test_stats = handcrafted_matrix(test).toarray()
    scaled_train, scaled_others, _ = _scale_features(train_stats, test_stats)
    scaled_test = scaled_others[0]
    train_stats_sparse = sparse.csr_matrix(scaled_train)
    test_stats_sparse = sparse.csr_matrix(scaled_test)

    x_full = sparse.hstack([train_tfidf, train_stats_sparse]).tocsr()
    x_test = sparse.hstack([test_tfidf, test_stats_sparse]).tocsr()

    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    folds = sorted(train["fold"].unique())
    oof = np.zeros_like(y)
    test_pred = np.zeros((x_test.shape[0], len(TARGET_COLUMNS)), dtype=float)
    fold_scores = []
    chosen_alphas_per_fold = []
    alphas = [1.0, 2.0, 4.0, 6.0, 10.0, 16.0]

    for fold in folds:
        trn_idx = train.index[train["fold"] != fold].to_numpy()
        val_idx = train.index[train["fold"] == fold].to_numpy()
        val_pred, fold_test, chosen = _ridge_per_target(
            x_full[trn_idx], y[trn_idx], x_full[val_idx], x_test, alphas, int(config["seed"]) + fold
        )
        oof[val_idx] = val_pred
        test_pred += fold_test / len(folds)
        fold_scores.append(_fold_record(y[val_idx], val_pred, int(fold)))
        chosen_alphas_per_fold.append({"fold": int(fold), "alphas": chosen})

    return ModelOutputs(
        name="ridge_tfidf_fused",
        oof=oof,
        test_pred=test_pred,
        cv_mcrmse=mcrmse(y, oof),
        column_rmse=columnwise_rmse(y, oof),
        fold_scores=fold_scores,
        extras={
            "feature_size": int(x_full.shape[1]),
            "handcrafted_features": TEXT_FEATURE_COLUMNS,
            "chosen_alphas_per_fold": chosen_alphas_per_fold,
        },
    )


def run_lgbm_svd(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> ModelOutputs:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for run_lgbm_svd")

    vectorizer = build_tfidf_vectorizer(40000, 40000)
    all_text = pd.concat([train, test], axis=0)["full_text"].astype(str)
    tfidf = vectorizer.fit_transform(all_text)
    svd = TruncatedSVD(n_components=128, random_state=int(config["seed"]))
    svd_features = svd.fit_transform(tfidf)
    train_svd = svd_features[: len(train)]
    test_svd = svd_features[len(train) :]

    train_stats = handcrafted_matrix(train).toarray()
    test_stats = handcrafted_matrix(test).toarray()
    scaled_train, scaled_others, _ = _scale_features(train_stats, test_stats)
    scaled_test = scaled_others[0]

    x_train = np.hstack([train_svd, scaled_train])
    x_test = np.hstack([test_svd, scaled_test])

    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    folds = sorted(train["fold"].unique())
    oof = np.zeros_like(y)
    test_pred = np.zeros((x_test.shape[0], len(TARGET_COLUMNS)), dtype=float)
    fold_scores = []

    for fold in folds:
        trn_idx = train.index[train["fold"] != fold].to_numpy()
        val_idx = train.index[train["fold"] == fold].to_numpy()
        fold_pred = np.zeros((len(val_idx), len(TARGET_COLUMNS)), dtype=float)
        fold_test = np.zeros((len(test), len(TARGET_COLUMNS)), dtype=float)
        for col_idx, _name in enumerate(TARGET_COLUMNS):
            model = LGBMRegressor(
                n_estimators=600,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=int(config["seed"]) + col_idx,
                verbosity=-1,
            )
            model.fit(x_train[trn_idx], y[trn_idx, col_idx])
            fold_pred[:, col_idx] = model.predict(x_train[val_idx])
            fold_test[:, col_idx] = model.predict(x_test)
        oof[val_idx] = fold_pred
        test_pred += fold_test / len(folds)
        fold_scores.append(_fold_record(y[val_idx], fold_pred, int(fold)))

    return ModelOutputs(
        name="lgbm_svd_fused",
        oof=oof,
        test_pred=test_pred,
        cv_mcrmse=mcrmse(y, oof),
        column_rmse=columnwise_rmse(y, oof),
        fold_scores=fold_scores,
        extras={"svd_components": 128},
    )


def stacked_ensemble(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    components: list[ModelOutputs],
) -> ModelOutputs:
    """Per-target convex combination of the supplied components, fitted on OOF."""

    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    grid = np.linspace(0.0, 1.0, 21)
    weight_combos = []
    if len(components) == 2:
        weight_combos = [(w, 1 - w) for w in grid]
    else:
        for w0 in grid:
            for w1 in grid:
                if w0 + w1 <= 1 + 1e-9:
                    weight_combos.append((w0, w1, max(0.0, 1 - w0 - w1)))

    oof_pred = np.zeros_like(y)
    test_pred = np.zeros_like(components[0].test_pred)
    chosen_weights: dict[str, list[float]] = {}

    for col_idx, name in enumerate(TARGET_COLUMNS):
        best_score = float("inf")
        best_combo = weight_combos[0]
        for combo in weight_combos:
            stacked = sum(
                float(weight) * comp.oof[:, col_idx] for weight, comp in zip(combo, components)
            )
            score = float(np.sqrt(np.mean((y[:, col_idx] - stacked) ** 2)))
            if score < best_score:
                best_score = score
                best_combo = combo
        chosen_weights[name] = list(map(float, best_combo))
        oof_pred[:, col_idx] = sum(
            w * comp.oof[:, col_idx] for w, comp in zip(best_combo, components)
        )
        test_pred[:, col_idx] = sum(
            w * comp.test_pred[:, col_idx] for w, comp in zip(best_combo, components)
        )

    return ModelOutputs(
        name="stacked_ensemble",
        oof=oof_pred,
        test_pred=test_pred,
        cv_mcrmse=mcrmse(y, oof_pred),
        column_rmse=columnwise_rmse(y, oof_pred),
        fold_scores=[],
        extras={
            "components": [comp.name for comp in components],
            "weights_per_target": chosen_weights,
        },
    )


def error_analysis(train: pd.DataFrame, oof: np.ndarray) -> dict[str, Any]:
    y = train[TARGET_COLUMNS].to_numpy(dtype=float)
    residual = oof - y
    abs_res = np.abs(residual)
    word_count = train["full_text"].fillna("").str.split().map(len).to_numpy()
    quartiles = np.quantile(word_count, [0.25, 0.5, 0.75])
    bins = np.digitize(word_count, quartiles)
    bin_labels = ["Q1 short", "Q2 medium-short", "Q3 medium-long", "Q4 long"]

    bucket_stats = []
    for bucket_id in range(4):
        mask = bins == bucket_id
        if mask.sum() == 0:
            continue
        bucket_stats.append(
            {
                "bucket": bin_labels[bucket_id],
                "count": int(mask.sum()),
                "mcrmse": mcrmse(y[mask], oof[mask]),
                "column_rmse": columnwise_rmse(y[mask], oof[mask]),
                "mean_word_count": float(np.mean(word_count[mask])),
            }
        )

    score_bucket_stats = []
    average_targets = y.mean(axis=1)
    score_bins = np.digitize(average_targets, [2.5, 3.5])
    score_labels = ["low (<=2.5)", "medium (2.5-3.5)", "high (>3.5)"]
    for bucket_id in range(3):
        mask = score_bins == bucket_id
        if mask.sum() == 0:
            continue
        score_bucket_stats.append(
            {
                "bucket": score_labels[bucket_id],
                "count": int(mask.sum()),
                "mcrmse": mcrmse(y[mask], oof[mask]),
                "column_rmse": columnwise_rmse(y[mask], oof[mask]),
                "mean_average_target": float(np.mean(average_targets[mask])),
            }
        )

    overall_residual = {
        "mean": {name: float(residual[:, idx].mean()) for idx, name in enumerate(TARGET_COLUMNS)},
        "std": {name: float(residual[:, idx].std()) for idx, name in enumerate(TARGET_COLUMNS)},
        "mae": {name: float(abs_res[:, idx].mean()) for idx, name in enumerate(TARGET_COLUMNS)},
        "max_abs": {name: float(abs_res[:, idx].max()) for idx, name in enumerate(TARGET_COLUMNS)},
    }

    return {
        "overall": overall_residual,
        "length_buckets": bucket_stats,
        "score_buckets": score_bucket_stats,
    }


def output_payload(model: ModelOutputs) -> dict[str, Any]:
    payload = {
        "name": model.name,
        "cv_mcrmse": model.cv_mcrmse,
        "column_rmse": model.column_rmse,
        "fold_scores": model.fold_scores,
        "submission_path": model.submission_path,
    }
    if model.extras:
        payload["extras"] = model.extras
    return payload


def run_enhanced(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    artifact_dir = ensure_dir(config["output"]["artifacts_dir"])
    submission_dir = ensure_dir(config["output"]["submissions_dir"])
    model_dir = ensure_dir("experiments/models")

    train = load_train(config["data"]["train_path"])
    test = load_test(config["data"]["test_path"])
    sample = load_sample_submission(config["data"]["sample_submission_path"])

    if config.get("debug_rows"):
        train = train.head(int(config["debug_rows"])).copy()
        test = test.head(min(len(test), 16)).copy()
        sample = sample.head(len(test)).copy()

    train = make_folds(train, int(config["n_splits"]), int(config["seed"]))

    components: list[ModelOutputs] = []

    ridge_pt = run_ridge_per_target(train, test, config)
    components.append(ridge_pt)
    ridge_pt.submission_path = str(submission_dir / "submission_ridge_per_target.csv")
    save_submission(sample, ridge_pt.test_pred, ridge_pt.submission_path)

    ridge_fused = run_ridge_fused(train, test, config)
    components.append(ridge_fused)
    ridge_fused.submission_path = str(submission_dir / "submission_ridge_fused.csv")
    save_submission(sample, ridge_fused.test_pred, ridge_fused.submission_path)

    lgbm_svd = run_lgbm_svd(train, test, config)
    components.append(lgbm_svd)
    lgbm_svd.submission_path = str(submission_dir / "submission_lgbm_svd.csv")
    save_submission(sample, lgbm_svd.test_pred, lgbm_svd.submission_path)

    ensemble = stacked_ensemble(train, sample, components)
    ensemble.submission_path = str(submission_dir / "submission_stacked_ensemble.csv")
    save_submission(sample, ensemble.test_pred, ensemble.submission_path)

    payload = {
        "components": [output_payload(model) for model in components],
        "ensemble": output_payload(ensemble),
    }
    write_json(payload, artifact_dir / "enhanced_metrics.json")
    np.save(artifact_dir / "ensemble_oof.npy", ensemble.oof)
    np.save(artifact_dir / "ensemble_test.npy", ensemble.test_pred)

    err = error_analysis(train, ensemble.oof)
    write_json(err, artifact_dir / "error_analysis.json")

    return {
        "components": [output_payload(model) for model in components],
        "ensemble": output_payload(ensemble),
        "error_analysis": err,
    }
