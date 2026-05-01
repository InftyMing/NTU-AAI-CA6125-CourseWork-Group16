"""Data loading, validation, splitting, and audit helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from feedback_ell.constants import (
    ID_COLUMN,
    REQUIRED_TEST_COLUMNS,
    REQUIRED_TRAIN_COLUMNS,
    TARGET_COLUMNS,
    TEXT_COLUMN,
)
from feedback_ell.utils import write_json


def load_train(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_TRAIN_COLUMNS, "train")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    return df


def load_test(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_TEST_COLUMNS, "test")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    return df


def load_sample_submission(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, [ID_COLUMN, *TARGET_COLUMNS], "sample_submission")
    return df


def validate_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    text = result[TEXT_COLUMN].fillna("").astype(str)
    words = text.str.split()
    result["char_count"] = text.str.len()
    result["word_count"] = words.map(len)
    result["sentence_count"] = text.str.count(r"[.!?]+").clip(lower=1)
    result["paragraph_count"] = text.str.count(r"\n+").add(1)
    result["avg_word_len"] = text.map(
        lambda value: float(np.mean([len(w) for w in value.split()])) if value.split() else 0.0
    )
    result["comma_count"] = text.str.count(",")
    result["semicolon_count"] = text.str.count(";")
    result["quote_count"] = text.str.count('"')
    result["uppercase_ratio"] = text.map(
        lambda value: sum(ch.isupper() for ch in value) / max(1, sum(ch.isalpha() for ch in value))
    )
    result["digit_ratio"] = text.map(lambda value: sum(ch.isdigit() for ch in value) / max(1, len(value)))
    return result


def make_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """Create deterministic folds using rounded target sums as a weak stratification signal."""

    folded = df.copy()
    folded["fold"] = -1
    n_splits = max(2, min(int(n_splits), len(df)))
    y_signal = df[TARGET_COLUMNS].mean(axis=1).round(1).astype(str)
    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df, y_signal))
    except ValueError:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df))

    for fold, (_, valid_idx) in enumerate(splits):
        folded.loc[valid_idx, "fold"] = fold
    return folded


def build_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    train_stats = add_text_stats(train)
    test_stats = add_text_stats(test)
    target_summary = train[TARGET_COLUMNS].describe().round(4).to_dict()
    text_columns = ["char_count", "word_count", "sentence_count", "paragraph_count", "avg_word_len"]
    train_text_summary = train_stats[text_columns].describe().round(4).to_dict()
    test_text_summary = test_stats[text_columns].describe().round(4).to_dict()
    correlations = train[TARGET_COLUMNS].corr().round(4).to_dict()

    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target_columns": TARGET_COLUMNS,
        "missing_values_train": train.isna().sum().astype(int).to_dict(),
        "missing_values_test": test.isna().sum().astype(int).to_dict(),
        "target_summary": target_summary,
        "target_correlations": correlations,
        "train_text_summary": train_text_summary,
        "test_text_summary": test_text_summary,
        "sample_text_ids": train[ID_COLUMN].head(5).tolist(),
    }


def write_audit(train_path: str | Path, test_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    train = load_train(train_path)
    test = load_test(test_path)
    audit = build_audit(train, test)
    write_json(audit, output_path)
    return audit
