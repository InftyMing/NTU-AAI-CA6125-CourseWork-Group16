"""Feature builders for traditional NLP baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from feedback_ell.constants import TEXT_COLUMN
from feedback_ell.data import add_text_stats


TEXT_FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "sentence_count",
    "paragraph_count",
    "avg_word_len",
    "comma_count",
    "semicolon_count",
    "quote_count",
    "uppercase_ratio",
    "digit_ratio",
]


def build_tfidf_vectorizer(
    word_max_features: int = 60000,
    char_max_features: int = 80000,
) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=word_max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=char_max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def fit_transform_tfidf(vectorizer: FeatureUnion, train: pd.DataFrame):
    return vectorizer.fit_transform(train[TEXT_COLUMN].fillna("").astype(str))


def transform_tfidf(vectorizer: FeatureUnion, df: pd.DataFrame):
    return vectorizer.transform(df[TEXT_COLUMN].fillna("").astype(str))


def handcrafted_matrix(df: pd.DataFrame):
    stats = add_text_stats(df)
    arr = stats[TEXT_FEATURE_COLUMNS].replace([np.inf, -np.inf], 0).fillna(0).to_numpy(dtype=float)
    return sparse.csr_matrix(arr)


def combine_sparse(*matrices):
    return sparse.hstack(matrices).tocsr()
