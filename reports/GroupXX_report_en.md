# Automated Essay Proficiency Scoring for English Language Learners

Course project for CA6125 LLM and RAG

Generated on: 2026-05-02 01:58

## 1. Introduction

This project studies the Feedback Prize - English Language Learning task. The goal is to
predict six analytic scores for argumentative essays written by English Language Learners:
cohesion, syntax, vocabulary, phraseology, grammar, and conventions. The task is naturally a
multi-output regression problem, because each essay receives a continuous proficiency profile
rather than a single class label.

The practical motivation is straightforward. Human feedback is expensive and slow, while ELL
students benefit from timely and fine-grained comments. A reliable model can help teachers
prioritize review effort and provide students with earlier formative signals.

## 2. Data

The official ELLIPSE corpus is used. The training set contains 6
essays and the test set contains 3 essays. Scores range from 1.0
to 5.0 in steps of 0.5. The submitted system does not use test labels or leaderboard feedback
for training.

Key data checks include missing-value validation, text-length statistics, target distributions,
and target correlation analysis. These checks are saved in `experiments/artifacts/data_audit.json`
and are also shown in the web demo.

## 3. Method

We built the system in three layers. First, a mean predictor establishes a lower bound. Second,
traditional NLP baselines use word and character TF-IDF features with Ridge or SVR regression,
plus a feature-based LightGBM model using essay length, punctuation, paragraph, and casing
statistics. Third, a DeBERTa-v3-base model is fine-tuned with a six-output regression head.

This design makes the comparison interpretable. TF-IDF baselines capture lexical overlap and
surface usage patterns, while DeBERTa captures broader contextual and syntactic information.
The final selection is based on cross-validation MCRMSE rather than a single public leaderboard
submission.

## 4. Evaluation

The official metric is mean column-wise root mean squared error:

MCRMSE = mean_j sqrt(mean_i (y_ij - p_ij)^2)

We report the overall MCRMSE, per-target RMSE, and fold-level scores. All experiments use the
same deterministic fold assignment where possible.

| Model | CV MCRMSE | Submission |
|---|---:|---|
| mean_baseline | 0.65349 | `experiments\submissions\submission_mean_baseline.csv` |
| lgbm_text_features | 0.77940 | `experiments\submissions\submission_lgbm_text_features.csv` |
| ridge_tfidf | 0.78558 | `experiments\submissions\submission_ridge_tfidf.csv` |
| svr_tfidf | 0.80016 | `experiments\submissions\submission_svr_tfidf.csv` |
| deberta_v3_base | not available | `None` |

The current best local model is `mean_baseline` with CV MCRMSE
0.65349.

## 5. Error Analysis

The six targets are related but not identical. Cohesion and phraseology tend to reflect discourse
flow and idiomatic expression, while grammar and conventions are more sensitive to local mistakes.
The analysis therefore checks errors per target, not only the average score. Long essays can be
challenging because they include more opportunities for inconsistency; very short essays can be
challenging because the model has less evidence.

## 6. Discussion

The main advantage of the Transformer model is that it uses context rather than isolated word
counts. This is important for ELL writing because two essays may use similar vocabulary while
differing in sentence control, transitions, and phrase naturalness. However, the model is not a
replacement for teachers. It can inherit corpus biases, overfit writing prompts, and provide a
score without explaining every judgment. For this reason, the demo presents scores as decision
support rather than final grades.

## 7. Conclusion

The project delivers a reproducible scoring pipeline, compares traditional and Transformer-based
approaches, and packages the result in a deployable web demo. The evidence supports using
pretrained language models for analytic essay scoring, while the baseline experiments keep the
conclusion grounded and measurable.

## References

- Kaggle. Feedback Prize - English Language Learning.
- He et al. DeBERTa: Decoding-enhanced BERT with disentangled attention.
- Scikit-learn documentation for TF-IDF, Ridge regression, and support vector regression.
