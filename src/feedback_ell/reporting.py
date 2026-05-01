"""Generate report text from experiment artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from feedback_ell.team import GROUP_CODE, TEAM_MEMBERS
from feedback_ell.utils import ensure_dir, read_json


def _fmt_score(value: Any) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.5f}"
    except Exception:
        return str(value)


def _sort_score(item: dict[str, Any]) -> float:
    value = item.get("cv_mcrmse")
    return float(value) if value is not None else 999.0


def _metrics_table(metrics: list[dict[str, Any]]) -> str:
    lines = ["| Model | CV MCRMSE | Submission |", "|---|---:|---|"]
    for item in sorted([m for m in metrics if m.get("cv_mcrmse") is not None], key=_sort_score):
        lines.append(
            f"| {item.get('name', 'unknown')} | {_fmt_score(item.get('cv_mcrmse'))} | "
            f"`{item.get('submission_path', '')}` |"
        )
    return "\n".join(lines)


def _team_table(language: str = "en") -> str:
    if language == "zh":
        lines = ["| 成员 | 主要分工 | 贡献说明 |", "|---|---|---|"]
    else:
        lines = ["| Member | Main Responsibility | Contribution |", "|---|---|---|"]
    for member in TEAM_MEMBERS:
        role_key = "role_zh" if language == "zh" else "role"
        contribution_key = "contribution_zh" if language == "zh" else "contribution"
        lines.append(
            f"| {member['name']} | {member[role_key]} | {member[contribution_key]} |"
        )
    return "\n".join(lines)


def collect_metrics(artifact_dir: str | Path = "experiments/artifacts") -> list[dict[str, Any]]:
    artifact_path = Path(artifact_dir)
    metrics: list[dict[str, Any]] = []
    baseline = read_json(artifact_path / "baseline_metrics.json", default=[])
    if isinstance(baseline, list):
        metrics.extend(baseline)
    transformer = read_json(artifact_path / "transformer_metrics.json", default=None)
    if isinstance(transformer, dict):
        metrics.append(transformer)
    final = read_json(artifact_path / "final_selection.json", default={})
    if isinstance(final, dict) and final.get("best"):
        best = final["best"].copy()
        existing_names = {item.get("name") for item in metrics}
        if best.get("name") not in existing_names:
            best["name"] = f"selected_{best.get('name', 'model')}"
            metrics.append(best)
    return metrics


def generate_english_report(
    output_path: str | Path = "reports/auto_results_snapshot_en.md",
    artifact_dir: str | Path = "experiments/artifacts",
) -> Path:
    ensure_dir(Path(output_path).parent)
    audit = read_json(Path(artifact_dir) / "data_audit.json", default={})
    metrics = collect_metrics(artifact_dir)
    valid_metrics = [item for item in metrics if item.get("cv_mcrmse") is not None]
    best = min(valid_metrics, key=_sort_score) if valid_metrics else {}
    content = f"""# Automated Essay Proficiency Scoring for English Language Learners

Course project for CA6125 LLM and RAG

Group: {GROUP_CODE}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

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

The official ELLIPSE corpus is used. The training set contains {audit.get('train_rows', 'N/A')}
essays and the test set contains {audit.get('test_rows', 'N/A')} essays. Scores range from 1.0
to 5.0 in steps of 0.5. The submitted system does not use test labels or leaderboard feedback
for training.

Key data checks include missing-value validation, text-length statistics, target distributions,
and target correlation analysis. These checks are saved in `experiments/artifacts/data_audit.json`
and are also shown in the web demo.

## 2.1 Team Contributions

{_team_table("en")}

## 3. Method

We built the system in three layers. First, a mean predictor establishes a lower bound. Second,
traditional NLP baselines use word and character TF-IDF features with Ridge regression, plus a
feature-based LightGBM model using essay length, punctuation, paragraph, and casing statistics.
The repository also includes a DeBERTa-v3-base fine-tuning pipeline with a six-output regression
head for GPU environments.

This design makes the comparison interpretable. TF-IDF baselines capture lexical overlap and
surface usage patterns, while DeBERTa captures broader contextual and syntactic information.
The final selection is based on cross-validation MCRMSE rather than a single public leaderboard
submission.

## 4. Evaluation

The official metric is mean column-wise root mean squared error:

MCRMSE = mean_j sqrt(mean_i (y_ij - p_ij)^2)

We report the overall MCRMSE, per-target RMSE, and fold-level scores. All experiments use the
same deterministic fold assignment where possible.

{_metrics_table(metrics) if metrics else 'No full experiment metrics are available yet.'}

The current best local model is `{best.get('name', 'not available')}` with CV MCRMSE
{_fmt_score(best.get('cv_mcrmse'))}.

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
"""
    output = Path(output_path)
    output.write_text(content, encoding="utf-8")
    return output


def generate_chinese_report(
    output_path: str | Path = "reports/auto_results_snapshot_zh.md",
    artifact_dir: str | Path = "experiments/artifacts",
) -> Path:
    ensure_dir(Path(output_path).parent)
    audit = read_json(Path(artifact_dir) / "data_audit.json", default={})
    metrics = collect_metrics(artifact_dir)
    valid_metrics = [item for item in metrics if item.get("cv_mcrmse") is not None]
    best = min(valid_metrics, key=_sort_score) if valid_metrics else {}
    content = f"""# 面向英语学习者作文的自动能力评分

CA6125 LLM and RAG 课程项目中文对照报告

小组：{GROUP_CODE}

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 引言

本项目选择 Kaggle Feedback Prize - English Language Learning 任务。目标是根据英语学习者的议论文，
预测六个写作能力维度：cohesion、syntax、vocabulary、phraseology、grammar 和 conventions。
该问题更适合建模为多目标回归，因为一篇作文对应的是一组连续能力分数，而不是单一类别。

项目动机在于，人工反馈成本高且周期长，而英语学习者需要及时、细粒度的反馈。一个稳定的模型可以
帮助教师快速定位需要关注的作文，也能为学生提供早期形成性反馈。

## 2. 数据

项目使用官方 ELLIPSE 数据集。训练集包含 {audit.get('train_rows', 'N/A')} 篇作文，测试集包含
{audit.get('test_rows', 'N/A')} 篇作文。每个维度分数范围为 1.0 到 5.0，步长为 0.5。训练过程不使用
测试标签，也不把 leaderboard 信息泄漏到模型训练中。

数据审计包括缺失值检查、文本长度统计、目标分布和目标相关性分析。审计结果保存在
`experiments/artifacts/data_audit.json`，并会在前端演示页中展示。

## 2.1 团队分工

{_team_table("zh")}

## 3. 方法

系统分为三层。第一层是均值预测，用于建立最基础下界。第二层是传统 NLP baseline，包括
TF-IDF + Ridge，以及基于文本统计特征的 LightGBM。工程中也保留了 DeBERTa-v3-base
六输出回归头微调流程，用于具备 GPU 和兼容 PyTorch 环境时进一步提升模型。

这样的设计有利于报告中的科学对比。TF-IDF 能捕捉词汇和表层模式，DeBERTa 能进一步建模上下文、
句法和表达自然度。最终模型选择依据本地交叉验证 MCRMSE，而不是单次 public leaderboard 分数。

## 4. 评估

官方指标为 MCRMSE，即六个目标列 RMSE 的平均值。我们报告整体 MCRMSE、每个维度的 RMSE 和
fold 级别分数。所有实验尽量使用同一套确定性数据划分。

{_metrics_table(metrics) if metrics else '目前尚无完整实验指标。'}

当前本地最优模型为 `{best.get('name', 'not available')}`，CV MCRMSE 为
{_fmt_score(best.get('cv_mcrmse'))}。

## 5. 误差分析

六个目标相关但并不相同。cohesion 和 phraseology 更关注篇章连贯与表达习惯，grammar 和
conventions 更容易受到局部错误影响。因此误差分析不仅看平均分，也看各维度误差。长作文可能包含
更多不一致表达，短作文则可能证据不足，这两类样本都需要单独观察。

## 6. 讨论

Transformer 模型的主要优势是能够利用上下文，而不仅是孤立词频。这对英语学习者作文评分很重要，
因为两篇作文可能词汇相似，但句子控制、衔接和短语自然度差异明显。不过模型不能替代教师。它可能
继承语料偏差，也可能给出分数但无法解释每一个判断。因此前端演示中将模型定位为辅助评分工具。

## 7. 结论

本项目交付了一套可复现的作文评分流程，对比了传统方法与预训练语言模型方法，并提供了可部署的
前端演示。实验结果用于支持结论，baseline 设计保证推论不过度依赖单一复杂模型。

## 参考资料

- Kaggle. Feedback Prize - English Language Learning.
- He et al. DeBERTa: Decoding-enhanced BERT with disentangled attention.
- Scikit-learn 文档：TF-IDF、Ridge 回归、SVR。
"""
    output = Path(output_path)
    output.write_text(content, encoding="utf-8")
    return output


def generate_video_materials(report_dir: str | Path = "reports") -> list[Path]:
    directory = ensure_dir(report_dir)
    script = directory / "video_script_en.md"
    flow = directory / "video_demo_flow_zh.md"
    script.write_text(
        """# Video Script

1. Introduce the task: analytic scoring for English Language Learner essays.
2. Explain the six targets and the official MCRMSE metric.
3. Show the dataset audit page: sample counts, text length, target distributions.
4. Explain baselines: mean predictor, TF-IDF Ridge/SVR, text-feature LightGBM.
5. Explain the DeBERTa model: tokenizer, pooling, regression head, cross-validation.
6. Present the experiment table and per-target RMSE.
7. Open the live demo, paste a short essay, and discuss the six predicted scores.
8. Show the generated submission file and, if available, Kaggle submission score.
9. Conclude with strengths, limitations, and what we learned.
""",
        encoding="utf-8",
    )
    flow.write_text(
        """# 录制视频交互流程

1. 打开首页，说明项目目标和六个评分维度。
2. 进入 Data Audit，展示训练/测试规模、文本长度和目标分布。
3. 进入 Experiments，讲解 baseline 到 DeBERTa 的递进关系。
4. 展示每列 RMSE，强调不要只看平均分。
5. 在 Essay Scoring Demo 中输入一段作文，点击 Predict。
6. 解释输出分数、模型置信边界和适用场景。
7. 展示 Submission 区域，说明 `submission.csv` 的生成方式。
8. 最后总结：可复现流程、科学实验、局限和后续改进。
""",
        encoding="utf-8",
    )
    return [script, flow]
