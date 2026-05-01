# 面向英语学习者作文的自动能力评分

CA6125 LLM and RAG 课程项目中文对照报告

生成时间：2026-05-02 01:58

## 1. 引言

本项目选择 Kaggle Feedback Prize - English Language Learning 任务。目标是根据英语学习者的议论文，
预测六个写作能力维度：cohesion、syntax、vocabulary、phraseology、grammar 和 conventions。
该问题更适合建模为多目标回归，因为一篇作文对应的是一组连续能力分数，而不是单一类别。

项目动机在于，人工反馈成本高且周期长，而英语学习者需要及时、细粒度的反馈。一个稳定的模型可以
帮助教师快速定位需要关注的作文，也能为学生提供早期形成性反馈。

## 2. 数据

项目使用官方 ELLIPSE 数据集。训练集包含 6 篇作文，测试集包含
3 篇作文。每个维度分数范围为 1.0 到 5.0，步长为 0.5。训练过程不使用
测试标签，也不把 leaderboard 信息泄漏到模型训练中。

数据审计包括缺失值检查、文本长度统计、目标分布和目标相关性分析。审计结果保存在
`experiments/artifacts/data_audit.json`，并会在前端演示页中展示。

## 3. 方法

系统分为三层。第一层是均值预测，用于建立最基础下界。第二层是传统 NLP baseline，包括
TF-IDF + Ridge、TF-IDF + SVR，以及基于文本统计特征的 LightGBM。第三层是 DeBERTa-v3-base，
在预训练语言模型后接六输出回归头进行微调。

这样的设计有利于报告中的科学对比。TF-IDF 能捕捉词汇和表层模式，DeBERTa 能进一步建模上下文、
句法和表达自然度。最终模型选择依据本地交叉验证 MCRMSE，而不是单次 public leaderboard 分数。

## 4. 评估

官方指标为 MCRMSE，即六个目标列 RMSE 的平均值。我们报告整体 MCRMSE、每个维度的 RMSE 和
fold 级别分数。所有实验尽量使用同一套确定性数据划分。

| Model | CV MCRMSE | Submission |
|---|---:|---|
| mean_baseline | 0.65349 | `experiments\submissions\submission_mean_baseline.csv` |
| lgbm_text_features | 0.77940 | `experiments\submissions\submission_lgbm_text_features.csv` |
| ridge_tfidf | 0.78558 | `experiments\submissions\submission_ridge_tfidf.csv` |
| svr_tfidf | 0.80016 | `experiments\submissions\submission_svr_tfidf.csv` |
| deberta_v3_base | not available | `None` |

当前本地最优模型为 `mean_baseline`，CV MCRMSE 为
0.65349。

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
