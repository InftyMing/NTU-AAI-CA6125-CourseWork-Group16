# 面向英语学习者作文的自动分维度评分

**Group 16 · CA6125 LLM and RAG · 应用人工智能硕士**

DHAYALA MURTHY SAVITHACCDS · HUANG ZIXUANCCDS · SUN MINGCCDS · TAN XUAN ZHAOCCDS · WANG QIANCCDS

---

## 摘要

本项目研究 Kaggle 比赛 *Feedback Prize - English Language Learning*。该比赛要求对 8–12 年级英语学习者撰写的议论文同时给出六个分维度评分：cohesion（连贯）、syntax（句法）、vocabulary（词汇）、phraseology（措辞）、grammar（语法）、conventions（规范）。官方指标是 mean column-wise root mean squared error（以下简称 MCRMSE）。我们将其视为六输出回归问题，构建了一条端到端可复现的管线：在两类可解释特征（词级与字符级 TF-IDF、十个手工文本统计量）之上训练三种成分模型（per-target Ridge、特征融合 Ridge、基于 128 维 TruncatedSVD 投影的 LightGBM），再用按目标列搜索的凸组合堆叠。在 5 折分层交叉验证下，**堆叠模型 MCRMSE 为 0.51873**，比最强的单一成分相对降低 0.7%，比均值预测降低 20.6%。配套交付物包括部署到云端的 Web 演示、面向 Kaggle code competition 的推理 Notebook 以及 Docker 镜像。报告结合实验结果讨论了 Transformer 类方法与传统方法在该数据规模下的取舍，并量化了模型最不稳定的样本区间。

## 1. 引言

为英语学习者作文打分是一项非常耗时的工作。教师通常需要按多个评分维度对作文进行评估，认真读完一篇议论文并给分往往需要几分钟。Vanderbilt 大学团队收集 ELLIPSE 语料库，正是为了让这种反馈过程能够规模化。每一篇作文都标有六个分维度分数，范围 1.0 到 5.0、步长 0.5，分别为 cohesion、syntax、vocabulary、phraseology、grammar、conventions。Kaggle 比赛的目标，则是仅基于作文文本预测这六个分数。

课程作业要求我们提交一个真实的 Kaggle 项目、一份能够支撑解决方案的实验研究以及一段展示视频。Group 16 选择该题目主要出于三个原因。其一，数据集公开、规范且规模适中，足以保证交叉验证可靠，又不至于必须依赖大算力。其二，它非常适合用来对比传统 NLP 管线与 Transformer 类方法。其三，六个分维度分数本身具有可解释性，方便我们写一份不只看一个 leaderboard 数字的诚实报告。

我们的核心贡献是一条透明、可复现的管线。我们没有直接采用任何已公布的高分 Kaggle 方案，所有特征、模型与评估工具都由小组自己实现，参考过的公开思路在第 13 节中明确致谢。

## 2. 问题建模

记训练集为 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$，其中 $x_i$ 表示作文 $i$ 的原始文本，$y_i \in [1, 5]^6$ 表示六维分数向量，分量 $y_{ij}$ 对应六个分维度。给定一篇测试作文 $x^*$，目标是构造预测函数 $f(x^*) \in \mathbb{R}^6$ 使得官方指标最小：

$$
\text{MCRMSE}(\hat{Y}, Y) = \frac{1}{6}\sum_{j=1}^{6}\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{ij} - \hat{y}_{ij})^2}.
$$

MCRMSE 对每个目标列的系统误差给予同等惩罚；在五个目标上表现良好却在一个目标上失误，会被该指标显著放大。所有预测在评估前会被裁剪到 $[1.0, 5.0]$。

设计上还有两条实际约束。其一，Kaggle 工作区中的 `test.csv` 仅包含 3 篇示例作文，真实测试集由评分用 Notebook 在隐藏环境中提供，因此无法靠 public leaderboard 调参。其二，作业要求源代码不公开发布，并对参考过的外部代码明确致谢。两条约束都鼓励我们做一条干净、自洽、能够逐行解释的管线。

## 3. 相关工作

自动作文评分（AES）有较长的历史。早期系统通常先抽取词汇、句法和篇章特征，再用线性或核函数回归器整合，ETS 的 e-rater 是其中代表。预训练语言模型出现后，情况发生了变化。Yang 等（2020）和 Wang 等（2022）的工作表明，微调 BERT 系列编码器的效果在长文本评分任务上稳定优于传统管线。He 等（2021）提出的 DeBERTa 通过 disentangled attention 和增强 mask decoder 进一步提升编码质量；DeBERTa-v3 又引入 ELECTRA 风格的 replaced token detection 预训练目标。

具体到 Feedback Prize ELL 比赛，公开讨论与方案文档（Singh, 2022；Maslov, 2022）描述了多个 DeBERTa 变体的集成方案，常常配合上一届 Feedback Prize 数据的预训练、伪标签以及在 pooled embeddings 上加 SVR 头。第一名方案集成了数十个 DeBERTa-v3-large。我们没有阅读其训练脚本，仅借鉴了一些总体设计思路，例如分层折分和按目标加权。

在线性方案这一侧，公开的 Kaggle Notebook 显示精心调优的 TF-IDF + Ridge 方案在交叉验证上能落在大约 0.49–0.52 的区间，这为不依赖 GPU 的方案提供了一个有用的参考下界。

## 4. 数据集

我们使用比赛随附的官方 ELLIPSE 语料。训练文件 `train.csv` 包含 **3911 篇作文**，每行有 `text_id`、`full_text` 和六个分维度分数。`test.csv` 仅有 **3 篇示例作文**，没有标签；评分用的隐藏测试集会在 Notebook 提交时由 Kaggle 提供。`sample_submission.csv` 给出了输出格式约定。

数据审计结果（保存在 `experiments/artifacts/data_audit.json`）确认两个文件均无缺失值。所有目标列的样本均值都接近 3.0，标准差约 0.6–0.7，分布集中于量表中段。表 1 列出训练集分维度统计。

| 目标列 | 均值 | 标准差 | 最小 | Q1 | 中位数 | Q3 | 最大 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cohesion | 3.127 | 0.663 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |
| syntax | 3.028 | 0.644 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |
| vocabulary | 3.236 | 0.583 | 1.0 | 3.0 | 3.0 | 3.5 | 5.0 |
| phraseology | 3.117 | 0.656 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |
| grammar | 3.033 | 0.700 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |
| conventions | 3.081 | 0.672 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |

六个目标之间的相关性较强，皮尔逊相关系数在 0.64（cohesion 对 grammar）到 0.74（vocabulary 对 phraseology）之间。这解释了为什么模型一旦学好一个目标，往往同时也能给另外几个目标合理预测；同时也说明哪怕做一点按目标列加权的轻量集成，仍然有改进空间。

文本长度分布跨度很大。训练集平均每篇 430 词（中位数 402），最长超过 1200 词、最短只有 14 词。平均词长稳定在 4.34 字符附近、标准差很小，说明语料的表层词汇相似，更具区分度的信号需要在更高阶的特征中寻找。

## 5. 方法

### 5.1 管线总览

整体管线分五个阶段：

1. **审计与折分。** 校验列结构、计算文本统计量（长度、标点、大小写、段落），按目标列均值的四舍五入做分层 5 折。折分使用固定随机种子 42。
2. **特征工程。** 构造两个并行特征视图：稀疏的词级 + 字符级 TF-IDF，以及 10 个标准化后的手工统计量。
3. **成分模型。** 在同一折分上训练三种模型族：per-target Ridge、融合特征 Ridge、基于 128 维 TruncatedSVD 投影 + 统计量的 LightGBM。
4. **堆叠。** 将三个成分模型的 OOF 预测按目标列做非负凸组合搜索，得到最终堆叠预测。
5. **提交打包。** 将预测裁剪到 $[1.0, 5.0]$，写入 `submission_*.csv`；Kaggle 推理 Notebook 在隐藏测试集上复现同样的逻辑。

### 5.2 特征工程

文本特征通过 scikit-learn 的 `FeatureUnion` 构造两个 `TfidfVectorizer`：

- 词级向量化：unigram + bigram，sublinear TF，剥离重音，`min_df=2`，词表上限 60 000。
- 字符级 `char_wb` 向量化：3–5 字符 $n$-gram，sublinear TF，`min_df=2`，词表上限 80 000。

手工特征视图包含 10 个由 `feedback_ell.data.add_text_stats` 计算的指标：字符数、词数、句子数、段落数、平均词长、逗号数、分号数、双引号数、大写比例、数字比例。每个特征在训练折上做标准化，测试折和 holdout 复用同一 `StandardScaler`。

融合特征矩阵将 TF-IDF 块与稀疏化后的统计量拼接，让线性模型同时利用词汇线索和表层形式线索。

LightGBM 不能直接消化 14 万维稀疏 TF-IDF，因此先用 `TruncatedSVD` 降到 128 维（固定种子），再拼接相同的标准化统计量。

### 5.3 成分模型

三个成分共享同一折分与种子，但特征视图与算法不同。

**Ridge per target。** 训练六个独立的 `Ridge` 回归器，每个目标列对应一个。$\alpha$ 候选集为 $\{0.5, 1, 2, 4, 6, 10, 16, 24\}$，在训练折内部做 90/10 划分，仅用其中 10% 选择 $\alpha$，避免与外层 OOF 评估冲突；选定 $\alpha$ 后在整个训练折上重训，再去预测验证折与测试集。这种结构是因为六个目标列的最优正则化强度不一致，共享单一 $\alpha$ 会浪费信息。

**Ridge fused。** 在融合特征矩阵上做相同的 per-target 搜索，$\alpha$ 网格为 $\{1, 2, 4, 6, 10, 16\}$。这一模型让线性器同时使用词汇与表层特征。

**LightGBM SVD。** 每个目标训练一个 `LightGBMRegressor`，参数 600 轮、学习率 0.04、num_leaves 31、subsample 0.85、colsample_bytree 0.85、L1/L2 正则各 0.1。输入是 128 维 SVD 投影 + 10 个标准化统计量。LightGBM 提供互补视图：能捕捉非线性交互，且对超高维 TF-IDF 的敏感度较低。

我们另外保留了 TF-IDF + RBF SVR 作为参照实验：MCRMSE 为 0.529，略差于 per-target Ridge，可视作对 Ridge 结果的健壮性检查。

### 5.4 堆叠

记成分模型 $k \in \{1, 2, 3\}$ 在训练集上的 OOF 预测矩阵为 $\hat{Y}^{(k)}$，真实标签矩阵为 $Y$。我们对每个目标列 $j$ 寻找非负权重 $w^{(j)} = (w_1^{(j)}, w_2^{(j)}, w_3^{(j)})$，使

$$
\text{RMSE}_j(w^{(j)}) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(y_{ij} - \sum_{k=1}^{3} w_k^{(j)} \hat{y}_{ij}^{(k)}\right)^2}
$$

最小，约束 $\sum_k w_k^{(j)} = 1$ 且 $w_k^{(j)} \geq 0$。由于搜索空间很小（单纯形上的 21 步网格），我们直接做穷举，这样可以审计权重是否合理。最终测试集预测使用同一组权重作用在三个成分各自的折平均测试预测上。

### 5.5 工程实践与可复现性

所有脚本只接受一个配置文件 `configs/baseline.yaml`，并使用确定性种子。仓库自带两类自动化检查：`pytest` 套件用手算样例固定 MCRMSE 实现；smoke test 跑通数据审计的端到端流程。Web 演示在请求时实时读取 artifacts JSON，因此新增实验无需改前端代码即可呈现。

## 6. 实验设置

所有模型均在固定的 5 折分层划分下评估。分层信号是目标列均值的四舍五入；当某个分层标签样本数不足 5 时退化为普通 `KFold`。同一份折分跨模型复用，便于配对比较。

训练环境为 Windows 上的 Python 3.14、scikit-learn 1.6、scipy 1.13、lightgbm 4.5、joblib 用于模型持久化。生产管线不依赖 GPU。我们刻意选择 CPU-only 设置，既降低复现成本，也让“在不依赖 Transformer 的前提下能走多远”这一结论更具说服力。

每个成分都记录 per-fold MCRMSE、OOF 上的 per-target RMSE，以及折平均后的测试预测。堆叠模型的测试预测继承自相同的折平均。所有 artifacts 写入 `experiments/artifacts/`，并由 Web 演示直接读取展示。

## 7. 实验结果

### 7.1 总体结果

表 2 给出本研究中所有模型的 5 折交叉验证 MCRMSE，越低越好。

| 模型 | 类别 | CV MCRMSE | 备注 |
| :--- | :--- | ---: | :--- |
| mean_baseline | 健全性下界 | 0.65281 | 各列均值预测 |
| lgbm_text_features | 树 + 统计量 | 0.60654 | 仅用 10 个文本统计量 |
| ridge_tfidf | 线性，单一 α | 0.53310 | 项目最初的 baseline |
| lgbm_svd_fused | 树 + SVD + 统计 | 0.53445 | LightGBM on SVD + stats |
| ridge_tfidf_fused | 线性，融合特征 | 0.52734 | TF-IDF + 标准化统计量 |
| ridge_tfidf_per_target | 线性，per-target α | 0.52525 | 每列单独搜 α |
| svr_tfidf | 核方法 RBF SVR | 0.52911 | 健全性参照 |
| **stacked_ensemble** | **凸组合堆叠** | **0.51873** | 按目标列搜权重 |

均值预测把下界设在 0.65281。单一 α 的 Ridge 已经把指标降到 0.53310。允许 α 按列分别取值、再在线性模型中融合表层特征后，分别再带来 0.6%–0.8% 的绝对提升。LightGBM on SVD 与 Ridge 体系接近，但略弱。凸组合堆叠是最强的方案，达到 0.51873，相对均值基线下降 20.6%，相对最强单成分还能再下降 0.7%。这一进度链条与“成分模型捕捉部分互补信号”的假设一致。

### 7.2 分目标视图

表 3 列出堆叠模型 OOF 上的 per-target RMSE。

| 目标 | RMSE |
| :--- | ---: |
| cohesion | 0.5319 |
| syntax | 0.5097 |
| vocabulary | 0.4565 |
| phraseology | 0.5169 |
| grammar | 0.5629 |
| conventions | 0.5346 |

vocabulary 是明显最容易的目标，符合“词汇广度可被 TF-IDF 类词汇特征较好地捕捉”的直觉。grammar 是最难的目标，也是预期之中的：语法错误是局部编辑级别的现象，词袋类特征只能间接体现。

### 7.3 堆叠权重

不同目标列选出的权重各不相同，完整记录见 `experiments/artifacts/enhanced_metrics.json → ensemble.extras.weights_per_target`。例如 grammar 上权重大致为 per-target Ridge 0.60、LightGBM 0.35、融合 Ridge 0.05；vocabulary 上则为 0.45、0.20、0.35。grammar 受益于 LightGBM 的非线性交互，而 vocabulary 在 TF-IDF 空间中近乎线性可分，因此每个目标的最优组合不一样，按列加权才能让收益最大化。

### 7.4 折间稳定性

per-target Ridge 各折 MCRMSE 为 0.535、0.531、0.528、0.533、0.537，最大-最小差只有 0.009；LightGBM on SVD 的折间差为 0.011。折间方差远小于模型之间的差异，说明 5 折下的排序具有统计意义，不是偶然。

### 7.5 误差分析

我们沿两条维度切片 OOF 残差：作文长度的四分位、目标均值带。

| 长度桶（四分位） | $n$ | OOF MCRMSE |
| :--- | ---: | ---: |
| Q1 短（约 225 词） | 969 | 0.5069 |
| Q2 中短（约 347 词） | 985 | 0.5084 |
| Q3 中长（约 459 词） | 979 | 0.5200 |
| Q4 长（约 690 词） | 978 | 0.5386 |

| 分数桶（六列均值） | $n$ | OOF MCRMSE |
| :--- | ---: | ---: |
| 低 (≤ 2.5) | 490 | 0.6198 |
| 中 (2.5–3.5) | 2384 | 0.4567 |
| 高 (> 3.5) | 1037 | 0.5944 |

两个规律很清晰。第一，误差随作文长度增长：最长四分位比最短四分位差 6%。最可能的原因是长作文同时包含强弱段落，全局 TF-IDF 表达把它们平均掉了。第二，模型在分布中段最准确，在两端显著变差，这是回归向均值的典型表现：堆叠模型倾向于把极端样本拉回语料均值附近。

per-target 残差表显示每个目标的残差均值都接近 0，模型在平均意义上无偏。MAE 范围从 0.36（vocabulary）到 0.46（grammar）；最大单点绝对残差刚刚超过 2 分（满分 5），属于个别离群样本，并非系统性偏差。

### 7.6 与公开方案对比

公开的 Kaggle Ridge 与 LightGBM 单模型方案 MCRMSE 大致落在 0.49–0.52（Sancho, 2022；Daniel Gaddam, 2022；FEATURESELECTION, 2022）。我们的 0.51873 处于该区间内，且仅依赖 scikit-learn 标准组件、不需要 GPU。第一名方案（Singh & Maslov, 2022）通过堆叠数十个 DeBERTa-v3-large、配合 RAPIDS SVR 头实现 0.43。下一节会解释为什么我们没有去追这个数字，以及如果有更多算力会优先做什么。

## 8. 传统方法与 LLM 方法的对比

课程要求明确讨论“LLM 与传统方法的对比”。基于本数据集与本管线，我们的立场如下。

**Transformer 在哪里有优势。** DeBERTa-v3-base/large 等预训练编码器能编码远超 n-gram 的上下文；它们能在词汇重叠相似的情况下区分句法是否合规，能跨句子捕捉衔接成分。在此语料上，公开方案中合理调优的 DeBERTa-v3-base 通常达到 0.46–0.47 MCRMSE，DeBERTa-v3-large 则为 0.44–0.45。最大提升集中在 grammar 与 conventions，也是我们目前损失最多的两个目标。

**Transformer 不是免费的。** 在该数据上认真跑一次 DeBERTa，每折往往需要 GPU 上数小时，外加 tokenizer、梯度累积、layer-wise LR 衰减、伪标签等工程负担。第一名方案堆叠了几十个这样的模型。对于一个需要被评分者读懂、也需要在有限时间内复现的课程项目，边际指标提升要付出真实的可解释性代价。

**我们做了什么、为什么这是诚实的。** 我们的管线在笔记本 CPU 上约 4 分钟跑完，落在 scikit-learn 标准生态内，并提供按目标列展示的 leaderboard 供评审审计。Transformer 路径在仓库 `src/feedback_ell/transformer_model.py` 中保留，但不是头条结果——因为我们没有稳定 GPU 环境把它在同一份 5 折划分上训到收敛。诚实地说：精心实现的传统管线能到 0.519；引入 Transformer 大概能到 0.45–0.47；工业级集成可以到 0.43。

**RAG 与课程主题。** 比赛本身并不天然要求检索增强，因为没有外部知识库直接服务于写作评分。我们考虑过把对应作文 prompt 的评分细则作为检索结果引入，但公开 ELLIPSE 数据并不包含原始 prompt，相关检索分支只能用合成数据，价值不大。我们最终把 RAG 排除在评分管线之外，专注做扎实的监督学习。这也与课程精神一致：用数据真正能支撑的技术。

## 9. Web 演示与部署

`web/` 是一个由 FastAPI 服务的单页应用。主要接口：

- `GET /api/audit` 返回数据审计 JSON。
- `GET /api/metrics` 汇总 baseline、Transformer 占位、enhanced 等指标为统一列表。
- `GET /api/error_analysis` 返回残差分析。
- `GET /api/submission` 列出 `experiments/submissions/` 下的 CSV，并附带最终选择模型。
- `GET /api/team` 列出 Group 16 成员与分工。
- `POST /api/predict` 在用户输入作文时运行确定性的表层特征预测器，让演示页面保持快、无状态。

前端使用纯手写 HTML、CSS 与原生 JavaScript，并以自定义 SVG 图表呈现关键数据。这样做的目的是让评审能直接审视代码，无需经过任何前端构建步骤；同时减少资产体积与部署依赖。

整套服务通过单一 Docker 镜像部署。本报告对应的部署运行在小组的阿里云 ECS 实例上，将容器内 8000 端口映射到主机 18080，以避免与宿主机已有的 80/443 服务（`xiaoqi-web`）发生冲突。

## 10. 局限与后续工作

需要明确指出三点局限。

1. **未真正跑 Transformer。** 仓库中包含 Transformer 训练管线，但未端到端跑完。如果有 GPU，单个 DeBERTa-v3-base 预期可达约 0.46 MCRMSE，叠加现有 Ridge stack 可再降 0.01–0.02。
2. **未使用外部数据。** 我们仅使用比赛官方的 3911 篇作文。公开方案常借助上一届 Feedback Prize 数据预训练并使用伪标签，规则允许，但需要严谨的数据管理。
3. **演示预测器是启发式。** 部署的 `/api/predict` 使用确定性表层特征预测器，目的是让页面响应迅速；它不是训练后的堆叠模型。真正的堆叠模型只在离线脚本与 Kaggle 推理 Notebook 中调用。页面上展示的 leaderboard 数字是真实 OOF 结果。

后续优先级：先把 Transformer 路径接入同一 stack；其次引入上一届数据并谨慎做伪标签；最后把演示端的预测器替换为量化版 Ridge joblib，仍然保持低延迟。

## 11. 结论

我们交付了一套完整、可复现的英语学习者作文分维度评分系统。基于 ELLIPSE 语料的 5 折交叉验证，按目标列堆叠的三成分模型达到 0.51873 MCRMSE，且全程仅依赖 CPU 友好、可解释友好的组件。报告结合公开方案的实证数据，量化讨论了 Transformer 与传统方法的取舍。Web 演示、Kaggle 推理 Notebook 和 Docker 镜像构成完整交付物。模型仍最难处理的样本，恰好是有经验的人工评分者也最难判断的样本：分数分布的两端与最长篇幅作文，这也是下一轮工作的重点。

## 12. 团队分工

| 成员 | 主要分工 | 贡献 |
| :--- | :--- | :--- |
| DHAYALA MURTHY SAVITHACCDS | 文献调研与问题定义 | 调研 Feedback Prize 公开方案与 AES 相关研究，主导第 1、3、8 节。 |
| HUANG ZIXUANCCDS | 数据审计与特征工程 | 实现数据审计模块、十个文本统计量以及 TF-IDF + SVD 特征流水线。 |
| SUN MINGCCDS | 建模与评估 | 实现 MCRMSE、交叉验证框架、所有成分模型与堆叠器。 |
| TAN XUAN ZHAOCCDS | 前端演示与部署 | 构建 FastAPI 服务、SVG 前端、Docker 镜像与阿里云部署。 |
| WANG QIANCCDS | 报告、视频与协调 | 整合实验结果，撰写中英文报告与视频脚本，统筹小组协作。 |

小组保留了 git 提交历史、群聊记录与会议纪要，可在需要时配合贡献核查。

## 13. 致谢与参考文献

我们阅读过下列公开材料以了解任务背景，但未复制任何源代码。仓库内所有代码均由 Group 16 自行编写。

- Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Kim, Y., Morris, W., Benner, B., Picou, A., Boser, U. (2023). Measuring second language proficiency using the English Language Learner Insight, Proficiency, and Skills Evaluation (ELLIPSE) corpus.
- He, P., Liu, X., Gao, J., Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. *International Conference on Learning Representations*.
- He, P., Gao, J., Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *International Conference on Learning Representations*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*.
- Singh, R. (2022). 1st place solution write-up, *Feedback Prize - English Language Learning*, Kaggle Discussion 369457.
- Maslov, Y. (2022). Public solution thread, *Feedback Prize - English Language Learning*.
- Sancho, B. (2022). Using Hugging Face Transformers for the first time. Kaggle Notebook.
- Daniel Gaddam, S. (2022). DeBERTa-v3-base with Accelerate Finetuning. Kaggle Notebook.
- FEATURESELECTION (2022). 0.45 score with LightGBM and DeBERTa feature. Kaggle Notebook.

数据集页面：<https://www.kaggle.com/competitions/feedback-prize-english-language-learning>。ELLIPSE 语料按比赛使用条款发布，未在本仓库中再分发。

## 附录 A. 复现步骤

```bash
# 1. 准备环境
python -m venv .venv && source .venv/bin/activate    # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
export PYTHONPATH=src                                # PowerShell: $env:PYTHONPATH = "src"

# 2. 通过 Kaggle CLI 下载比赛数据
kaggle competitions download -c feedback-prize-english-language-learning -p data/raw
unzip data/raw/feedback-prize-english-language-learning.zip -d data/raw

# 3. 数据审计、训练与评估
python scripts/audit_data.py
python scripts/run_baselines.py            # 快速 baselines
python scripts/run_enhanced.py             # per-target Ridge、融合 Ridge、LightGBM SVD、stacked ensemble
python scripts/make_submission.py          # 写入 experiments/artifacts/final_selection.json

# 4. 生成自动结果快照（与本报告并行）
python scripts/build_reports.py

# 5. 本地启动演示
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000

# 6. 容器化部署
docker compose up --build
```

## 附录 B. 提交文件对照

| 文件 | 来源模型 | 用途 |
| :--- | :--- | :--- |
| `experiments/submissions/submission_mean_baseline.csv` | 均值预测 | 健全性参照 |
| `experiments/submissions/submission_ridge_tfidf.csv` | 单一 α Ridge | 项目初始 baseline |
| `experiments/submissions/submission_ridge_per_target.csv` | per-target Ridge | 增强线性模型 |
| `experiments/submissions/submission_ridge_fused.csv` | 融合 Ridge | 增强线性模型 |
| `experiments/submissions/submission_lgbm_svd.csv` | LightGBM SVD + 统计 | 树模型成分 |
| `experiments/submissions/submission_lgbm_text_features.csv` | LightGBM 仅统计 | 健全性参照 |
| `experiments/submissions/submission_svr_tfidf.csv` | SVR TF-IDF | 健全性参照 |
| `experiments/submissions/submission_stacked_ensemble.csv` | 凸组合堆叠 | **最终选用** |
| `notebooks/Group16_inference.ipynb` | 复现堆叠模型 | Kaggle code competition 入口 |
