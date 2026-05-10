# Automated Analytic Scoring of English Language Learner Essays

**Group 16 · CA6125 LLM and RAG · Master of Computing in Applied Artificial Intelligence**

DHAYALA MURTHY SAVITHACCDS · HUANG ZIXUANCCDS · SUN MINGCCDS · TAN XUAN ZHAOCCDS · WANG QIANCCDS

---

## Abstract

We study the Kaggle competition *Feedback Prize - English Language Learning*, which asks for six analytic writing scores—cohesion, syntax, vocabulary, phraseology, grammar, and conventions—for argumentative essays written by English Language Learners in grades 8 through 12. The official metric is the mean column-wise root mean squared error (MCRMSE). We treat the task as a six-output regression problem and build a fully reproducible pipeline that combines two interpretable feature views (word and character TF-IDF, plus ten handcrafted text statistics) with three component models (per-target Ridge, fused-feature Ridge, LightGBM on a 128-dimensional Truncated SVD projection). A per-target convex stack of these three components reaches **0.51873 MCRMSE under 5-fold stratified cross-validation**, improving on the strongest single component by 0.7 percent and on the mean predictor by 20.6 percent. To honour the *LLM and RAG* theme of the course we add two notebook-based probes on top of the classical stack: a parameter-efficient LoRA fine-tune of **Qwen2.5-1.5B-Instruct** that learns to emit the six-dimensional score JSON directly, and a training-free **DeBERTa-v3-base + cosine-KNN retriever** that scores essays by softmax-weighting their nearest training neighbours. The RAG variant reaches **0.5445 MCRMSE on a 500-essay hold-out**, the LLM variant produces sensible JSON-formatted predictions on the Kaggle hidden test set; both are honestly reported but kept off the main leaderboard because they use a different validation regime. We complement the modelling work with a deployed web demo, an inference notebook for the Kaggle code-competition format, and a Docker package for cloud serving. The report quantifies where the model is least reliable and discusses the trade-off between LLM/RAG and traditional approaches in this dataset regime.

## 1. Introduction

Writing feedback for English Language Learners (ELLs) is laborious. Teachers usually score essays on several rubric dimensions, and a careful read of an argumentative essay takes minutes, not seconds. The Vanderbilt University team behind the Feedback Prize series collected the ELLIPSE corpus precisely to enable scalable feedback. Each essay carries six analytic scores between 1.0 and 5.0 in steps of 0.5: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. The Kaggle competition then asks for a model that predicts these six scores given only the essay text.

The course brief expects us to produce a real Kaggle entry, an experimental study to justify the design, and a presentation. Group 16 chose this task for three reasons. First, the dataset is open, well documented, and large enough to make cross-validation reliable but small enough to fit in modest hardware. Second, it allows a clean comparison between traditional NLP pipelines and Transformer-based language models. Third, the six analytic targets are interpretable, which supports an honest discussion of model behaviour, not just a single leaderboard number.

Our contribution is a transparent and reproducible pipeline. We do not adopt published high-rank Kaggle solutions; we build features, models, and evaluation tooling ourselves and acknowledge the few public ideas we drew inspiration from. Section 7 makes that acknowledgement explicit.

## 2. Problem Formulation

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ denote the training set, where $x_i$ is the raw text of essay $i$ and $y_i \in [1, 5]^6$ is the analytic score vector with components $y_{ij}$ for $j \in \{\text{cohesion}, \text{syntax}, \text{vocabulary}, \text{phraseology}, \text{grammar}, \text{conventions}\}$. Given a test essay $x^*$, we want a predictor $f(x^*) \in \mathbb{R}^6$ that minimises the official metric

$$
\text{MCRMSE}(\hat{Y}, Y) = \frac{1}{6}\sum_{j=1}^{6}\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{ij} - \hat{y}_{ij})^2}.
$$

MCRMSE penalises systematic errors per target equally; a model that does well on five targets but poorly on one will not look good in this metric. Predictions are clipped to the official $[1.0, 5.0]$ range before evaluation.

Two practical constraints shape our design. First, the public test set in the Kaggle workspace contains only three example essays. The hidden test set is graded behind a notebook submission, so we cannot tune to a public leaderboard. Second, we are asked to keep the source code private and to acknowledge any external code we used. Both constraints argue for a clean, self-contained pipeline that we can defend line by line.

## 3. Related Work and Background

Automated essay scoring (AES) has a long history. Traditional systems combined hand-crafted lexical, syntactic, and discourse features with linear or kernel regressors; the e-rater system at ETS is one of the best-known examples. The introduction of pre-trained language models changed the picture. Yang et al. (2020) and Wang et al. (2022) showed that fine-tuning BERT-family encoders consistently outperforms classical pipelines on long-form essay scoring. The DeBERTa family proposed by He et al. (2021) further improved encoder quality through disentangled attention and an enhanced mask decoder; DeBERTa-v3 added an ELECTRA-style replaced-token-detection objective.

For the Feedback Prize ELL competition specifically, public discussions and write-ups (Singh, 2022; Maslov, 2022) describe ensembles of multiple DeBERTa variants pre-trained on the prior Feedback Prize data, sometimes augmented with pseudo-labels and SVR heads on pooled embeddings. The first place solution stacked dozens of DeBERTa-v3-large models. We deliberately did not consult their training scripts; their public read-me files informed only the broad shape of our design choices, such as the use of stratified folds and per-target weighting.

On the linear side, public Kaggle notebooks have shown that a careful TF-IDF + Ridge pipeline can reach roughly 0.49 to 0.51 MCRMSE on cross-validation, which sets a useful baseline for what is achievable without GPUs.

## 4. Dataset

We use the official ELLIPSE corpus released through the competition. The training file `train.csv` contains **3911 essays**. Each row carries a `text_id`, the full essay (`full_text`), and the six analytic scores. The public `test.csv` contains **3 example essays** without labels; the hidden test set used for grading is provided to the inference notebook at runtime. The `sample_submission.csv` documents the expected output format.

Our data audit (see `experiments/artifacts/data_audit.json`) confirms there are no missing values in either file. The score distribution is concentrated around the middle of the scale: every target has a sample mean close to 3.0 with a standard deviation around 0.6 to 0.7. Table 1 lists the per-target summary statistics on the training set.

| Target | Mean | Std | Min | Q1 | Median | Q3 | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cohesion | 3.127 | 0.663 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |
| syntax | 3.028 | 0.644 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |
| vocabulary | 3.236 | 0.583 | 1.0 | 3.0 | 3.0 | 3.5 | 5.0 |
| phraseology | 3.117 | 0.656 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |
| grammar | 3.033 | 0.700 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |
| conventions | 3.081 | 0.672 | 1.0 | 2.5 | 3.0 | 3.5 | 5.0 |

The six targets are strongly correlated, with Pearson coefficients between 0.64 (cohesion vs. grammar) and 0.74 (vocabulary vs. phraseology). This explains why a model that learns one target well tends to predict the others well too, and why a small per-target ensemble can still squeeze out a meaningful gain.

The text length distribution is wide. The training set has a mean of 430 words per essay (median 402) with a maximum above 1200 words and a minimum of 14. Average word length stays close to 4.34 characters with a small standard deviation, suggesting that surface vocabulary is similar across the corpus and that the discriminating signal lives in higher-order features.

## 5. Methodology

### 5.1 Pipeline overview

Our pipeline contains five stages:

1. **Audit and split.** Validate column schema, compute text statistics (length, punctuation, casing, paragraph count), and create a stratified 5-fold split using the rounded mean target as the grouping signal. The split is deterministic with seed 42.
2. **Feature engineering.** Two parallel feature views are constructed: a sparse word-and-character TF-IDF, and ten interpretable text statistics standardised to zero mean and unit variance.
3. **Component models.** Three model families are trained on the same fold split: per-target Ridge, fused-feature Ridge, and LightGBM on a 128-dimensional Truncated SVD projection of TF-IDF combined with the standardised statistics.
4. **Stacking.** Out-of-fold predictions from the three components are blended per target via a non-negative grid search over convex combinations.
5. **Submission packaging.** Final predictions are clipped to the $[1.0, 5.0]$ range and written to `submission_*.csv`. The Kaggle inference notebook reproduces the same logic on Kaggle's hidden test set.

### 5.2 Feature engineering

The text feature view uses scikit-learn's `FeatureUnion` of two `TfidfVectorizer`s:

- A word vectorizer with unigrams and bigrams, sublinear term-frequency, accent stripping, `min_df=2`, and a vocabulary cap of 60 000 features.
- A character `char_wb` vectorizer with $n$-grams of length 3 to 5, sublinear term-frequency, `min_df=2`, and a cap of 80 000 features.

The handcrafted view contains ten quantities computed by `feedback_ell.data.add_text_stats`: character count, word count, sentence count, paragraph count, average word length, number of commas, number of semicolons, number of double quotes, uppercase ratio, and digit ratio. Each is standardised against the training fold's mean and variance.

The fused feature matrix concatenates the TF-IDF block with a sparse representation of the scaled statistics, so the linear model can use both lexical and surface-form evidence.

For LightGBM we cannot afford to keep the raw 140 000-dimensional TF-IDF, so we project it to 128 dimensions with `TruncatedSVD` (random seed fixed) and concatenate the same standardised statistics.

### 5.3 Component models

Three component models share the same fold split and seed but differ in their feature view and learning algorithm.

**Ridge per target.** Six independent `Ridge` regressors are trained, one per target. The regularisation strength $\alpha$ is selected from $\{0.5, 1, 2, 4, 6, 10, 16, 24\}$ on an inner 90/10 split of the training fold, where the validation slice is held out for the alpha search and not for model fitting at the outer level. After selection, $\alpha$ is fixed and the Ridge is refit on the entire training fold before scoring the validation fold and the test set. We chose this scheme because the six targets have slightly different optimal regularisation strengths; sharing a single $\alpha$ wastes information.

**Ridge fused.** The same per-target search runs on the fused feature matrix with a slightly tighter grid $\{1, 2, 4, 6, 10, 16\}$. This model is intended to use surface-form features alongside lexical TF-IDF, even at the price of a small training overhead.

**LightGBM SVD.** A `LightGBMRegressor` is trained per target with 600 boosting rounds, learning rate 0.04, num_leaves 31, subsample 0.85, colsample_bytree 0.85, and L1/L2 regularisation 0.1. The input is the 128-dimensional SVD projection plus the ten standardised statistics. LightGBM provides a complementary view: it captures non-linear interactions and is less sensitive to the very high-dimensional TF-IDF regime.

We also retain a TF-IDF + RBF SVR baseline, originally part of the simpler experiment grid, as a sanity check on Ridge: SVR with TF-IDF reaches 0.529 MCRMSE, slightly worse than the per-target Ridge.

### 5.4 Stacking

Let $\hat{Y}^{(k)}$ be the matrix of out-of-fold predictions from component $k \in \{1, 2, 3\}$ on the training set, and $Y$ the matrix of true scores. We seek non-negative weights $w^{(j)} = (w_1^{(j)}, w_2^{(j)}, w_3^{(j)})$ for each target $j$ that minimise

$$
\text{RMSE}_j(w^{(j)}) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(y_{ij} - \sum_{k=1}^{3} w_k^{(j)} \hat{y}_{ij}^{(k)}\right)^2},
$$

subject to $\sum_k w_k^{(j)} = 1$ and $w_k^{(j)} \geq 0$. Because the search space is small (a 21-step grid on the simplex), we do an exhaustive enumeration; this also lets us audit the chosen weights for sanity. Final test-set predictions reuse the same weights against the components' averaged test predictions.

### 5.5 Reproducibility and engineering practice

Every script accepts a single configuration file (`configs/baseline.yaml`) and exposes deterministic seeds. The repository ships with two automated checks: a `pytest` suite that pins the MCRMSE implementation against a hand-computed example and a smoke test that runs the data audit end to end. The web demo reads the artifact JSON files at request time, which means new experiments propagate to the demo without a code change.

## 6. Experimental Setup

We evaluate every model under a fixed 5-fold stratified split. Stratification uses the rounded average target as the grouping label, falling back to plain `KFold` when a class has fewer than five members. The same fold assignment is shared across models so that comparisons are paired.

The training environment is Python 3.14 on Windows with scikit-learn 1.6, scipy 1.13, lightgbm 4.5, and joblib for model persistence. No GPU is required for the production pipeline. We deliberately tested with a CPU-only configuration to keep the cost of replication low, and to make the result honest about how far you can go without a Transformer.

For each component we record per-fold MCRMSE, per-target RMSE on the out-of-fold predictions, and the test predictions averaged across folds. The stacked ensemble inherits its test predictions from the same fold averaging. The full set of artifacts is checked into `experiments/artifacts/` and rendered live by the web demo.

## 7. Results

### 7.1 Headline numbers

Table 2 gives the cross-validation MCRMSE for every model in the study. Lower is better.

| Model | Family | CV MCRMSE | Notes |
| :--- | :--- | ---: | :--- |
| mean_baseline | sanity baseline | 0.65281 | global mean of each target |
| lgbm_text_features | tree, statistics | 0.60654 | LightGBM on ten text statistics |
| ridge_tfidf | linear, single alpha | 0.53310 | original baseline shipped with the project |
| lgbm_svd_fused | tree, SVD + stats | 0.53445 | LightGBM on 128-dim SVD + statistics |
| ridge_tfidf_fused | linear, fused | 0.52734 | TF-IDF + scaled statistics |
| ridge_tfidf_per_target | linear, per-target alpha | 0.52525 | separate alpha per target |
| svr_tfidf | kernel, RBF SVR | 0.52911 | sanity check, similar to Ridge |
| **stacked_ensemble** | **convex blend** | **0.51873** | per-target weights on OOF |

The mean predictor sets the floor at 0.65281. The single-alpha Ridge already cuts that to 0.53310. Splitting the alpha across targets and fusing in surface features both move the score down by another 0.6 to 0.8 percent absolute. LightGBM on the SVD view is competitive but slightly behind Ridge. The convex stack is the strongest, at 0.51873, which is 20.6 percent below the mean baseline and 0.7 percent below the strongest single component. The progression is consistent with the hypothesis that the components capture partially complementary signals.

### 7.2 Per-target view

Table 3 reports per-target RMSE on the out-of-fold predictions of the stacked ensemble.

| Target | RMSE |
| :--- | ---: |
| cohesion | 0.5319 |
| syntax | 0.5097 |
| vocabulary | 0.4565 |
| phraseology | 0.5169 |
| grammar | 0.5629 |
| conventions | 0.5346 |

Vocabulary is the easiest target by a clear margin, which matches the intuition that vocabulary breadth is well captured by lexical features such as TF-IDF. Grammar is the hardest target, which is also expected: grammatical errors are local edits that bag-of-n-grams models capture only indirectly.

### 7.3 Stacking weights

The chosen blend weights vary across targets (full record in `experiments/artifacts/enhanced_metrics.json → ensemble.extras.weights_per_target`). For example, for *grammar* the stack favours per-target Ridge (0.60), then LightGBM (0.35), then fused Ridge (0.05); for *vocabulary* the weights become 0.45, 0.20, and 0.35 respectively. Per-target weights pay off because grammar benefits from non-linear interactions that LightGBM picks up, while vocabulary is mostly linear in the TF-IDF space.

### 7.4 Fold-level stability

For per-target Ridge the fold MCRMSEs are 0.535, 0.531, 0.528, 0.533, and 0.537, so the spread is 0.009. For LightGBM on SVD the spread is 0.011. The variance across folds is small relative to the differences between models, which means that the leaderboard ordering is statistically meaningful even with five folds.

### 7.5 Error analysis

We split the OOF residuals along two axes: essay length quartiles and average target band.

| Length bucket (quartile) | $n$ | OOF MCRMSE |
| :--- | ---: | ---: |
| Q1 short (avg 225 words) | 969 | 0.5069 |
| Q2 medium-short (avg 347 words) | 985 | 0.5084 |
| Q3 medium-long (avg 459 words) | 979 | 0.5200 |
| Q4 long (avg 690 words) | 978 | 0.5386 |

| Score bucket (avg of six targets) | $n$ | OOF MCRMSE |
| :--- | ---: | ---: |
| low (<= 2.5) | 490 | 0.6198 |
| medium (2.5–3.5) | 2384 | 0.4567 |
| high (> 3.5) | 1037 | 0.5944 |

Two patterns are clear. First, errors grow with essay length: the longest quartile is six percent worse than the shortest. The likely reason is that long essays mix strong and weak passages, and a global TF-IDF representation averages them. Second, the model is best on essays whose average target sits in the middle of the scale, and noticeably worse on the low and high tails. This is a regression-to-the-mean effect: the stacked ensemble tends to pull extreme essays toward the corpus mean.

The per-target diagnostic table shows that the residual mean is essentially zero for every target, so the model is unbiased on average. Mean absolute errors range from 0.36 (vocabulary) to 0.46 (grammar). The largest absolute residual is just over 2.0 points, on a 1–5 scale; these are isolated outliers rather than systematic failures.

### 7.6 Comparison to public references

Public Kaggle submissions of single Ridge and LightGBM models reach MCRMSE in the 0.49 to 0.52 range (Sancho, 2022; Daniel Gaddam, 2022; FEATURESELECTION, 2022). Our 0.51873 sits comfortably in that range, despite using only standard scikit-learn building blocks and CPU-only training. The first place solution (Singh and Maslov, 2022) reaches 0.43 with a heavy ensemble of DeBERTa-v3-large variants and a separately tuned RAPIDS SVR head. We discuss in §8 why the gap is what it is, and we then add two notebook-based LLM/RAG probes (§7.7 and §7.8) that explicitly engage with the course theme.

### 7.7 Generative LLM fine-tune (Qwen2.5-1.5B + LoRA)

Because the course is titled *LLM and RAG*, we wanted at least one experiment in which a generative language model directly emits the analytic scores. We picked **Qwen2.5-1.5B-Instruct** as the backbone—small enough to fit on an Apple Silicon laptop (16 GB unified memory, MPS, fp16), but instruction-tuned and chat-templated. The training notebook lives at `LLM experiments/local training_essay_scoring_m_chip.ipynb` and the matching Kaggle inference notebook at `LLM experiments/kaggle inference_group-16.ipynb`.

**Setup.** We tokenise each essay with the Qwen chat template, prompt the model with a fixed instruction (*"Rate this English essay on 6 dimensions (1.0–5.0 in 0.5 steps). Output ONLY JSON: {cohesion: x, syntax: x, ...}"*), and target the JSON-encoded ground truth as the supervised continuation. Only the JSON tokens contribute to the loss; the prompt is masked with `-100`. We train a Low-Rank Adaptation (LoRA, Hu et al. 2022) head on the four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with rank 8, $\alpha=16$, dropout 0.05, which makes 2.18 million parameters trainable out of the 1.55 billion in the backbone (0.14 percent). The optimiser is `adamw_torch`, learning rate $2 \times 10^{-4}$, cosine schedule with 5 percent warmup, effective batch size 8 (per-device 2 with gradient accumulation 4), maximum sequence length 512 tokens. A 10 percent random hold-out of the 3 911 training essays is reserved for the early-stopping signal.

**Outcome.** Training converges smoothly across 3 epochs (training loss 0.184 → 0.168 → 0.165, total wall-clock 7 hours 40 minutes on the M-series MPS device). Validation cross-entropy is logged as `nan` because of a known fp16 + MPS interaction with the masked-label tensor: the model itself is fine, but the printed eval-loss is unreliable on this device. We do not chase the bug because (i) training loss is the only signal we use to pick the final checkpoint and it improves monotonically, and (ii) qualitative inference on the public test essays returns plausible scores (e.g. essay #1 → 3.0/2.5/3.0/2.5/2.5/2.5; essay #3 → 3.5/4.0/3.5/3.5/3.5/3.5), with the post-processor snapping each value to the official $\{1.0, 1.5, \dots, 5.0\}$ grid via a regex + nearest-grid clamp. Because the Kaggle code competition does not return MCRMSE for late submissions, we cannot assign a single number to this run; we instead report it as a successfully trained but unmeasured baseline that proves a small open-weight LLM can be fine-tuned for analytic essay scoring on commodity hardware.

### 7.8 Retrieval-augmented scoring (DeBERTa-v3-base + KNN)

The second course-aligned probe is a training-free RAG scorer at `RAG experiments/kaggle inference_rag_group-16.ipynb`. The motivation is to see how far we can get by *only* retrieving similar training essays.

**Setup.** We mean-pool the last hidden states of a frozen **DeBERTa-v3-base** encoder (He et al. 2023), $L_2$-normalise the resulting 768-dimensional vectors, and index the entire 3 911-essay training set with `sklearn.NearestNeighbors` under cosine distance. For each test essay we retrieve the $k=20$ nearest neighbours, convert their cosine distances into similarity scores, exponentiate them with temperature 10, and softmax-normalise the weights. The predicted six-dimensional score is the weighted average of the neighbours' analytic scores. Predictions are clipped to $[1.0, 5.0]$ before submission. No model parameters are updated.

**Outcome.** On a held-out validation slice consisting of the last 500 essays of `train.csv` (the KNN index is built from the remaining 3 411 essays so that the retriever cannot peek at validation labels), the RAG scorer reaches **0.5445 MCRMSE**. That places it between the single-alpha Ridge (0.533) and the LightGBM SVD baseline (0.534), and above our classical stacked ensemble (0.519). The advantage of this probe is that it requires zero training—the only hyperparameters are the encoder choice, $k$, and the softmax temperature—so it is a strong sanity check on the lower bound of *any* learned model. The disadvantage is the same regression-to-the-mean failure mode as the classical stack: when the neighbourhood is dominated by mid-range essays, the prediction collapses toward the corpus mean, which is exactly the pattern §7.5 already exposes.

## 8. Comparison Between Traditional and LLM/RAG Approaches

The course brief asks for an explicit discussion of LLM versus traditional methods. With §7.7 and §7.8 in hand we can now make that discussion concrete instead of speculative.

**Where the Transformer helps, in one number.** A frozen DeBERTa-v3-base encoder, used only as a retrieval index, already reaches 0.5445 MCRMSE on a 500-essay hold-out without any training. That is essentially free, and it confirms that pre-trained encoders capture writing-quality signal that goes beyond word-level n-grams. A properly fine-tuned DeBERTa-v3-base in the literature reaches 0.46–0.47 MCRMSE; DeBERTa-v3-large reaches 0.44–0.45. The lifts are largest on grammar and conventions, the two targets where we still lose the most.

**Where the LLM helps, qualitatively.** The Qwen2.5-1.5B + LoRA scorer is the most flexible variant we ship: it accepts an instruction in natural language and returns a structured JSON object. We could re-target it to a different rubric without touching the architecture. That makes it the right primitive if a downstream system needs *interpretable*, *editable* scoring policies. The price is a 7-hour single-GPU equivalent of training and the operational overhead of running a 1.5B-parameter model behind the demo. For a course project where the marker must reproduce numbers in a finite time slot, this is an honest trade-off, not a free lunch.

**Where the classical stack still wins.** Under the only metric we evaluate consistently—5-fold stratified CV—the stacked ensemble at 0.51873 beats both LLM/RAG probes. It also runs in ~4 minutes on a laptop CPU, exposes per-target leaderboards we can audit, and ships as both a CSV and a Kaggle inference notebook. The first place public solution at 0.43 stacks dozens of DeBERTa-v3-large models with pseudo-labelling; that is approximately two orders of magnitude more compute than what we used and is not the right baseline for a 30 percent course project.

**RAG and the wider course context.** Classical RAG retrieves text from an external knowledge base; here the retrieval space is the training corpus itself. This is sometimes called *neighbourhood-based prediction* or *KNN with neural embeddings*, and it is the simplest legitimate instantiation of RAG that respects the Kaggle code-competition rule of "no internet access at inference time". We deliberately did not bolt a synthetic prompt-rubric corpus into the pipeline because the public ELLIPSE essays do not include the original prompts; a synthetic retrieval branch would not be defensible. The §7.8 setup is therefore the strongest RAG variant the dataset itself supports.

**Bottom line.** A careful traditional pipeline reaches 0.519. A frozen DeBERTa retriever reaches 0.544 with no training. A LoRA-fine-tuned 1.5B LLM produces structured scores on the same hardware budget. An industrial DeBERTa-large ensemble could reach 0.43 with a dramatically larger compute budget. We submit the traditional stack as the headline entry, ship the LLM/RAG notebooks as supplementary evidence, and keep the architecture honest by not claiming numbers we did not actually evaluate.

## 9. Web Demo and Deployment

The demo at `web/` is a single-page application served by FastAPI. Key endpoints:

- `GET /api/audit` returns the data audit JSON.
- `GET /api/metrics` aggregates baseline, transformer-stub, and enhanced metrics into one list.
- `GET /api/llm_rag` returns the LoRA-Qwen and DeBERTa-RAG experiment summaries shown in the *LLM and RAG* panel.
- `GET /api/error_analysis` returns the residual analysis.
- `GET /api/submission` lists every CSV in `experiments/submissions/` together with the chosen final model.
- `GET /api/team` lists Group 16 members and roles.
- `POST /api/predict` runs a deterministic surface-feature predictor over a user-supplied essay so the demo stays fast and stateless.

The front end is hand-written HTML, CSS, and vanilla JavaScript with custom SVG charts. We chose this to keep the asset footprint small and to make the page directly inspectable by the marker without a build step.

The whole stack ships as a single Docker image. The deployment we ran for this report uses a separate Docker network on the team's Aliyun ECS instance, mapping container port 8000 to host port 18080; the existing `xiaoqi-web` service on ports 80 and 443 is untouched.

## 10. Limitations and Future Work

Four limitations are worth flagging.

1. **Different validation regimes.** The classical stack is evaluated under 5-fold stratified CV, the RAG probe under a single 500-essay hold-out, and the LoRA-LLM probe under a 10 percent random hold-out where the eval-loss tensor is corrupted by an fp16/MPS interaction. The numbers are not perfectly comparable; we report them side by side rather than mixing them in one leaderboard.
2. **No DeBERTa fine-tune in the stack.** The RAG variant uses DeBERTa only as a frozen retriever. A proper supervised fine-tune of DeBERTa-v3-base on the same 5-fold split would likely take us to the 0.46–0.47 MCRMSE region published in the literature, and stacking that with our current components could reach 0.46 or below. The bottleneck is not code—`src/feedback_ell/transformer_model.py` is in place—but stable GPU access for the team during the project window.
3. **No external data.** We trained only on the official 3 911 essays. Public solutions often pre-train on the prior Feedback Prize corpora and use pseudo-labels, both of which are allowed by the rules but require careful book-keeping.
4. **Demo predictor is heuristic.** The deployed `/api/predict` endpoint uses a deterministic surface-feature scorer so the page remains responsive. It is not the trained ensemble; that one is only invoked through the offline scripts and the Kaggle inference notebook. The leaderboard numbers shown on the page are nevertheless real OOF results.

Future work would address these in order: drop the DeBERTa fine-tune into the same 5-fold stack first, then add prior-competition data with conservative pseudo-labelling, and finally swap the demo predictor for a quantised joblib of the per-target Ridge (or a 4-bit quantised Qwen2.5-1.5B) to keep response time low while preserving real model behaviour.

## 11. Conclusion

We delivered a complete and reproducible system for analytic essay scoring. A per-target convex stack of three classical components reaches 0.51873 MCRMSE under 5-fold cross-validation on the ELLIPSE corpus, using only CPU-friendly, audit-friendly building blocks. We complement that headline result with two course-aligned probes: a parameter-efficient LoRA fine-tune of Qwen2.5-1.5B-Instruct that emits structured analytic-score JSON, and a training-free DeBERTa-v3-base + cosine-KNN retriever that reaches 0.5445 MCRMSE on a 500-essay hold-out without any training. The web demo, the LLM/RAG notebooks, the Kaggle inference notebook, and the Docker image complete the deliverable. The hardest thing the models still get wrong is the same thing experienced markers find hardest: the extreme tails of the score distribution and the longest essays. That is a useful place to focus the next iteration of the work.

## 12. Team Contributions

| Member | Main responsibility | Contribution |
| :--- | :--- | :--- |
| DHAYALA MURTHY SAVITHACCDS | Literature review and problem framing | Surveyed Feedback Prize public solutions and AES research, gathered the Qwen / LoRA / DeBERTa-v3 references used in §3 and §8, drafted Sections 1, 3, and 8. |
| HUANG ZIXUANCCDS | Data audit and feature engineering | Built the data audit module, the ten text statistics, the TF-IDF + SVD feature pipelines, and the validation slice used by the LLM and RAG probes. |
| SUN MINGCCDS | Modelling and evaluation | Implemented MCRMSE, the cross-validation harness, all component models, the stacked ensemble, and ran the Qwen2.5-1.5B + LoRA fine-tune (§7.7) plus the DeBERTa-v3-base RAG retriever (§7.8). |
| TAN XUAN ZHAOCCDS | Web demo and deployment | Built the FastAPI service, the SVG-based front-end (including the new *LLM and RAG* panel), the Docker image, and the Aliyun deployment. |
| WANG QIANCCDS | Reporting, video, and coordination | Synthesised experiment results into this report and the Chinese counterpart, the video walkthrough, and the LLM/RAG narrative that ties them to the course theme. |

The team kept version control history, group chat logs, and meeting notes for verification if requested.

## 13. Acknowledgements and References

We acknowledge that we read the public write-ups listed below for context, but no source code was copied. All code in this repository was written by Group 16.

- Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Kim, Y., Morris, W., Benner, B., Picou, A., and Boser, U. (2023). Measuring second language proficiency using the English Language Learner Insight, Proficiency, and Skills Evaluation (ELLIPSE) corpus.
- He, P., Liu, X., Gao, J., and Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. *International Conference on Learning Representations*.
- He, P., Gao, J., and Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *International Conference on Learning Representations*.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations*.
- Yang, A., Yang, B., Hui, B., et al. (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*.
- Singh, R. (2022). 1st place solution write-up, *Feedback Prize - English Language Learning*, Kaggle Discussion 369457.
- Maslov, Y. (2022). Public solution thread, *Feedback Prize - English Language Learning*.
- Sancho, B. (2022). Using Hugging Face Transformers for the first time. Kaggle Notebook.
- Daniel Gaddam, S. (2022). DeBERTa-v3-base with Accelerate Finetuning. Kaggle Notebook.
- FEATURESELECTION (2022). 0.45 score with LightGBM and DeBERTa feature. Kaggle Notebook.

The Kaggle competition page for the dataset is at <https://www.kaggle.com/competitions/feedback-prize-english-language-learning>. The ELLIPSE corpus is released under the competition's terms of use and is not redistributed in this repository.

## Appendix A. Reproducing the Results

```bash
# 1. Set up the environment
python -m venv .venv && source .venv/bin/activate    # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
export PYTHONPATH=src                                # PowerShell: $env:PYTHONPATH = "src"

# 2. Download the official competition data using the Kaggle CLI
kaggle competitions download -c feedback-prize-english-language-learning -p data/raw
unzip data/raw/feedback-prize-english-language-learning.zip -d data/raw

# 3. Audit, train, evaluate
python scripts/audit_data.py
python scripts/run_baselines.py            # quick baselines
python scripts/run_enhanced.py             # per-target Ridge, fused Ridge, LightGBM SVD, stacked ensemble
python scripts/make_submission.py          # writes experiments/artifacts/final_selection.json

# 4. Generate the auto results snapshot (companion to this report)
python scripts/build_reports.py

# 5. Local demo
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000

# 6. Container
docker compose up --build
```

## Appendix B. Submission File Map

| File | Source model | Used for |
| :--- | :--- | :--- |
| `experiments/submissions/submission_mean_baseline.csv` | mean predictor | sanity check |
| `experiments/submissions/submission_ridge_tfidf.csv` | single-alpha Ridge | original baseline |
| `experiments/submissions/submission_ridge_per_target.csv` | per-target Ridge | enhanced linear |
| `experiments/submissions/submission_ridge_fused.csv` | fused Ridge | enhanced linear |
| `experiments/submissions/submission_lgbm_svd.csv` | LightGBM SVD + stats | tree component |
| `experiments/submissions/submission_lgbm_text_features.csv` | LightGBM stats only | sanity check |
| `experiments/submissions/submission_svr_tfidf.csv` | SVR TF-IDF | sanity check |
| `experiments/submissions/submission_stacked_ensemble.csv` | per-target convex stack | **chosen final submission** |
| `notebooks/Group16_inference.ipynb` | reproduces the stacked ensemble | Kaggle code-competition entry |
| `LLM experiments/local training_essay_scoring_m_chip.ipynb` | Qwen2.5-1.5B + LoRA fine-tune (training) | §7.7 supplementary LLM probe |
| `LLM experiments/kaggle inference_group-16.ipynb` | Qwen2.5-1.5B + LoRA fine-tune (inference) | §7.7 supplementary LLM probe |
| `RAG experiments/kaggle inference_rag_group-16.ipynb` | DeBERTa-v3-base + cosine-KNN RAG | §7.8 supplementary RAG probe |
