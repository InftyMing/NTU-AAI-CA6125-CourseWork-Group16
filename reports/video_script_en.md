# Group 16 · CA6125 LLM and RAG · Project Video Script

**Target length:** 10 minutes (hard cap 15 min per course brief).
**Recording resolution:** 1920 × 1080, browser at 100% zoom, terminal font ≥ 16 pt.
**Audio:** single narrator, headset microphone, no background music. Read in a relaxed, slightly informal tone — pause for half a second between sections instead of using filler words.
**Window layout:**
- Tab A: live demo at `http://47.237.107.46:18080`.
- Tab B: GitHub repo (`InftyMing/NTU-AAI-CA6125-CourseWork-Group16`).
- Tab C: Kaggle competition page (`feedback-prize-english-language-learning`).
- Tab D: VS Code with the project open.
- Tab E: the PDF report (`Group16_report.pdf`).

The web demo carries the bulk of the story (roughly 6 minutes 30 seconds), with short pivots into the report and the codebase. Treat the timestamps as targets, not contracts; record once end-to-end first, then trim.

---

## 0:00 – 0:30 · Cold open (title card + framing)

[Screen: a still slide that shows the project title, the six target names in two rows, and the Group 16 list. Narrator on camera in a small corner is fine but optional.]

> "Hi, this is Group 16 for CA6125. Our project is the Kaggle Feedback Prize English Language Learning competition. The job sounds simple — read an essay written by a non-native English student and give six analytic scores: cohesion, syntax, vocabulary, phraseology, grammar, conventions. The official metric is mean column-wise RMSE, lower is better. In the next ten minutes I'll walk through what we built, what numbers we actually got, and where the model is honest about being wrong."

[End the slide on the word "wrong"; cut to Tab C.]

---

## 0:30 – 1:00 · Why this competition (Kaggle page)

[Screen: scroll the Kaggle competition page slowly from the title to the evaluation block, then to the dataset block.]

> "This is the official competition page. A few things matter for our design. One — it's a code competition, so our submission has to be a notebook, not just a CSV. Two — the public test set is three essays. We can't tune to the leaderboard; we have to trust local cross-validation. Three — scores are continuous between 1.0 and 5.0 in steps of 0.5. Treating it as regression, with a six-output stratified split, is the natural choice."

[Cut.]

---

## 1:00 – 1:40 · The pipeline shape (VS Code)

[Screen: VS Code, file tree expanded to show `src/feedback_ell/`, `scripts/`, `web/`, `experiments/artifacts/`. Don't read code line by line.]

> "Here is how the repo is laid out. Everything that produces a number lives under `src/feedback_ell`: the data audit, the baseline models, and a new module called `enhanced.py` for the per-target Ridge, the fused Ridge, the LightGBM stack, and the convex blend. The four scripts in `scripts/` are thin wrappers — audit data, run baselines, run enhanced, build submission. Whatever the scripts produce ends up as a JSON or CSV in `experiments/artifacts/` and `experiments/submissions/`. The web app reads those JSONs at request time, which is why everything you'll see in the demo is the actual cross-validation result, not a hand-edited table."

[Click open `src/feedback_ell/enhanced.py` for two seconds, scroll through `_search_alpha_per_target` and `_blend_weights`, then close.]

> "If you're checking the code afterwards, the convex blend search lives at `_blend_weights`. We grid-search a 21-step simplex per target, on out-of-fold predictions only."

[Cut to Tab A — the demo top, fresh reload.]

---

## 1:40 – 2:30 · Demo overview (hero + KPIs + pipeline)

[Screen: demo home, scroll-to-top.]

> "This is the public demo running on our Aliyun server, port 18080. Three KPI cards at the top — best CV MCRMSE, training set size, and the six target dimensions. The best number, **0.51873**, is the stacked ensemble. That's a 20.6% relative improvement over the mean predictor and 0.7% over the strongest single component. Three thousand nine hundred eleven essays in the training set, no synthetic data, no external corpus."

[Scroll one screen down to the Pipeline section.]

> "Just below the hero we have the pipeline diagram in five steps: audit, feature views, three component models, the convex stack, and the submission packaging. This mirrors the headings in the report, so a marker can map any number on this page back to a section."

[Pause briefly so the SVG finishes drawing.]

---

## 2:30 – 3:30 · Data audit (mean bars + correlation heatmap + length boxplot)

[Screen: scroll to the `#data` panel.]

> "Data audit first. The four cards on top come straight from `data_audit.json`. Train rows, test rows, missing values, mean essay length. Three cards say zero missing, which is good — no imputation tricks needed."

[Hover the mean-score bar chart on the left.]

> "Mean per target. All six sit close to 3.0, with vocabulary slightly higher and grammar slightly lower. That tells us a constant predictor is already at 0.65 MCRMSE, which is the floor we have to beat."

[Move cursor to the correlation heatmap.]

> "On the right is the Pearson correlation between the six targets. The lightest cell is cohesion against grammar at 0.64, the darkest is vocabulary against phraseology at 0.74. That density of correlation is why per-target weighting in the stack actually matters — the targets are not independent, but they're not the same either."

[Scroll the box-plot for length.]

> "Length distribution at the bottom — quartile summary in one box. Median around 400 words, longest above 1200, shortest 14. Later in error analysis you'll see that the long tail is also where the model loses ground."

[Cut.]

---

## 3:30 – 5:00 · Models and leaderboard

[Screen: scroll to `#models`.]

> "Models section. The leaderboard table is sorted by CV MCRMSE. The numbers from worst to best — mean predictor 0.65281, LightGBM on text statistics 0.60654, single-alpha Ridge 0.53310, LightGBM on SVD 0.53445, Ridge fused 0.52734, Ridge per-target 0.52525, SVR on TF-IDF 0.52911, and the stacked ensemble at 0.51873."

[Click the `stacked_ensemble` row in the table.]

> "Clicking a row lights up the per-target chart on the right. For the stack you can see vocabulary at 0.4565, syntax 0.5097, phraseology 0.5169, cohesion 0.5319, conventions 0.5346, and grammar 0.5629. Vocabulary is easy because TF-IDF is a vocabulary detector by design. Grammar is hardest, and that matches the literature — local edit-level errors are not what bag-of-n-grams captures well."

[Scroll to the fold-stability line chart below.]

> "Below that is the per-fold MCRMSE. For per-target Ridge the spread between folds is 0.009, for LightGBM SVD it's 0.011. The gap between models is bigger than the gap between folds, so the leaderboard ordering is not noise."

[Pause for two seconds. Cut.]

---

## 5:00 – 6:00 · Error analysis

[Screen: scroll to `#errors`.]

> "Honest part of the demo. We slice OOF residuals two ways."

[Hover the length-bucket bar chart.]

> "By essay length quartile, the shortest quartile is at 0.5069 MCRMSE, the longest at 0.5386. About six percent worse on the longest essays. Long essays mix strong and weak passages, and global TF-IDF averages them."

[Hover the score-band chart.]

> "By the average target band — low, medium, high — the model is best in the middle, around 0.46, and worse on both tails, 0.62 on low and 0.59 on high. That is regression-to-the-mean. Whatever signal we picked up still pulls extreme essays toward 3.0. If we had time, this is exactly what a calibration head or a Transformer fine-tune would address."

[Scroll to the per-target diagnostic table.]

> "Bottom table — per target mean residual, mean absolute error, and worst absolute residual. Mean residual is essentially zero everywhere, so the model is unbiased on average. Mean absolute error walks from 0.36 on vocabulary up to 0.46 on grammar. The biggest single miss is a little above two points, which is one outlier essay, not a pattern."

[Cut.]

---

## 6:00 – 7:30 · Live prediction

[Screen: scroll to `#demo`. Click the first sample tab so the textarea fills.]

> "Live demo. We pre-loaded a real student essay; you can paste your own. The hosted predictor uses an interpretable surface-feature scorer so the page is fast and stateless — the leaderboard above is the actual stacked ensemble, the textarea here is a quick sanity tool."

[Click "Predict scores".]

> "Predict. The overall card on the right shows the rounded mean. Each of the six bars is a target, with the predicted score and the bar fill scaled to the 1.0 to 5.0 range. The signal list at the bottom explains what surface features triggered which adjustment — short essay, high uppercase ratio, only one paragraph, things like that. None of this is hand-coded for the demo; it comes from the same `add_text_stats` we used during training."

[Clear the textarea, paste a clearly bad sample (mostly uppercase, one paragraph, no punctuation). Click Predict.]

> "If I throw a deliberately weak example at it — all caps, no punctuation, one paragraph — the conventions and grammar bars drop, and the signal list flags casing and structure. That's the kind of behaviour we want a marker to be able to verify in five seconds."

[Cut.]

---

## 7:30 – 8:00 · Team and submission

[Screen: scroll to `#team`.]

> "Team grid. Five members, five roles — literature, data and features, modelling, web and deploy, reporting and video. Every name is one of the official Group 16 members. The breakdown matches Section 12 of the report."

[Scroll to `#submission`.]

> "Submission section. The `Selected model` card shows the final pick from `final_selection.json` — stacked ensemble, CV MCRMSE 0.51873, written to `submission_stacked_ensemble.csv`. The list on the right is every CSV under `experiments/submissions/`, including the sanity baselines, so a marker can pick any of them and rerun the comparison."

[Cut.]

---

## 8:00 – 8:30 · Report (PDF)

[Screen: switch to Tab E. Open `Group16_report.pdf`. Scroll the abstract, then the headings panel.]

> "The report mirrors the demo. Abstract on page one, problem formulation in Section 2, methodology in Section 5, results and the leaderboard table in Section 7, error analysis in Section 7.5, and the comparison between traditional and Transformer methods in Section 8. The numbers in every table on this page are the same numbers you just saw in the browser."

[Scroll quickly to Section 8 and stop on the "Where the Transformer would help" paragraph.]

> "Section 8 is the part the brief asks for explicitly. Our position is that a CPU-only traditional pipeline reaches 0.519, a single DeBERTa-v3-base would push us into 0.46, and an industrial ensemble like the public first place sits around 0.43. We did not run the Transformer end-to-end this round because we don't have a stable GPU; the path is still in `src/feedback_ell/transformer_model.py` for the future."

[Cut.]

---

## 8:30 – 9:00 · Code on GitHub and Kaggle inference notebook

[Screen: switch to Tab B — GitHub repo. Hover `notebooks/Group16_inference.ipynb`, then click into it.]

> "Source is on GitHub at `InftyMing/NTU-AAI-CA6125-CourseWork-Group16`. The Kaggle entry point is `notebooks/Group16_inference.ipynb` — that notebook reproduces the stacked ensemble end-to-end on Kaggle's hidden test set, so the marker can verify the score without retraining anything locally. Sensitive items — Kaggle tokens, raw competition data, the deploy bundle — are excluded from the repo through `.gitignore`, per the course rules about not redistributing the dataset."

[Cut.]

---

## 9:00 – 9:30 · What we learned, honestly

[Screen: back on Tab A, scrolled to the top of the demo. Don't move the mouse much.]

> "Three takeaways before we wrap up."
>
> "First — careful classical features still go a long way. Per-target alpha plus a fused TF-IDF and statistics matrix gets us most of the way to a Transformer baseline, on a laptop CPU, in four minutes."
>
> "Second — stacking only helps when components actually disagree. The grammar weights lean on LightGBM, the vocabulary weights lean on Ridge. If we'd stacked three near-identical models we'd have spent compute for nothing."
>
> "Third — the demo and the report should answer the same questions. We deliberately built the page so a marker can re-derive every claim in the PDF without trusting us."

---

## 9:30 – 10:00 · Acknowledgements + sign-off

[Screen: title card again, brief credits text — group code, course code, "Thanks for watching".]

> "Group 16 — DHAYALA MURTHY SAVITHA, HUANG ZIXUAN, SUN MING, TAN XUAN ZHAO, WANG QIAN. We acknowledge the public Kaggle write-ups we referenced for context — Singh, Maslov, Sancho, Daniel Gaddam, FEATURESELECTION — none of their training scripts were copied. Full reference list is in Section 13 of the report. Thanks for watching."

[Hold on the title card for two seconds. End.]

---

## Recording checklist

- Reload `http://47.237.107.46:18080` once before recording so the data is hot in the browser cache.
- Confirm `experiments/artifacts/final_selection.json` shows the stacked ensemble.
- Run the demo predict twice off-camera so the first request doesn't pay the cold-start cost.
- Have the report PDF on page one before you start; do not search inside the PDF on camera.
- Use OBS scene transitions (cut, not fade) to keep the pace tight.
- Total spoken time should be 9:30 – 9:50; budget 10–30 seconds for tab switches.
- Upload to YouTube as **unlisted**, then save the link inside `Group16_video.txt`.
