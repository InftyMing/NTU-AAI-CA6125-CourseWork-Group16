# CA6125 Group 16 — Submission Compliance Checklist

This is the pre-flight check we run before uploading to NTULearn on or before
**2026-05-17 23:59 UTC+8** (hard deadline). Source: `context.md` plus the
course PDF "Course general information.pdf".

Status legend: ✅ done · ⏳ to be done before submission · ⚠ needs human action.

---

## 1. Files to upload

| Required file | Status | Where |
| :--- | :--- | :--- |
| `Group16_report.pdf` (≤ 20 pages, 12 pt, single-spaced) | ⏳ export from `reports/Group16_report_en.md` | `reports/Group16_report.pdf` |
| `Group16_video.txt` (single line: YouTube link) | ⏳ create after recording | repo root or `reports/` |
| Kaggle entry: `.csv` **or** `.ipynb` | ✅ both ready | `experiments/submissions/submission_stacked_ensemble.csv`, `notebooks/Group16_inference.ipynb` |

**Action for PDF export:** open `reports/Group16_report_en.md` in VS Code with
*Markdown PDF* extension, or run `pandoc reports/Group16_report_en.md
-o reports/Group16_report.pdf --pdf-engine=xelatex -V mainfont="Times New Roman"
-V fontsize=12pt -V geometry=margin=2.5cm`. Verify final page count ≤ 20.

**Action for video.txt:** after uploading the YouTube video, write the link:

```text
https://youtu.be/<unlisted-video-id>
```

Confirm the YouTube visibility is **Unlisted** (not Private — graders need
the link to work without an account).

---

## 2. Report content (rubric, 15 % of total grade)

| Requirement | Section in `Group16_report_en.md` | Status |
| :--- | :--- | :--- |
| Members + roles + concrete contributions | §12 Team Contributions | ✅ |
| Kaggle CV / leaderboard score | Abstract + §7.1 (CV MCRMSE 0.51873) | ✅ |
| Task in our own words | §1 Introduction, §2 Problem Formulation | ✅ |
| Challenges analysis | §2 (constraints), §7.5 (error analysis), §10 (limitations) | ✅ |
| Comparison vs. traditional ML, with LLM/RAG strengths and weaknesses | §8 Comparison | ✅ |
| Solution details | §5 Methodology | ✅ |
| Experimental study justifying the design | §6 Setup, §7 Results | ✅ |
| What we learned | §11 Conclusion | ✅ |
| Acknowledgement of any reference code | §13 Acknowledgements and References | ✅ |

Length sanity: 4 390 English words, ten tables, three appendices. At 12 pt
single-spaced on A4 with 2.5 cm margins this fits in 14 – 17 pages — well
under the 20-page cap.

---

## 3. Video content (rubric, 10 % of total grade)

| Requirement | Status | Notes |
| :--- | :--- | :--- |
| Length ≤ 15 min | ⏳ planned at 10:00 in `reports/video_script_en.md` |
| YouTube hosted (private/unlisted permitted) | ⏳ upload as **Unlisted** |
| Summarises project, NOT a code dump | ✅ script is 60–80 % web demo + report walk-through |
| Mentions team and contributions | ✅ §0:00 cold open, §7:30 team grid, §9:30 sign-off |
| Numbers consistent with report | ✅ MCRMSE 0.51873 cited identically |
| Honest about limitations | ✅ §5:00 error analysis, §8:00 LLM trade-off |

---

## 4. Convincingness / novelty / writing (5 + 5 + 5 grading axes)

- **Convincingness** — every leaderboard number on the demo and in the
  report comes from `experiments/artifacts/*.json`, written by
  `scripts/run_enhanced.py`. Reproducible end-to-end on a CPU laptop in
  about 4 minutes. Tests: `pytest -q` passes (2 / 2).
- **Novelty** — per-target convex stack of heterogeneous components
  (per-target Ridge with inner-fold $\alpha$ search, fused TF-IDF + scaled
  statistics, LightGBM on Truncated SVD) fitted on out-of-fold predictions
  with a 21-step simplex grid per target. Documented in §5.4 of the report.
- **Writing** — both English and Chinese reports proofread; consistent
  notation; no stray AI-style phrasing.

---

## 5. Source-code privacy

> Course rule (PDF page 2): *"Do NOT publicly publish your source code."*

| Item | Action |
| :--- | :--- |
| GitHub repo `InftyMing/NTU-AAI-CA6125-CourseWork-Group16` | ⚠ **Set to PRIVATE before submission deadline.** Public repo would violate the rule. The `gh` command is `gh repo edit InftyMing/NTU-AAI-CA6125-CourseWork-Group16 --visibility private`. |
| Aliyun web demo at `47.237.107.46:18080` | ✅ serves only HTML/JSON/CSS, no source files. |
| Inference notebook on Kaggle | ✅ kept in private mode while attached to the team's Kaggle account. |
| `Group16_inference.ipynb` (uploaded to NTULearn for grading) | ✅ allowed — graders use it to verify results, not for publication. |

---

## 6. Kaggle / data redistribution

| Item | Status |
| :--- | :--- |
| Did NOT copy any high-rank Kaggle solution code | ✅ all components in `src/feedback_ell/` written by Group 16 |
| Public write-ups read for context (Singh, Maslov, Sancho, Daniel Gaddam, FEATURESELECTION) | ✅ acknowledged in §13 |
| Raw `train.csv` / `test.csv` not redistributed in repo | ✅ `data/raw/*.csv` is in `.gitignore` |
| Kaggle API token | ✅ `kaggle.json` and `*.pem` are in `.gitignore` |
| `Course general information.pdf` | ✅ in `.gitignore`, kept locally only |

---

## 7. Peer evaluation

| Window | 2026-05-11 – 2026-05-20 | Action |
| :--- | :--- | :--- |
| Required for every member | ⏳ each of the five members fills the form once it opens |
| Comments mandatory for scores ≤ 2/5 or = 5/5 | ⏳ each member writes one short comment per teammate |

Missing peer evaluation forfeits 5 points (the entire peer-evaluation slice).

---

## 8. Final pre-upload checks (run on the morning of the deadline)

```bash
# 1. From repo root: run all tests + smoke
pytest -q                                  # expect: 2 passed
python scripts/smoke_test.py               # expect: no traceback

# 2. Confirm live demo is reachable
curl -sf http://47.237.107.46:18080/api/health    # expect: {"status":"ok"}
curl -sf http://47.237.107.46:18080/api/metrics | python -m json.tool | head

# 3. Confirm the chosen model is the stack
python -c "import json; d=json.load(open('experiments/artifacts/final_selection.json')); print(d['best']['name'], d['best']['cv_mcrmse'])"
# expect: stacked_ensemble 0.5187...

# 4. Render the report PDF and check the page count
pandoc reports/Group16_report_en.md -o reports/Group16_report.pdf --pdf-engine=xelatex -V fontsize=12pt -V geometry=margin=2.5cm
# verify: ≤ 20 pages

# 5. Make the GitHub repo private
gh repo edit InftyMing/NTU-AAI-CA6125-CourseWork-Group16 --visibility private --accept-visibility-change-consequences
```

When all of the above pass, upload `Group16_report.pdf`, `Group16_video.txt`,
and `submission_stacked_ensemble.csv` (plus `Group16_inference.ipynb` if the
Kaggle entry is the notebook track) to NTULearn before 23:59 on 17 May.
