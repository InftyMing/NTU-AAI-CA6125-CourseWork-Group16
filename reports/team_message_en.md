# Quick update for Group 16

Hey team — quick status note so we're all on the same page before we record
the video and ship the final submission.

**1. What's done**

- The pipeline is locked in. We're running a per-target convex stack of
  three components (per-target Ridge, fused Ridge, LightGBM on SVD).
  Our 5-fold CV MCRMSE is **0.51873**, which is 20.6 % below the mean
  predictor and a hair under the strongest single model.
- Both the English and the Chinese reports are rewritten from scratch
  with real numbers everywhere (`reports/Group16_report_en.md` and
  `reports/Group16_report_zh.md`). The leaderboard, the per-target RMSE,
  the error analysis, and the LLM-vs-traditional discussion all use
  the same OOF results.
- The web demo is rebuilt with custom SVG charts (no chart library) and
  is redeployed on our Aliyun box.

**2. What you can click on right now**

- Code: <https://github.com/InftyMing/NTU-AAI-CA6125-CourseWork-Group16>
  (treat the repo as private — set the visibility to **Private** in
  GitHub before submission day, course rules say no public source.)
- Live demo: <http://47.237.107.46:18080>
- Reports: `reports/Group16_report_en.md`, `reports/Group16_report_zh.md`
- Kaggle inference notebook: `notebooks/Group16_inference.ipynb`
- Video plan with shot-by-shot script: `reports/video_script_en.md`
- Compliance checklist for the NTULearn upload:
  `reports/compliance_checklist.md`

**3. What I need from each of you**

- **Everyone** — please skim `Group16_report_en.md` (or the zh version)
  and let me know within 48 hours if your name, role, and contribution
  are described accurately. Section 12 is the table to look at.
- **Everyone** — open the demo (`http://47.237.107.46:18080`),
  click around for two minutes, and report any layout glitch on your
  screen size. I tested 1080p and 1440p on Chrome and Edge.
- **Video and PDF — DHAYALA MURTHY SAVITHA + TAN XUAN ZHAO.** The
  shot-by-shot script is already in `reports/video_script_en.md`,
  so the work splits cleanly: DMS owns the narration pass (read the
  script through once, flag any line that sounds AI-translated,
  do the voiceover), TAN XUAN ZHAO drives the screen (he set up the
  Aliyun demo, so cuing the tabs and re-running the predict button
  on camera is fastest from his side). Whoever has the cleanest mic
  records audio. Target length is ten minutes, hard cap fifteen.
  Render `Group16_report_en.md` to PDF together, confirm it's under
  20 pages at 12 pt single-spaced, and save it as `Group16_report.pdf`.
  Upload the video to YouTube as **Unlisted** and drop the link into
  `Group16_video.txt` at the repo root (replace the placeholder).
- **LLM probe (Qwen2.5-1.5B + LoRA, §7.7) — WANG QIAN.** Re-run the
  inference notebook at `LLM experiments/kaggle inference_group-16.ipynb`
  once on Kaggle before submission morning so the output JSON is fresh.
- **RAG probe (DeBERTa + KNN, §7.8) — HUANG ZIXUAN.** Same drill on
  `RAG experiments/kaggle inference_rag_group-16.ipynb`. We've already
  pinned the 0.5445 MCRMSE number in the report, so the goal here is
  just to confirm the notebook still runs end-to-end.
- **Peer eval** — the form opens 11 May. Please do not skip it,
  unfilled forms cost the whole 5 % peer-evaluation slice.

**4. What's still on my plate**

- Render the markdown report to PDF (`Group16_report.pdf`), confirm
  it's under 20 pages, and re-check the 12 pt single-spaced setting.
- Run one final pass of `pytest -q` and `scripts/smoke_test.py` on
  deadline morning, plus a `curl /api/health` against the Aliyun box.
- Flip the GitHub repo to private after the video link is shared.

**5. Deadline reminder**

Final submission to NTULearn: **Sun 17 May 2026, 23:59 (UTC+8)**, hard.
Peer evaluation window: 11 May – 20 May.

If anything I've written above doesn't match what you think you signed
up for, ping me on the group chat tonight and we'll fix it.

Thanks team.
