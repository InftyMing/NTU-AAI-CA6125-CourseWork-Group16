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
- **One of us records the video** — the script is already there.
  Whoever has the cleanest mic and the calmer voice should take it.
  My suggestion: WANG QIAN does the narration, TAN XUAN ZHAO drives
  the screen if WANG QIAN can't share screen smoothly during the take.
  Target length is ten minutes, hard cap fifteen. Upload to YouTube
  as **Unlisted** and drop the link into `Group16_video.txt` at the
  repo root (replace the placeholder).
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
