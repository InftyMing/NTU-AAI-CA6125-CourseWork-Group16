# CA6125 Feedback Prize ELL Project

This repository implements the CA6125 LLM & RAG course project for Kaggle Feedback Prize - English Language Learning.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Data

Preferred automated route:

```powershell
python scripts/download_data.py
```

Kaggle still requires an API token for programmatic downloads and submissions. If the browser is logged in but no token exists, open `https://www.kaggle.com/settings`, create an API token, and place `kaggle.json` in `%USERPROFILE%\.kaggle\` or the project root.

For local smoke tests without official data:

```powershell
python scripts/create_demo_data.py
python scripts/smoke_test.py
```

## Experiments

```powershell
python scripts/audit_data.py
python scripts/run_baselines.py
python scripts/train_transformer.py
python scripts/make_submission.py
```

The main metric is MCRMSE. Experiment summaries are written to `experiments/artifacts/`, and submissions are written to `experiments/submissions/`.

## Reports

```powershell
python scripts/build_reports.py
```

Outputs:

- `reports/Group16_report_en.md`
- `reports/Group16_report_zh.md`
- `reports/video_script_en.md`
- `reports/video_demo_flow_zh.md`

The Kaggle notebook-style inference artifact is `notebooks/Group16_inference.ipynb`.

## Web Demo

```powershell
$env:PYTHONPATH = "src"
uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Docker

```powershell
docker compose up --build
```

Open `http://localhost:8000`.

## Final Submission Checklist

- `Group16_report.pdf`
- `Group16_video.txt`
- Final Kaggle `.csv` or `.ipynb`
- Keep source code private unless requested for plagiarism verification.
