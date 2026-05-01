from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from feedback_ell.constants import TARGET_COLUMNS
from feedback_ell.data import add_text_stats
from feedback_ell.team import GROUP_CODE, TEAM_MEMBERS
from feedback_ell.utils import read_json

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"
ARTIFACTS = ROOT / "experiments" / "artifacts"
SUBMISSIONS = ROOT / "experiments" / "submissions"

app = FastAPI(title="Feedback ELL Demo", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class EssayRequest(BaseModel):
    text: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/audit")
def audit():
    return read_json(ARTIFACTS / "data_audit.json", default={"status": "missing"})


@app.get("/api/metrics")
def metrics():
    baseline = read_json(ARTIFACTS / "baseline_metrics.json", default=[])
    transformer = read_json(ARTIFACTS / "transformer_metrics.json", default=None)
    enhanced = read_json(ARTIFACTS / "enhanced_metrics.json", default=None)
    final = read_json(ARTIFACTS / "final_selection.json", default={})
    items: list[dict] = []
    if isinstance(baseline, list):
        items.extend(baseline)
    if isinstance(transformer, dict):
        items.append(transformer)
    if isinstance(enhanced, dict):
        if isinstance(enhanced.get("components"), list):
            items.extend(enhanced["components"])
        if isinstance(enhanced.get("ensemble"), dict):
            items.append(enhanced["ensemble"])
    seen: dict[str, dict] = {}
    for item in items:
        key = str(item.get("name"))
        existing = seen.get(key)
        if not existing or (item.get("cv_mcrmse") is not None and item.get("cv_mcrmse") < existing.get("cv_mcrmse", 999)):
            seen[key] = item
    return {"items": list(seen.values()), "final": final}


@app.get("/api/error_analysis")
def error_analysis():
    return read_json(ARTIFACTS / "error_analysis.json", default={"status": "missing"})


@app.get("/api/submission")
def submission():
    final = read_json(ARTIFACTS / "final_selection.json", default={})
    available = sorted(SUBMISSIONS.glob("submission_*.csv"))
    payload = {
        "final": final,
        "available": [
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
            }
            for path in available
        ],
    }
    return payload


@app.get("/api/team")
def team():
    return {"group": GROUP_CODE, "members": TEAM_MEMBERS}


def _z(value: float) -> float:
    return float(np.clip(value, 1.0, 5.0))


def heuristic_predict(text: str) -> dict:
    import pandas as pd

    df = pd.DataFrame([{"text_id": "input", "full_text": text}])
    stats = add_text_stats(df).iloc[0]
    word_count = float(stats["word_count"])
    char_count = float(stats["char_count"])
    avg_word_len = float(stats["avg_word_len"])
    sentence_count = float(stats["sentence_count"])
    paragraph_count = float(stats["paragraph_count"])
    comma_count = float(stats["comma_count"])
    semicolon_count = float(stats["semicolon_count"])
    uppercase_ratio = float(stats["uppercase_ratio"])
    digit_ratio = float(stats["digit_ratio"])

    length_signal = min(word_count, 600) / 600
    sentence_signal = min(sentence_count, 24) / 24
    avg_word_signal = (min(max(avg_word_len, 3.0), 5.5) - 3.0) / 2.5
    discourse_signal = min(comma_count + 1.5 * semicolon_count, 30) / 30
    structure_signal = min(paragraph_count, 6) / 6
    casing_penalty = min(0.6, max(0.0, uppercase_ratio - 0.05) * 4.0)
    digit_penalty = min(0.4, digit_ratio * 6.0)

    base = 2.4 + 1.4 * (0.55 * length_signal + 0.35 * avg_word_signal + 0.10 * sentence_signal)
    cohesion = base + 0.35 * discourse_signal + 0.20 * structure_signal - 0.05 * casing_penalty
    syntax = base + 0.25 * discourse_signal + 0.30 * sentence_signal - 0.10 * casing_penalty
    vocabulary = base + 0.40 * avg_word_signal + 0.10 * length_signal - 0.05 * digit_penalty
    phraseology = base + 0.30 * discourse_signal + 0.20 * avg_word_signal - 0.05 * casing_penalty
    grammar = base - casing_penalty - 0.5 * digit_penalty + 0.20 * sentence_signal
    conventions = base - 0.5 * casing_penalty - digit_penalty + 0.10 * length_signal

    scores = {
        "cohesion": _z(cohesion),
        "syntax": _z(syntax),
        "vocabulary": _z(vocabulary),
        "phraseology": _z(phraseology),
        "grammar": _z(grammar),
        "conventions": _z(conventions),
    }
    notes = []
    if word_count < 120:
        notes.append("The essay is quite short, so the model has limited evidence for higher scores.")
    elif word_count > 650:
        notes.append("The essay is long; high scores require sustained quality across paragraphs.")
    if uppercase_ratio > 0.10:
        notes.append("High uppercase ratio is treated as a casing/conventions risk.")
    if paragraph_count <= 1:
        notes.append("Only one paragraph detected; cohesion benefits from clear structuring.")
    if digit_ratio > 0.05:
        notes.append("Many digits in the text reduce vocabulary and convention estimates.")
    if not notes:
        notes.append("No obvious issues detected in surface features.")

    overall = float(np.mean(list(scores.values())))
    return {
        "scores": scores,
        "overall": overall,
        "targets": TARGET_COLUMNS,
        "stats": {
            "word_count": int(word_count),
            "char_count": int(char_count),
            "sentence_count": int(sentence_count),
            "paragraph_count": int(paragraph_count),
            "avg_word_len": round(avg_word_len, 2),
            "comma_count": int(comma_count),
            "semicolon_count": int(semicolon_count),
            "uppercase_ratio": round(uppercase_ratio, 3),
            "digit_ratio": round(digit_ratio, 3),
        },
        "notes": notes,
    }


@app.post("/api/predict")
def predict(payload: EssayRequest):
    text = payload.text.strip()
    if not text:
        return {"error": "Please paste an essay before requesting a prediction."}
    return heuristic_predict(text)


@app.get("/api/health")
def health():
    return {"status": "ok"}
