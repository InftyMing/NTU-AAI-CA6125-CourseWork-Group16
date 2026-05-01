"""Kaggle download and submission helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from feedback_ell.constants import COMPETITION_SLUG


def _import_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Kaggle package is unavailable. Install requirements.txt and configure Kaggle access."
        ) from exc
    api = KaggleApi()
    api.authenticate()
    return api


def has_kaggle_credentials() -> bool:
    explicit = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    api_token = os.environ.get("KAGGLE_API_TOKEN")
    default_token = Path.home() / ".kaggle" / "kaggle.json"
    local_token = Path("kaggle.json")
    return bool(explicit or api_token or default_token.exists() or local_token.exists())


def download_competition_data(output_dir: str | Path = "data/raw") -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    api = _import_kaggle_api()
    api.competition_download_files(COMPETITION_SLUG, path=str(output), quiet=False)
    zip_path = output / f"{COMPETITION_SLUG}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output)
    return sorted(output.glob("*.csv"))


def submit_to_kaggle(
    submission_path: str | Path,
    message: str = "CA6125 Feedback ELL project submission",
) -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        executable = shutil.which("kaggle")
        fallback = Path.home() / "AppData" / "Roaming" / "Python" / "Python314" / "Scripts" / "kaggle.exe"
        if executable is None and fallback.exists():
            executable = str(fallback)
        if executable:
            subprocess.run(
                [
                    executable,
                    "competitions",
                    "submit",
                    "-c",
                    COMPETITION_SLUG,
                    "-f",
                    str(submission_path),
                    "-m",
                    message,
                ],
                check=True,
            )
            return
    api = _import_kaggle_api()
    api.competition_submit(str(submission_path), message, COMPETITION_SLUG)


def kaggle_auth_hint() -> str:
    return (
        "Kaggle API needs credentials even if the browser is logged in. "
        "Open https://www.kaggle.com/settings, create an API token, and place kaggle.json "
        "under ~/.kaggle/ or this project root. Browser login is still useful for joining rules."
    )
