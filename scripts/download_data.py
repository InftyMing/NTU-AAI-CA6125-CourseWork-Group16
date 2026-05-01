from __future__ import annotations

from feedback_ell.kaggle_io import download_competition_data, has_kaggle_credentials, kaggle_auth_hint


def main() -> None:
    if not has_kaggle_credentials():
        raise SystemExit(kaggle_auth_hint())
    files = download_competition_data("data/raw")
    print("Downloaded files:")
    for file in files:
        print(f"- {file}")


if __name__ == "__main__":
    main()
