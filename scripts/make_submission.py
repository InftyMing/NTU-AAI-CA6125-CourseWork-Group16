from __future__ import annotations

from pathlib import Path

from feedback_ell.kaggle_io import has_kaggle_credentials, kaggle_auth_hint, submit_to_kaggle
from feedback_ell.submission import choose_best_submission


def main() -> None:
    best = choose_best_submission(
        [
            "experiments/artifacts/baseline_metrics.json",
            "experiments/artifacts/transformer_metrics.json",
            "experiments/artifacts/enhanced_metrics.json",
        ]
    )
    if not best:
        raise SystemExit("No candidate submission found. Run experiments first.")
    submission_path = Path(best["submission_path"])
    print(f"Selected submission: {submission_path}")
    print(f"CV MCRMSE: {best.get('cv_mcrmse'):.5f}")
    if has_kaggle_credentials():
        try:
            submit_to_kaggle(submission_path, f"CA6125 final candidate: {best['name']}")
            print("Submitted to Kaggle.")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"Kaggle direct CSV submission failed ({exc}).")
            print(
                "This is expected for code competitions; use notebooks/Group16_inference.ipynb on Kaggle."
            )
    else:
        print(kaggle_auth_hint())
        print("Submission file is ready for manual upload.")


if __name__ == "__main__":
    main()
