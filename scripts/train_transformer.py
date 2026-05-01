from __future__ import annotations

from feedback_ell.transformer_model import run_transformer
from feedback_ell.utils import read_yaml


def main() -> None:
    config = read_yaml("configs/transformer.yaml")
    summary = run_transformer(config)
    print(f"Transformer CV MCRMSE: {summary['cv_mcrmse']}")
    print(f"Submission: {summary['submission_path']}")


if __name__ == "__main__":
    main()
