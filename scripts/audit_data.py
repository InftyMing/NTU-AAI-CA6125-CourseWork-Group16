from __future__ import annotations

from feedback_ell.data import write_audit


def main() -> None:
    audit = write_audit(
        "data/raw/train.csv",
        "data/raw/test.csv",
        "experiments/artifacts/data_audit.json",
    )
    print(f"Train rows: {audit['train_rows']}")
    print(f"Test rows: {audit['test_rows']}")


if __name__ == "__main__":
    main()
