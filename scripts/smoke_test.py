from __future__ import annotations

from feedback_ell.data import write_audit
from feedback_ell.metrics import mcrmse


def main() -> None:
    score = mcrmse([[1, 2], [3, 4]], [[1, 2], [4, 6]], columns=["a", "b"])
    assert round(score, 6) == round(((0.5**0.5) + (2.0**0.5)) / 2, 6)
    audit = write_audit(
        "data/raw/train.csv",
        "data/raw/test.csv",
        "experiments/artifacts/data_audit.json",
    )
    assert audit["train_rows"] > 0
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
