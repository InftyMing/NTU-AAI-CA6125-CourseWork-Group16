from __future__ import annotations

from feedback_ell.baseline import run_baselines
from feedback_ell.utils import read_yaml


def main() -> None:
    config = read_yaml("configs/baseline.yaml")
    results = run_baselines(config)
    for result in sorted(results, key=lambda item: item.cv_mcrmse):
        print(f"{result.name}: {result.cv_mcrmse:.5f}")


if __name__ == "__main__":
    main()
