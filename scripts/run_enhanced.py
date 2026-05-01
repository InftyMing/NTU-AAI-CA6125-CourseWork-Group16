from __future__ import annotations

import json

from feedback_ell.enhanced import run_enhanced
from feedback_ell.utils import read_yaml


def main() -> None:
    config = read_yaml("configs/baseline.yaml")
    result = run_enhanced(config)
    rows = []
    for comp in result["components"]:
        rows.append((comp["name"], comp["cv_mcrmse"]))
    rows.append((result["ensemble"]["name"], result["ensemble"]["cv_mcrmse"]))
    rows.sort(key=lambda r: r[1])
    print("Enhanced experiment summary:")
    for name, score in rows:
        print(f"  {name}: {score:.5f}")
    print()
    print("Detailed JSON saved to experiments/artifacts/enhanced_metrics.json")
    print("Error analysis saved to experiments/artifacts/error_analysis.json")
    print(json.dumps(result["ensemble"], indent=2))


if __name__ == "__main__":
    main()
