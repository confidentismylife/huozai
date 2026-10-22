from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.lu_warden_validation import validate_published_scenarios, write_validation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the 12 Lu et al. (2025) exit-choice scenarios.")
    parser.add_argument("--draws-per-scenario", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "lu_warden_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, metrics = validate_published_scenarios(
        draws_per_scenario=args.draws_per_scenario,
        seed=args.seed,
    )
    write_validation_outputs(rows, metrics, args.output)
    print(f"Wrote {len(rows)} scenario results to {args.output}")
    print(f"Analytic MAE: {float(metrics['analytic_mae']):.4f}")
    print(f"Simulated MAE: {float(metrics['simulated_mae']):.4f}")
    print(f"Majority direction accuracy: {float(metrics['majority_direction_accuracy']):.2%}")


if __name__ == "__main__":
    main()

