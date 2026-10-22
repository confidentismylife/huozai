from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.li_group_validation import (
    reproduce_exit_choice_model,
    reproduce_group_splitting,
    write_validation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Lu et al. (2026) three-exit equations and group splitting.")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "li_group_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_rows, sensitivity_rows, choice_metrics = reproduce_exit_choice_model(args.draws, args.seed)
    split_rows, split_metrics = reproduce_group_splitting(args.draws, args.seed + 1)
    write_validation_outputs(
        scenario_rows,
        sensitivity_rows,
        choice_metrics,
        split_rows,
        split_metrics,
        args.output,
    )
    print(f"Wrote {len(scenario_rows)} scenario-mode results to {args.output}")
    print(f"Choice Monte Carlo MAE: {float(choice_metrics['monte_carlo_probability_mae']):.4f}")
    print(f"Splitting Monte Carlo MAE: {float(split_metrics['simulation_mae']):.4f}")


if __name__ == "__main__":
    main()

