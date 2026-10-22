from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fire_evac.behavior_data import (
    classification_metrics,
    fit_hierarchical_logit,
    generate_synthetic_behavior_data,
    read_behavior_csv,
    rows_for_participants,
    save_model,
    split_participants,
    write_behavior_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe participant-level behavior data.")
    parser.add_argument("--input", type=Path, help="Real behavior CSV. Omit only to generate a labelled synthetic demo.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "behavior_pipeline")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--demo-participants", type=int, default=60)
    parser.add_argument("--demo-trials", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input is None:
        rows = generate_synthetic_behavior_data(args.demo_participants, args.demo_trials, args.seed)
        source_path = args.output_dir / "synthetic_behavior_demo.csv"
        write_behavior_csv(rows, source_path)
        source_kind = "synthetic_demo_not_human_evidence"
    else:
        rows = read_behavior_csv(args.input)
        source_path = args.input.resolve()
        source_kind = "user_supplied"

    split = split_participants(rows, seed=args.seed)
    split_rows = {
        "train": rows_for_participants(rows, split.train),
        "validation": rows_for_participants(rows, split.validation),
        "test": rows_for_participants(rows, split.test),
    }
    for name, items in split_rows.items():
        write_behavior_csv(items, args.output_dir / f"{name}.csv")

    model = fit_hierarchical_logit(split_rows["train"])
    metrics = {
        name: classification_metrics(items, model.predict_probability(items, use_random_effects=name == "train"))
        for name, items in split_rows.items()
    }
    metadata = {
        "source_path": str(source_path),
        "source_kind": source_kind,
        "split_seed": args.seed,
        "participant_counts": {name: len(getattr(split, name)) for name in ("train", "validation", "test")},
        "row_counts": {name: len(items) for name, items in split_rows.items()},
    }
    save_model(model, args.output_dir / "behavior_model.json", metadata)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"metadata": metadata, "metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "participant_split.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["participant_id", "split"])
        writer.writeheader()
        for name in ("train", "validation", "test"):
            writer.writerows({"participant_id": participant_id, "split": name} for participant_id in getattr(split, name))

    print(json.dumps({"metadata": metadata, "metrics": metrics}, ensure_ascii=False, indent=2))
    if args.input is None:
        print("WARNING: generated data are synthetic and must not be reported as human evidence.")


if __name__ == "__main__":
    main()
