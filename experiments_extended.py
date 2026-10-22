import argparse
import os
from pathlib import Path

import torch

from src.records import read, save


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["all", "generalization", "scenarios", "report"], default="all"
    )
    parser.add_argument("--records")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-config")
    parser.add_argument("--stage1")
    parser.add_argument("--stage2")
    parser.add_argument("--evaluator")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--crs", default="EPSG:3035")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train-processes", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--criteria")
    parser.add_argument("--density-values")
    parser.add_argument("--density-validation")
    parser.add_argument("--require-full-coverage", action="store_true")
    return parser


def main():
    args = arguments().parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run one coordinator; set --train-processes for distributed training")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Use distinct nonnegative seeds")
    if args.suite in ["all", "scenarios"] and not all([args.stage1, args.stage2, args.evaluator]):
        raise ValueError(
            "Scenario experiments require both checkpoints and an independent evaluator"
        )
    if args.train_processes < 1:
        raise ValueError("Invalid training process count")
    if args.suite != "report" and not args.records:
        raise ValueError("--records is required")
    rows = read(args.records) if args.records else []
    outputs = {}
    if args.suite == "report":
        from src.scenario_experiments import refresh

        outputs["tables10_17"] = str(refresh(args))
    if args.suite in ["all", "generalization"]:
        from src.generalization import run

        outputs["tables7_9"] = str(run(rows, args))
    if args.suite in ["all", "scenarios"]:
        from src.scenario_experiments import run

        outputs["tables10_17"] = str(run(rows, args))
    save(Path(args.output) / "experiment_outputs.json", outputs)


if __name__ == "__main__":
    main()
