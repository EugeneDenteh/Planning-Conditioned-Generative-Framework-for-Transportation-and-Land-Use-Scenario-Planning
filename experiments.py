import gc
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import evaluate
from src.engine import arguments, run
from src.records import read, save
from src.distributed import finish


def main():
    p = arguments()
    p.add_argument("--evaluator")
    p.add_argument("--annotations")
    p.add_argument("--distribution", action="store_true")
    p.add_argument("--split", default="test")
    p.add_argument("--density-observations")
    p.add_argument("--require-full-coverage", action="store_true")
    args = p.parse_args()
    if not args.evaluator and not args.annotations:
        raise ValueError("An independent evaluator or annotations are required")
    root = Path(args.output)
    models = {}

    def fit(stage, variant):
        options = SimpleNamespace(**vars(args))
        options.stage, options.variant = stage, variant
        options.output = str(root / "models" / variant)
        options.resume = None
        path = run(options)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return str(path)

    full1, full2 = fit(1, "full"), fit(2, "full")
    models["full"] = dict(stage1=full1, stage2=full2)
    for variant in [
        "without_text",
        "without_control",
        "without_population",
        "without_walkability_condition",
        "without_city",
        "without_walkability_loss",
        "without_dice",
    ]:
        if variant == "without_city" and not any(
            "city=" in r["prompt_structured"] for r in read(args.records, "train")
        ):
            models[variant] = dict(stage1=full1, stage2=full2, input_changed=False)
            continue
        first = fit(1, variant)
        second = fit(2, variant) if variant in ["without_text", "without_population"] else full2
        models[variant] = dict(stage1=first, stage2=second)
    for variant in ["without_semantics", "without_perceptual", "without_reconstruction"]:
        models[variant] = dict(stage1=full1, stage2=fit(2, variant))
    models["semantic_only"] = dict(stage1=full1)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        save(root / "models.json", models)
    if dist.is_initialized():
        dist.barrier()
    finish()
    if rank != 0:
        return
    options = SimpleNamespace(**vars(args))
    options.experiments = str(root / "models.json")
    options.output = str(root / "evaluation")
    evaluate.run(options)


if __name__ == "__main__":
    try:
        main()
    finally:
        finish()
