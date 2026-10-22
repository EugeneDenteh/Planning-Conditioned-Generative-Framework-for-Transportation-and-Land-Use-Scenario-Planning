import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import core
from . import distributed
from .config import load
from .records import Records, read, save, seed, walk_stats
from .stages import SemanticStage, SatelliteStage


def arguments(stage=None):
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--variant", default="full")
    p.add_argument("--walk-stats")
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    if stage is not None:
        p.set_defaults(stage=stage)
    return p


@torch.no_grad()
def validate(stage, loader, value):
    totals, weight = {}, 0
    devices = [torch.cuda.current_device()] if stage.device.type == "cuda" else []
    wrapped = stage.model
    stage.model = distributed.unwrap(wrapped)
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(value)
            stage.model.eval()
            for batch in loader:
                losses = stage.loss(batch)
                for key, loss in losses.items():
                    if not torch.isfinite(loss):
                        raise FloatingPointError(key)
                    totals[key] = totals.get(key, 0.0) + loss.item() * len(batch["prompt"])
                weight += len(batch["prompt"])
    finally:
        stage.model = wrapped
        stage.model.train()
    return {key: value / weight for key, value in totals.items()}


def rate(step, c):
    if step < c["warmup_steps"]:
        return (step + 1) / max(c["warmup_steps"], 1)
    x = (step - c["warmup_steps"]) / max(c["max_steps"] - c["warmup_steps"], 1)
    return 0.5 * (1 + math.cos(math.pi * min(x, 1)))


def run(args):
    c = load(args.config, args.variant)
    device, rank, world = distributed.initialize(args.device)
    seed(c["seed"] + rank)
    rows = read(args.records)
    first = [r for r in rows if r.get("split") == "train"]
    second = [r for r in rows if r.get("split") == "val"]
    if not first or not second:
        raise ValueError("Required split is empty")
    samples = Records(first, args.root, c, args.stage)
    validation = Records(second, args.root, c, args.stage)
    effective = c[f"effective_batch_stage{args.stage}"]
    if (
        args.batch_size <= 0
        or effective % (args.batch_size * world)
        or len(samples) < args.batch_size * world
    ):
        raise ValueError("Invalid microbatch")
    accumulation = effective // (args.batch_size * world)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = out / f"stage{args.stage}.pt", out / f"stage{args.stage}_latest.pt"
    if best_path.exists() and not args.resume:
        raise FileExistsError(best_path)
    state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if state and (state["config"] != c or state["stage"] != args.stage):
        raise ValueError("Checkpoint configuration differs")
    if state and state.get("world_size", 1) != world:
        raise ValueError("Checkpoint process count differs")
    if state and Path(args.resume).resolve().parent != out.resolve():
        raise ValueError("Resume in the checkpoint directory")
    if args.stage == 1:
        if state:
            stats = core.WalkStats(**state["walk_stats"])
        elif args.walk_stats:
            stats = core.WalkStats(**json.loads(Path(args.walk_stats).read_text()))
        else:
            values = asdict(walk_stats(samples, c)) if rank == 0 else None
            stats = core.WalkStats(**distributed.broadcast(values, world))
        stage = SemanticStage(c, device, stats)
    else:
        stage = SatelliteStage(c, device)
    stage.model = distributed.wrap(stage.model, device, world)
    optimizer = torch.optim.AdamW(
        stage.model.parameters(), lr=c[f"lr_stage{args.stage}"], weight_decay=c["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: rate(s, c))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    progress = dict(step=0, epoch=0, offset=0, best=None, stale=0)
    if state:
        distributed.unwrap(stage.model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        progress = state["progress"]
        rng = state["rng"][rank] if isinstance(state["rng"], list) else state["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if device.type == "cuda":
            saved = rng["cuda"][device.index] if isinstance(rng["cuda"], list) else rng["cuda"]
            torch.cuda.set_rng_state(saved, device)
    loader = DataLoader(validation, batch_size=1, collate_fn=core.collate, num_workers=args.workers)

    def checkpoint(path):
        rng = distributed.gather(
            dict(
                torch=torch.get_rng_state(),
                numpy=np.random.get_state(),
                python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            ),
            world,
        )
        if rank != 0:
            return
        temporary = path.with_suffix(".tmp")
        torch.save(
            dict(
                stage=args.stage,
                config=c,
                model=distributed.unwrap(stage.model).state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                progress=dict(progress),
                walk_stats=asdict(stage.stats) if args.stage == 1 else None,
                world_size=world,
                rng=rng,
            ),
            temporary,
        )
        temporary.replace(path)

    optimizer.zero_grad(set_to_none=True)
    micro = 0
    stage.model.train()
    while progress["step"] < c["max_steps"]:
        order = list(range(len(samples)))
        random.Random(c["seed"] + progress["epoch"]).shuffle(order)
        if world > 1:
            order += order[: (-len(order)) % world]
            order = order[rank::world]
        batches = DataLoader(
            samples,
            batch_size=args.batch_size,
            sampler=order[progress["offset"] :],
            drop_last=True,
            collate_fn=core.collate,
            num_workers=args.workers,
            generator=torch.Generator().manual_seed(c["seed"] + progress["epoch"]),
        )
        for batch in batches:
            sync = (micro + 1) % accumulation == 0
            context = stage.model.no_sync() if world > 1 and not sync else nullcontext()
            with context:
                loss = stage.loss(batch)
                if not torch.isfinite(loss["total"]):
                    raise FloatingPointError("Nonfinite objective")
                scaler.scale(loss["total"] / accumulation).backward()
            progress["offset"] += args.batch_size
            micro += 1
            if micro % accumulation:
                continue
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(stage.model.parameters(), c["max_grad_norm"])
            if device.type != "cuda" and not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients")
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < previous:
                continue
            scheduler.step()
            progress["step"] += 1
            step = progress["step"]
            if step % 10 == 0 and rank == 0:
                print(
                    json.dumps(dict(stage=args.stage, step=step, loss=loss["total"].item())),
                    flush=True,
                )
            if step % c["validate_every"] == 0 or step == c["max_steps"]:
                scores = None
                if rank == 0:
                    try:
                        scores = validate(stage, loader, c["validation_seed"])
                    except Exception as error:
                        scores = dict(error=str(error))
                scores = distributed.broadcast(scores, world)
                if "error" in scores:
                    raise FloatingPointError(scores["error"])
                improved = (
                    progress["best"] is None or scores["total"] < progress["best"] - c["min_delta"]
                )
                progress["stale"] = 0 if improved else progress["stale"] + 1
                if improved:
                    progress["best"] = scores["total"]
                    checkpoint(best_path)
                checkpoint(latest_path)
                if rank == 0:
                    record = dict(
                        stage=args.stage,
                        step=step,
                        learning_rate=scheduler.get_last_lr()[0],
                        validation=scores,
                    )
                    save(out / f"stage{args.stage}_validation.json", dict(step=step, **scores))
                    with (out / f"stage{args.stage}_metrics.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                if c["patience"] > 0 and progress["stale"] >= c["patience"]:
                    return best_path
            if step >= c["max_steps"]:
                break
        progress["epoch"] += 1
        progress["offset"] = 0
    return best_path
