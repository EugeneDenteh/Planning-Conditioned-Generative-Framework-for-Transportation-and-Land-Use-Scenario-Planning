import gc
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from . import engine, metrics, tables
from .config import CLASSES
from .geography import assignments, city, digest, transfer_assignments
from .records import save


SETTINGS = [("stratified", 0, 0), ("block5", 5, 1), ("block10", 10, 2), ("block20", 20, 5)]


def ratio(value, reference, scale=1):
    return scale * value / reference if value is not None and reference else None


def fit(rows, args, out, identity):
    import evaluate

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        identity,
        records=rows,
        configuration=(
            json.loads(Path(args.train_config).read_text()) if args.train_config else None
        ),
    )
    specification = out / "specification.json"
    if specification.exists() and json.loads(specification.read_text()) != identity:
        raise ValueError("Existing experiment specification differs")
    save(specification, identity)
    completed = out / "complete.json"
    if completed.exists():
        entry = json.loads(completed.read_text())
        if digest(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after evaluation")
        return entry
    manifest = out / "records.json"
    save(manifest, rows)
    options = SimpleNamespace(
        config=args.train_config,
        variant="full",
        records=str(manifest),
        root=args.root,
        stage=1,
        device=args.device,
        output=str(out / "model"),
        resume=None,
        batch_size=args.batch_size,
        workers=args.workers,
        walk_stats=None,
    )
    if args.train_processes > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.train_processes}",
            str(Path(__file__).resolve().parents[1] / "train_stage1.py"),
            "--records",
            str(manifest),
            "--root",
            args.root,
            "--output",
            options.output,
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
        ]
        if args.train_config:
            command.extend(["--config", args.train_config])
        subprocess.run(command, check=True)
        checkpoint = Path(options.output) / "stage1.pt"
    else:
        checkpoint = engine.run(options)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test = [r for r in rows if r["split"] == "test"]
    options = SimpleNamespace(device=args.device, root=args.root, evaluator=None, annotations=None)
    result = evaluate.score(test, options, checkpoint, None, out / "evaluation")
    entry = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        sample_ids=[r["tile_id"] for r in test],
        counts={
            label: sum(r["split"] == label for r in rows) for label in ["train", "val", "test"]
        },
        result=result,
        identity=identity,
    )
    save(completed, entry)
    return entry


def report(out, registry):
    out = Path(out)
    grouped = {
        name: [r for r in registry if r["setting"] == name and r.get("entry")]
        for name, _, _ in SETTINGS
    }
    pooled, spread = {}, {}
    for name, values in grouped.items():
        if values:
            raw = sum(
                np.asarray(r["entry"]["result"]["overall"]["semantic"]["confusion"], dtype=np.int64)
                for r in values
            )
            pooled[name] = metrics.segmentation(raw)
            spread[name] = float(
                np.std([r["entry"]["result"]["overall"]["semantic"]["miou"] for r in values])
            )
    baseline = pooled.get("stratified")
    table7 = []
    for name, block, buffer in SETTINGS:
        summary = pooled.get(name)
        score = summary["miou"] if summary else None
        reference = baseline["miou"] if baseline else None
        change = score - reference if score is not None and reference is not None else None
        values = grouped[name]
        table7.append(
            dict(
                Evaluation_setting=name,
                Blocking_size_km=block,
                Buffer_distance_km=buffer,
                Test_tiles=sum(r["entry"]["counts"]["test"] for r in values),
                Unique_test_tiles=len(
                    {sid for r in values for sid in r["entry"].get("sample_ids", [])}
                ),
                Train_tiles=sum(r["entry"]["counts"]["train"] for r in values),
                Validation_tiles=sum(r["entry"]["counts"]["val"] for r in values),
                Repeats=len(values),
                Semantic_mIoU=score,
                Absolute_change=change,
                Relative_change_percent=ratio(change, reference, 100),
                Split_mIoU_SD=spread.get(name),
                Pooling="pixel counts across completed splits",
                Status="complete" if values else "unavailable",
            )
        )
    tables.emit(out, 7, table7)
    table9, reconciliation = [], {}
    for index, class_name in enumerate(CLASSES):
        reference = baseline["classes"][index]["IoU"] if baseline else None
        row = dict(Class=class_name, Baseline_IoU=reference)
        for name, _, _ in SETTINGS[1:]:
            value = pooled[name]["classes"][index]["IoU"] if name in pooled else None
            change = value - reference if value is not None and reference is not None else None
            row.update(
                {
                    f"{name}_IoU": value,
                    f"{name}_absolute_change": change,
                    f"{name}_relative_change_percent": ratio(change, reference, 100),
                }
            )
        table9.append(row)
    tables.emit(out, 9, table9)
    for name, summary in pooled.items():
        values = [r["IoU"] for r in summary["classes"]]
        if any(v is None for v in values):
            raise ValueError("All nine classes must have defined unions for macro reconciliation")
        reconstructed = float(np.mean(values))
        if not np.isclose(reconstructed, summary["miou"], rtol=0, atol=1e-12):
            raise AssertionError("Tables 7 and 9 disagree")
        reconciliation[name] = dict(
            reconstructed=reconstructed, reported=summary["miou"], passed=True
        )
    save(out / "geographic_pooled_confusions.json", pooled)
    save(out / "geographic_reconciliation.json", reconciliation)
    if baseline:
        cities = sorted(
            {name for r in grouped["stratified"] for name in r["entry"]["result"]["cities"]}
        )
        city_rows = []
        for name in cities:
            values = [
                r["entry"]["result"]["cities"][name]
                for r in grouped["stratified"]
                if name in r["entry"]["result"]["cities"]
            ]
            summary = metrics.segmentation(
                sum(np.asarray(r["semantic"]["confusion"]) for r in values)
            )
            city_rows.append(
                dict(
                    City=name,
                    Tiles=sum(r["tiles"] for r in values),
                    mIoU=summary["miou"],
                    Pixel_accuracy=summary["pixel_accuracy"],
                )
            )
        tables.emit(
            out,
            2,
            city_rows
            + [
                dict(
                    City="Overall",
                    Tiles=table7[0]["Test_tiles"],
                    mIoU=baseline["miou"],
                    Pixel_accuracy=baseline["pixel_accuracy"],
                )
            ],
        )
    return pooled


def run(rows, args):
    out = Path(args.output) / "generalization"
    out.mkdir(parents=True, exist_ok=True)
    save(out / "input_records.json", rows)
    save(
        out / "settings.json",
        dict(
            seeds=args.seeds,
            crs=args.crs,
            settings=SETTINGS,
            baseline="independently trained stratified splits; no borrowed baseline values",
        ),
    )
    registry, baselines = [], {}
    for seed in args.seeds:
        for setting, block, buffer in SETTINGS:
            folder = out / f"{setting}_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            entry = dict(setting=setting, seed=seed, block_km=block, buffer_km=buffer)
            try:
                records, geometry = assignments(rows, seed, block, buffer, args.crs)
            except ValueError as error:
                entry.update(status="unavailable", reason=str(error))
            else:
                save(folder / "splits.json", {k: v for k, v in geometry.items() if k != "blocks"})
                save(folder / "blocks.geojson", geometry["blocks"])
                entry["entry"] = fit(records, args, folder, dict(setting=setting, seed=seed))
                entry["status"] = "complete"
                if setting == "stratified":
                    baselines[seed] = entry["entry"]
            registry.append(entry)
            save(out / "registry.json", registry)
    report(out, registry)
    transfers = []
    for seed, baseline in baselines.items():
        for target in sorted({city(r) for r in rows}):
            ids = {r["tile_id"] for r in rows if city(r) == target} & set(baseline["sample_ids"])
            if not ids:
                continue
            records = transfer_assignments(rows, target, ids, seed)
            result = fit(
                records,
                args,
                out / f"transfer_{seed}" / target,
                dict(setting="leave_one_city_out", city=target, seed=seed),
            )
            if set(result["sample_ids"]) != ids:
                raise AssertionError("Transfer sample identifiers differ from the city baseline")
            transfers.append(dict(city=target, seed=seed, baseline=baseline, transfer=result))
            save(out / "transfers.json", transfers)
    table8 = []
    for target in sorted({city(r) for r in rows}):
        values = [r for r in transfers if r["city"] == target]
        if not values:
            table8.append(
                dict(
                    Target_city=target,
                    Stratified_mIoU=None,
                    Transfer_mIoU=None,
                    Transfer_index=None,
                    Performance_difference=None,
                    Test_tiles=0,
                    Repeats=0,
                    Status="unavailable",
                )
            )
            continue
        base_raw = sum(
            np.asarray(r["baseline"]["result"]["cities"][target]["semantic"]["confusion"])
            for r in values
        )
        transfer_raw = sum(
            np.asarray(r["transfer"]["result"]["overall"]["semantic"]["confusion"]) for r in values
        )
        base = metrics.segmentation(base_raw)["miou"]
        transferred = metrics.segmentation(transfer_raw)["miou"]
        table8.append(
            dict(
                Target_city=target,
                Stratified_mIoU=base,
                Transfer_mIoU=transferred,
                Transfer_index=ratio(transferred, base),
                Performance_difference=transferred - base,
                Test_tiles=sum(r["transfer"]["counts"]["test"] for r in values),
                Repeats=len(values),
                Status="complete",
            )
        )
    if table8:
        tables.emit(out, 8, table8)
    save(out / "complete.json", dict(spatial=registry, transfer=transfers))
    return out
