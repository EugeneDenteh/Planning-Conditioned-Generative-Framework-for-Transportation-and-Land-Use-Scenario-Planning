import argparse
import csv
import gc
import json
from pathlib import Path
import re
import shutil
from copy import copy

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src import core, metrics, tables
from src.config import IDS, NAMES, validate
from src.inference import Evaluator, satellite_images, semantic_maps
from src.records import Records, read, save, seed, walk_target, density_target


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--split", default="test")
    p.add_argument("--stage1")
    p.add_argument("--stage2")
    p.add_argument("--experiments")
    p.add_argument("--output", required=True)
    p.add_argument("--evaluator")
    p.add_argument("--annotations")
    p.add_argument("--distribution", action="store_true")
    p.add_argument("--density-observations")
    p.add_argument("--require-full-coverage", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def shares(row):
    values = []
    for i in IDS:
        match = re.search(
            r"\b" + NAMES[i] + r"=([0-9.]+)%",
            row["prompt_structured"].replace("water_wetland=", "water="),
        )
        key = "share_" + NAMES[i]
        if NAMES[i] == "water" and key not in row:
            key = "share_water_wetland"
        value = float(match[1]) / 100 if match else float(row[key])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid requested proportion")
        values.append(value)
    return np.array(values)


def density_observations(path, variant, rows):
    if not path:
        return {}
    observations = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["variant"], row["tile_id"]
            value = float(row["estimate"])
            if key in observations or not np.isfinite(value) or value < 0:
                raise ValueError("Invalid or duplicate density observation")
            observations[key] = value
    expected = {r["tile_id"] for r in rows}
    selected = {tile: value for (name, tile), value in observations.items() if name == variant and tile in expected}
    if set(selected) != expected:
        raise ValueError("Density observations must cover every evaluated tile for this variant")
    return selected


def summarize(items):
    semantic = metrics.segmentation(sum(r["semantic_raw"] for r in items))
    aligned = [r["alignment_raw"] for r in items if r["alignment_raw"] is not None]
    calibrated = [r["calibration_raw"] for r in items if r["calibration_raw"] is not None]
    rgb = [r["rgb"] for r in items if r["rgb"] is not None]
    paired_shares = all("requested_shares" in r and "achieved_shares" in r for r in items)
    density = [r for r in items if r.get("achieved_density") is not None and r.get("requested_density") is not None]
    if aligned:
        for r in items:
            if r["alignment_raw"] is not None and np.asarray(r["alignment_raw"]).sum() != np.asarray(r["semantic_raw"]).sum():
                raise ValueError("Semantic and image-map evaluations must use the same reference-valid pixels")
    return dict(
        tiles=len(items),
        semantic=semantic,
        image_map=metrics.segmentation(sum(aligned)) if aligned else None,
        evaluator_calibration=metrics.segmentation(sum(calibrated)) if calibrated else None,
        walkability=metrics.regression(
            [r["requested_walkability"] for r in items], [r["achieved_walkability"] for r in items]
        ),
        share_mae_pp=float(np.mean([r["share_error_pp"] for r in items])),
        pooled_share_mae_pp=(
            float(np.abs(np.mean([r["achieved_shares"] for r in items], axis=0) - np.mean([r["requested_shares"] for r in items], axis=0)).mean() * 100)
            if paired_shares else None
        ),
        density_mae=(float(np.mean([abs(r["achieved_density"] - r["requested_density"]) for r in density])) if density else None),
        fixed_feature_preservation=metrics.preserved(sum(r["fixed"] for r in items)),
        images=(
            {
                k: (
                    float(np.mean([r[k] for r in rgb if r[k] is not None]))
                    if any(r[k] is not None for r in rgb)
                    else None
                )
                for k in ["MSE", "MAE", "PSNR", "SSIM"]
            }
            if rgb
            else {}
        ),
    )


@torch.inference_mode()
def score(rows, args, first, second, out, semantic_source=None):
    if second and getattr(args, "distribution", False) and len(rows) <= 200:
        raise ValueError("Distribution metrics require a larger cohort")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    state = torch.load(first, map_location="cpu", weights_only=False)
    c, stats = state["config"], core.WalkStats(**state["walk_stats"])
    validate(c)
    del state
    seed(c["seed"])
    if semantic_source:
        (out / "semantic").mkdir(exist_ok=True)
        for row in rows:
            shutil.copy2(
                Path(semantic_source) / "semantic" / f"{row['tile_id']}.png",
                out / "semantic" / f"{row['tile_id']}.png",
            )
    else:
        semantic_maps(rows, args.root, first, out, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if second:
        satellite_images(rows, args.root, second, c, out, device)
    evaluator = (
        Evaluator(args.evaluator, device, c["num_classes"]) if args.evaluator and second else None
    )
    if second and not evaluator and not args.annotations:
        raise ValueError("An independent evaluator or annotation folder is required")
    samples = Records(rows, args.root, dict(c, variant="full"), 0)
    density_estimates = density_observations(getattr(args, "density_observations", None), getattr(args, "evaluation_variant", "full"), rows)
    references = out / "references"
    if second:
        references.mkdir(exist_ok=True)
    items = []
    for row, sample in zip(rows, samples):
        sid = row["tile_id"]
        pred = np.array(Image.open(out / "semantic" / f"{sid}.png"), dtype=np.int64)
        truth = sample["semantic"].numpy()
        valid = truth != c["ignore_index"]
        if getattr(args, "require_full_coverage", False) and not valid.all():
            raise ValueError("Full-coverage evaluation requires a valid reference class at every physical pixel")
        raw = metrics.confusion(truth, pred, c["num_classes"])
        prob = (
            F.one_hot(torch.tensor(pred, device=device), c["num_classes"])
            .permute(2, 0, 1)[None]
            .float()
        )
        density, mix = core.semantic_indicators(
            prob, c["road_class"], c["landuse_classes"], c["tile_area_km2"]
        )
        achieved = float(core.walkability_1_20(density, mix, stats).item())
        actual = np.bincount(pred.ravel(), minlength=c["num_classes"])[IDS] / pred.size
        fixed = metrics.preservation(sample["control"].argmax(0).numpy(), pred, valid_mask=valid)
        rgb, alignment, calibration = None, None, None
        if second:
            reference = (
                Image.open(samples.path(row["satellite_path"]))
                .convert("RGB")
                .resize((c["resolution"], c["resolution"]), Image.Resampling.BILINEAR)
            )
            reference.save(references / f"{sid}.png")
            generated = Image.open(out / "generated" / f"{sid}.png").convert("RGB")
            if generated.size != reference.size:
                raise ValueError("RGB grids differ")
            rgb = metrics.rgb_metrics(generated, reference)
            if evaluator:
                labels = evaluator(generated)
                calibration = metrics.confusion(truth, evaluator(reference), c["num_classes"])
            else:
                labels = np.asarray(
                    Image.open(Path(args.annotations) / f"{sid}.png"), dtype=np.int64
                )
            alignment = metrics.confusion(pred, labels, c["num_classes"], valid_mask=valid)
        items.append(
            dict(
                tile_id=sid,
                city=row.get("city") or row.get("prefix") or "Unspecified",
                semantic_raw=raw,
                alignment_raw=alignment,
                calibration_raw=calibration,
                requested_walkability=walk_target(row),
                achieved_walkability=achieved,
                requested_shares=shares(row).tolist(),
                achieved_shares=actual.tolist(),
                share_error_pp=float(np.abs(actual - shares(row)).mean() * 100),
                fixed=fixed,
                rgb=rgb,
            )
        )
        if density_estimates:
            items[-1].update(requested_density=density_target(row), achieved_density=density_estimates[sid])
    overall = summarize(items)
    cities = {
        city: summarize([r for r in items if r["city"] == city])
        for city in sorted({r["city"] for r in items})
    }
    if second and getattr(args, "distribution", False):
        overall["images"].update(
            metrics.distribution(out / "generated", references, device, c["seed"])
        )
    targets = np.array([r["requested_walkability"] for r in items])
    result = dict(
        overall=overall,
        cities=cities,
        requested_walkability=dict(
            min=float(targets.min()),
            max=float(targets.max()),
            mean=float(targets.mean()),
            std=float(targets.std()),
            quantiles=np.quantile(targets, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        ),
        pixel_accounting=dict(
            physical_pixels=len(items) * c["resolution"] ** 2,
            scored_pixels=overall["semantic"]["scored_pixels"],
            image_map_scored_pixels=overall["image_map"]["scored_pixels"] if overall["image_map"] else None,
            evaluation_mask="reference semantic labels != ignore_index for semantic and image-map comparisons",
            ignored_targets=len(items) * c["resolution"] ** 2
            - overall["semantic"]["scored_pixels"],
        ),
        observations=[
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r.items()
                if k not in ["semantic_raw", "alignment_raw", "calibration_raw", "fixed"]
            }
            for r in items
        ],
    )
    save(out / "results.json", result)
    tables.primary(out, result)
    return result


def run(args):
    rows = read(args.records, args.split)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.experiments:
        if not args.stage1:
            raise ValueError("--stage1 is required")
        return score(rows, args, args.stage1, args.stage2, out)
    entries = json.loads(Path(args.experiments).read_text())
    required = [
        "full",
        "without_text",
        "without_control",
        "without_population",
        "without_walkability_condition",
        "without_city",
        "without_walkability_loss",
        "without_semantics",
        "semantic_only",
        "without_dice",
        "without_perceptual",
        "without_reconstruction",
    ]
    if any(name not in entries for name in required):
        raise ValueError("Incomplete experiment configuration")
    results = {}
    for name in required:
        entry = entries[name]
        shared = (
            out / "full"
            if name != "full"
            and Path(entry["stage1"]).resolve() == Path(entries["full"]["stage1"]).resolve()
            else None
        )
        options = copy(args)
        options.evaluation_variant = name
        if args.annotations:
            options.annotations = str(Path(args.annotations) / name)
        results[name] = score(
            rows, options, entry["stage1"], entry.get("stage2"), out / name, shared
        )
        results[name]["changed"] = entry.get("input_changed", True)
    tables.primary(out, results["full"])
    tables.ablations(out, results)
    save(out / "experiments.json", results)
    return results


if __name__ == "__main__":
    run(arguments().parse_args())
