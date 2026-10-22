import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from . import core, metrics, spatial_metrics
from .config import NAMES, ROAD_ID, WATER_ID, FIXED_IDS, EDITABLE_IDS, validate
from .geography import area_km2, city, digest
from .inference import Evaluator, semantic_maps, satellite_images
from .records import Records, save, walk_target


DENSITIES = [
    ("very_low", 300),
    ("low", 800),
    ("moderate", 1500),
    ("high", 3000),
    ("very_high", 5000),
]
WALK_LEVELS = [1.5, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0]
TRAJECTORY = [
    ("very_low", 1.5),
    ("low", 4.0),
    ("moderate", 10.0),
    ("high", 16.0),
    ("very_high", 19.0),
]
SCENARIOS = {
    "baseline": "baseline",
    "residential": "residential",
    "industrial": "industrial",
    "population_dense": "density_3000",
    "population_sparse": "density_300",
    "high_walkability": "walk_19",
    "low_walkability": "walk_1.5",
    "mixed_use": "mixed_use",
}
CRITERIA = {
    "share_increase": 0.10,
    "share_tolerance": 0.02,
    "minimum_share_increase": 0.05,
    "density_relative_tolerance": 0.15,
    "walkability_increase": 3.0,
    "walkability_tolerance": 0.5,
    "minimum_preservation": 0.95,
    "mixed_use_entropy": 0.8,
    "distortion_thresholds": [0.01, 0.05],
}


def fields(text):
    return {
        part.partition("=")[0].strip(): part.partition("=")[2].strip()
        for part in text.replace("water_wetland=", "water=").split(";")
        if "=" in part
    }


def change(row, condition, criteria):
    import evaluate

    result = dict(row)
    original = row["prompt_structured"].replace("water_wetland=", "water=")
    values = fields(original)
    if condition.startswith("density_"):
        values["population_density"] = f"{float(condition.split('_')[1]):g} persons/km2"
    elif condition == "walk_increase":
        values["walkability"] = f"{min(20, walk_target(row) + criteria['walkability_increase']):g}"
    elif condition.startswith("walk_"):
        values["walkability"] = condition.split("_")[1]
    elif condition in ["residential", "industrial", "mixed_use"]:
        shares = evaluate.shares(row)
        if not np.isclose(shares.sum(), 1, atol=1e-3):
            raise ValueError("Scenario editing requires complete nine-class proportions")
        editable = EDITABLE_IDS
        remaining = 1 - shares[FIXED_IDS].sum()
        if condition == "mixed_use":
            shares[editable] = remaining / len(editable)
        else:
            selected = NAMES.index(condition)
            requested = min(remaining, shares[selected] + criteria["share_increase"])
            others = [i for i in editable if i != selected]
            weights = shares[others]
            weights = (
                weights / weights.sum() if weights.sum() else np.ones(len(others)) / len(others)
            )
            shares[others] = weights * (remaining - requested)
            shares[selected] = requested
        values.update({name: f"{100 * shares[i]:.8f}%" for i, name in enumerate(NAMES)})
    elif condition != "baseline":
        raise ValueError("Unknown scenario")
    pieces = []
    for part in original.split(";"):
        key, separator, _ = part.partition("=")
        pieces.append(
            f"{key.strip()}={values.pop(key.strip())}"
            if separator and key.strip() in values
            else part.strip()
        )
    pieces.extend(f"{key}={value}" for key, value in values.items())
    result["prompt_structured"] = "; ".join(pieces)
    return result


def density_values(path, validation):
    if not path:
        return {}, None
    if not validation:
        raise ValueError("Density observations require independent validation results")
    evidence = json.loads(Path(validation).read_text())
    if (
        not evidence.get("estimator_id")
        or not evidence.get("independent")
        or evidence.get("input_representation") != "semantic"
        or not evidence.get("validation_metrics")
    ):
        raise ValueError(
            "Density estimates require a validated independent semantic-to-density method"
        )
    if evidence.get("observations_sha256") != digest(path):
        raise ValueError("Density validation does not identify these observations")
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["condition"], row["tile_id"]
            value = float(row["estimate"])
            if key in result or not np.isfinite(value) or value < 0:
                raise ValueError("Invalid or duplicate density observation")
            result[key] = value
    return result, evidence


def preservation(first, second, valid_mask=None):
    if first.shape != second.shape:
        raise ValueError("Incompatible map and image-label grids")
    valid = np.ones(first.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if valid.shape != first.shape or valid.dtype != np.bool_:
        raise ValueError("Expected a boolean validity mask matching the semantic grid")
    counts = []
    for class_id in FIXED_IDS:
        mask = (first == class_id) & valid
        counts.append([int((mask & (second == class_id)).sum()), int(mask.sum())])
    return np.asarray(counts, np.int64)


def measure(row, labels, sample, c, stats, density):
    import evaluate

    measured = spatial_metrics.measurements(labels, area_km2(row))
    probabilities = F.one_hot(torch.as_tensor(labels).long(), len(NAMES)).permute(2, 0, 1)[None].float()
    intersection, mix = core.semantic_indicators(
        probabilities, c["road_class"], c["landuse_classes"], c["tile_area_km2"]
    )
    achieved = core.walkability_1_20(intersection, mix, stats).item()
    valid = sample["semantic"].numpy() != c["ignore_index"] if "semantic" in sample else None
    fixed = metrics.preservation(sample["control"].argmax(0).numpy(), labels, valid_mask=valid)
    requested = evaluate.shares(row)
    return dict(
        tile_id=row["tile_id"],
        city=city(row),
        prompt=row["prompt_structured"],
        spatial=measured,
        achieved_walkability=achieved,
        requested_walkability=walk_target(row),
        walk_error=abs(achieved - walk_target(row)),
        walk_bias=achieved - walk_target(row),
        walk_nonroad_entropy=float(mix.item()),
        walk_junction_proxy=float(intersection.item()),
        requested_shares=requested.tolist(),
        share_error_pp=float(np.abs(np.asarray(measured["shares_pp"]) - 100 * requested).mean()),
        fixed_counts=fixed.tolist(),
        fixed_f1=metrics.preserved(fixed),
        density=density,
    )


def run(rows, args):
    from .scenario_tables import report

    if not args.stage1:
        raise ValueError("Scenario experiments require --stage1")
    if not args.stage2 or not args.evaluator:
        raise ValueError(
            "The full scenario suite requires --stage2 and a fixed independent --evaluator"
        )
    out = Path(args.output) / "scenarios"
    out.mkdir(parents=True, exist_ok=True)
    first_state = torch.load(args.stage1, map_location="cpu", weights_only=False)
    c, stats = first_state["config"], core.WalkStats(**first_state["walk_stats"])
    validate(c)
    if c["resolution"] != 512:
        raise ValueError("These experiments require 512-pixel images")
    second_state = torch.load(args.stage2, map_location="cpu", weights_only=False)
    second_config = second_state["config"]
    validate(second_config)
    if second_config["semantic_palette"] != c["semantic_palette"]:
        raise ValueError("The stage palettes differ")
    del first_state, second_state
    selected = [r for r in rows if r.get("split") == args.split]
    if not selected:
        raise ValueError("No scenario inputs in the requested split")
    for row in selected:
        area_km2(row)
    estimates, evidence = density_values(args.density_values, args.density_validation)
    criteria = dict(CRITERIA)
    if args.criteria:
        criteria.update(json.loads(Path(args.criteria).read_text()))
    if (
        any(
            not np.isfinite(v) or not 0 <= v <= 1
            for key, v in criteria.items()
            if key not in ["distortion_thresholds", "walkability_increase", "walkability_tolerance"]
        )
        or not 0
        <= criteria["distortion_thresholds"][0]
        <= criteria["distortion_thresholds"][1]
        <= 1
        or criteria["walkability_increase"] <= 0
        or criteria["walkability_tolerance"] < 0
    ):
        raise ValueError("Invalid predeclared scenario criteria")
    identity = dict(
        evaluation_mask="reference_semantic_valid_pixels_v1",
        require_full_coverage=getattr(args, "require_full_coverage", False),
        stage1_sha256=digest(args.stage1),
        stage2_sha256=digest(args.stage2),
        evaluator_sha256=digest(args.evaluator),
        records=selected,
        seeds=args.seeds,
        criteria=criteria,
        density=evidence,
        semantic_stochastic=False,
        semantic_realizations="one per geographic control and condition; no duplicate seed counts",
    )
    spec = out / "specification.json"
    if spec.exists() and json.loads(spec.read_text()) != identity:
        raise ValueError("Existing scenario specification differs")
    save(spec, identity)
    save(
        out / "metric_definitions.json",
        dict(
            spatial=spatial_metrics.DEFINITIONS,
            density=evidence,
            walkability="same hard semantic labels, junction proxy, frozen scaling and tile area as Table 3",
            walk_entropy="natural-log normalized entropy over eight non-road classes",
            reported_land_entropy="natural-log normalized entropy over all nine classes",
            preservation="correct fixed road/water pixels divided by all fixed road/water pixels; rail remains a scored semantic class",
            distinctiveness="Euclidean distance between city-mean class shares and the other-city mean, divided by sqrt(2)",
            ignored_truth="255 excluded from semantic losses and confusion; predictions require exactly nine classes",
            evaluation_mask="reference semantic labels != 255 for semantic and image-map confusion and fixed-class preservation",
            rgb_and_spatial_extent="complete tile for RGB, planning indicators and boundary distortion",
            macro_iou="unweighted mean over nine classes; zero union gives zero; no scored pixels gives null",
            class_normalization="nine-class pixels",
            averaging="realization-weighted unless labeled pooled counts",
            criteria=criteria,
        ),
    )
    device = torch.device(args.device)
    conditions = ["baseline", "residential", "industrial", "mixed_use", "walk_increase"]
    conditions += [f"density_{level}" for _, level in DENSITIES]
    conditions += [f"walk_{level:g}" for level in WALK_LEVELS]
    evaluator = Evaluator(args.evaluator, device, len(NAMES))
    semantic, images = {}, {}
    for condition in conditions:
        folder = out / condition
        folder.mkdir(exist_ok=True)
        altered = [change(row, condition, criteria) for row in selected]
        save(folder / "conditions.json", altered)
        semantic_maps(altered, args.root, args.stage1, folder, device)
        samples = Records(altered, args.root, c, 1)
        observations = []
        valid_masks = {}
        for row, sample in zip(altered, samples):
            valid_masks[row["tile_id"]] = sample["semantic"].numpy() != c["ignore_index"]
            if getattr(args, "require_full_coverage", False) and not valid_masks[row["tile_id"]].all():
                raise ValueError("Full-coverage evaluation requires a valid reference class at every physical pixel")
            labels = np.array(
                Image.open(folder / "semantic" / f"{row['tile_id']}.png"), dtype=np.int64
            )
            observations.append(
                measure(row, labels, sample, c, stats, estimates.get((condition, row["tile_id"])))
            )
        semantic[condition] = observations
        save(folder / "semantic_measurements.json", observations)
        if condition in SCENARIOS.values():
            results = []
            for seed in args.seeds:
                realization = folder / f"seed_{seed}"
                (realization / "semantic").mkdir(parents=True, exist_ok=True)
                for row in altered:
                    shutil.copy2(
                        folder / "semantic" / f"{row['tile_id']}.png",
                        realization / "semantic" / f"{row['tile_id']}.png",
                    )
                satellite_images(
                    altered, args.root, args.stage2, c, realization, device, seed_value=seed
                )
                (realization / "classified").mkdir(exist_ok=True)
                for row in altered:
                    sid = row["tile_id"]
                    path = realization / "generated" / f"{sid}.png"
                    with Image.open(path) as image:
                        classified = evaluator(image.convert("RGB"))
                    original = np.array(
                        Image.open(realization / "semantic" / f"{sid}.png"), dtype=np.int64
                    )
                    Image.fromarray(classified.astype(np.uint8)).save(
                        realization / "classified" / f"{sid}.png"
                    )
                    results.append(
                        dict(
                            tile_id=sid,
                            seed=seed,
                            image_sha256=digest(path),
                            condition_sha256=digest(realization / "semantic" / f"{sid}.png"),
                            reference_valid_pixels=int(valid_masks[sid].sum()),
                            confusion=metrics.confusion(original, classified, valid_mask=valid_masks[sid]).tolist(),
                            preservation=preservation(original, classified, valid_mask=valid_masks[sid]).tolist(),
                            distortion=spatial_metrics.distortion(original, classified),
                        )
                    )
            images[condition] = results
            save(folder / "image_measurements.json", results)
        save(out / "progress.json", dict(completed_conditions=list(semantic), images=list(images)))
    calibration = np.zeros((len(NAMES), len(NAMES)), np.int64)
    baseline_raw = np.zeros((len(NAMES), len(NAMES)), np.int64)
    reference_metrics = []
    for row, sample in zip(selected, Records(selected, args.root, c, 1)):
        labels = np.array(
            Image.open(out / "baseline" / "semantic" / f"{row['tile_id']}.png"), dtype=np.int64
        )
        baseline_raw += metrics.confusion(sample["semantic"].numpy(), labels)
        if (sample["semantic"].numpy() != 255).any():
            reference_metrics.append(
                dict(
                    tile_id=row["tile_id"],
                    city=city(row),
                    spatial=spatial_metrics.measurements(
                        sample["semantic"].numpy(), area_km2(row), allow_ignore=True
                    ),
                )
            )
        path = Records(selected, args.root, c, 1).path(row["satellite_path"])
        with Image.open(path) as image:
            classified = evaluator(
                image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            )
        calibration += metrics.confusion(sample["semantic"].numpy(), classified)
    save(out / "evaluator_calibration.json", metrics.segmentation(calibration))
    save(out / "baseline_semantic.json", metrics.segmentation(baseline_raw))
    save(out / "reference_spatial_measurements.json", reference_metrics)
    report(out, semantic, images, criteria)
    for key, path in [
        ("stage1_sha256", args.stage1),
        ("stage2_sha256", args.stage2),
        ("evaluator_sha256", args.evaluator),
    ]:
        if digest(path) != identity[key]:
            raise ValueError("A checkpoint changed during the experiment")
    save(
        out / "complete.json",
        dict(identity=identity, conditions=list(semantic), image_conditions=list(images)),
    )
    return out


def refresh(args):
    from .scenario_tables import report

    out = Path(args.output) / "scenarios"
    completed = json.loads((out / "complete.json").read_text())
    if completed["identity"].get("evaluation_mask") != "reference_semantic_valid_pixels_v1":
        raise ValueError("Existing image-map measurements use a legacy mask; recompute them with the reference-valid mask")
    semantic = {
        name: json.loads((out / name / "semantic_measurements.json").read_text())
        for name in completed["conditions"]
    }
    images = {
        name: json.loads((out / name / "image_measurements.json").read_text())
        for name in completed["image_conditions"]
    }
    if args.density_values:
        estimates, evidence = density_values(args.density_values, args.density_validation)
        for condition, items in semantic.items():
            for item in items:
                item["density"] = estimates.get((condition, item["tile_id"]))
            save(out / condition / "semantic_measurements.json", items)
        save(out / "density_validation.json", evidence)
    report(out, semantic, images, completed["identity"]["criteria"])
    return out
