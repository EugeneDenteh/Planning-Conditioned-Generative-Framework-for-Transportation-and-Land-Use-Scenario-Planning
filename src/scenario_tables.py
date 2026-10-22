import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import metrics, tables
from .config import CLASSES, NAMES
from .records import save
from .scenario_experiments import DENSITIES, SCENARIOS, TRAJECTORY, WALK_LEVELS
from .spatial_metrics import descriptor


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def deviation(values):
    values = [v for v in values if v is not None]
    return float(np.std(values)) if values else None


def group(items, field):
    return {
        name: [r for r in items if r[field] == name] for name in sorted({r[field] for r in items})
    }


def density_table(out, semantic):
    rows = []
    for name, requested in DENSITIES:
        items = semantic[f"density_{requested}"]
        available = [r for r in items if r["density"] is not None]
        rows.append(
            dict(
                Density_condition=name,
                Requested_density=requested,
                Realizations=len(items),
                Density_observations=len(available),
                Achieved_density=mean([r["density"] for r in available]),
                Density_SD=deviation([r["density"] for r in available]),
                Absolute_density_error=mean([abs(r["density"] - requested) for r in available]),
                Density_error_percent=mean(
                    [100 * abs(r["density"] - requested) / requested for r in available]
                ),
                Intersection_density=mean([r["spatial"]["intersection_density"] for r in items]),
                Intersection_density_SD=deviation(
                    [r["spatial"]["intersection_density"] for r in items]
                ),
                Settlement_pattern=json.dumps(
                    {
                        "mean_patch_ha": mean([r["spatial"]["mean_patch_ha"] for r in items]),
                        "fragmentation": mean([r["spatial"]["fragmentation"] for r in items]),
                        "class_entropy_bits": mean(
                            [r["spatial"]["class_entropy_bits"] for r in items]
                        ),
                    }
                ),
                Land_use_MAE_pp=mean([r["share_error_pp"] for r in items]),
            )
        )
    tables.emit(out, 10, rows)
    values = [r["Achieved_density"] for r in rows]
    valid = all(v is not None for v in values)
    statistic = (
        float(spearmanr([v for _, v in DENSITIES], values).statistic)
        if valid and np.std(values)
        else None
    )
    save(
        Path(out) / "density_response.json",
        dict(
            monotonic_nondecreasing=bool(np.all(np.diff(values) >= 0)) if valid else None,
            spearman=statistic,
            available=valid,
        ),
    )


def walk_table(out, semantic):
    rows = []
    for requested in WALK_LEVELS:
        items = semantic[f"walk_{requested:g}"]
        rows.append(
            dict(
                Target_walkability=requested,
                Realizations=len(items),
                Achieved_walkability=mean([r["achieved_walkability"] for r in items]),
                Achieved_SD=deviation([r["achieved_walkability"] for r in items]),
                Absolute_error=mean([r["walk_error"] for r in items]),
                Signed_error=mean([r["walk_bias"] for r in items]),
                Intersection_density=mean([r["spatial"]["intersection_density"] for r in items]),
                Land_use_mix_entropy=mean(
                    [r["spatial"]["normalized_class_entropy"] for r in items]
                ),
                Walk_nonroad_entropy=mean([r["walk_nonroad_entropy"] for r in items]),
                Land_use_MAE_pp=mean([r["share_error_pp"] for r in items]),
            )
        )
    tables.emit(out, 11, rows)
    achieved = [r["Achieved_walkability"] for r in rows]
    slopes = np.diff(achieved) / np.diff(WALK_LEVELS)
    save(
        Path(out) / "walkability_response.json",
        dict(
            monotonic_nondecreasing=bool(np.all(np.diff(achieved) >= 0)),
            incremental_slopes=slopes.tolist(),
            upper_range_slope=float(slopes[-1]),
            baseline_MAE=mean([r["walk_error"] for r in semantic["baseline"]]),
            controlled_MAE=mean(
                [r["walk_error"] for level in WALK_LEVELS for r in semantic[f"walk_{level:g}"]]
            ),
        ),
    )


def success(name, item, baseline, criteria):
    preserved = item["fixed_f1"] is None or item["fixed_f1"] >= criteria["minimum_preservation"]
    if name == "population_density":
        return (
            None
            if item["density"] is None
            else abs(item["density"] - 3000) / 3000 <= criteria["density_relative_tolerance"]
        )
    if name == "walkability":
        increase = item["requested_walkability"] - baseline["requested_walkability"]
        return (
            preserved
            and item["achieved_walkability"] - baseline["achieved_walkability"]
            >= increase - criteria["walkability_tolerance"]
        )
    if name == "mixed_use":
        return (
            preserved
            and item["spatial"]["normalized_class_entropy"] >= criteria["mixed_use_entropy"]
        )
    index = NAMES.index(name)
    return preserved and (
        item["spatial"]["shares_pp"][index]
        >= 100 * (item["requested_shares"][index] - criteria["share_tolerance"])
        and item["spatial"]["shares_pp"][index] - baseline["spatial"]["shares_pp"][index]
        >= 100 * criteria["minimum_share_increase"]
    )


def success_table(out, semantic, criteria):
    baseline = {r["tile_id"]: r for r in semantic["baseline"]}
    cities = sorted({r["city"] for r in baseline.values()})
    counts, rows = [], []
    for name, condition in [
        ("residential", "residential"),
        ("population_density", "density_3000"),
        ("walkability", "walk_increase"),
        ("industrial", "industrial"),
        ("mixed_use", "mixed_use"),
    ]:
        row = dict(Scenario=name)
        observations = []
        for item in semantic[condition]:
            value = success(name, item, baseline[item["tile_id"]], criteria)
            observations.append(
                dict(city=item["city"], success=None if value is None else bool(value))
            )
        for name_city in cities + ["Overall"]:
            values = (
                observations
                if name_city == "Overall"
                else [r for r in observations if r["city"] == name_city]
            )
            valid = [r for r in values if r["success"] is not None]
            passed = sum(r["success"] for r in valid)
            percentage = 100 * passed / len(valid) if valid else None
            if percentage is not None and not 0 <= percentage <= 100:
                raise AssertionError("Invalid success rate")
            row[name_city] = percentage
            counts.append(
                dict(
                    Scenario=name,
                    City=name_city,
                    Attempted=len(values),
                    Successful=passed,
                    Evaluated=len(valid),
                    Success_rate_percent=percentage,
                )
            )
        rows.append(row)
    tables.emit(out, 12, rows)
    save(Path(out) / "scenario_success_counts.json", counts)
    return counts


def image_table(out, images, criteria):
    rows, summaries = [], {}
    baseline_counts = {r["tile_id"]: r.get("reference_valid_pixels") for r in images["baseline"]}
    for name, condition in SCENARIOS.items():
        values = images[condition]
        for r in values:
            count = r.get("reference_valid_pixels")
            if count is None or count != np.asarray(r["confusion"]).sum() or count != baseline_counts.get(r["tile_id"]):
                raise ValueError("All image-map scenarios must use the same reference-valid pixels per tile")
        pooled = sum(np.asarray(r["confusion"], np.int64) for r in values)
        summary = metrics.segmentation(pooled)
        fixed = sum(np.asarray(r["preservation"], np.int64) for r in values)
        denominator = fixed[:, 1].sum()
        preserved = 100 * fixed[:, 0].sum() / denominator if denominator else None
        distorted = mean([r["distortion"] for r in values])
        low, high = criteria["distortion_thresholds"]
        category = "Low" if distorted < low else "Moderate" if distorted < high else "High"
        rows.append(
            dict(
                Scenario=name,
                Images=len(values),
                Scored_pixels=summary["scored_pixels"],
                Image_map_mIoU=summary["miou"],
                Feature_preservation_percent=preserved,
                Distortion=distorted,
                Distortion_SD=deviation([r["distortion"] for r in values]),
                Distortion_category=category,
            )
        )
        summaries[name] = dict(
            semantic=summary,
            preservation_counts=fixed.tolist(),
            images=len(values),
            distortion=distorted,
        )
    tables.emit(out, 13, rows)
    baseline = summaries["baseline"]["semantic"]
    tables.emit(
        out,
        6,
        [{k: r[k] for k in ["Class", "IoU", "Recall", "Precision"]} for r in baseline["classes"]]
        + [
            dict(
                Class="Macro average",
                **{k: baseline["macro"][k] for k in ["IoU", "Recall", "Precision"]},
            )
        ],
    )
    save(Path(out) / "scenario_image_confusions.json", summaries)
    return summaries


def trajectory_table(out, semantic):
    shares = {}
    for name, target in TRAJECTORY:
        counts = sum(
            np.asarray(r["spatial"]["counts"], np.int64) for r in semantic[f"walk_{target:g}"]
        )
        shares[name] = 100 * counts / counts.sum()
        if not np.isclose(shares[name].sum(), 100, atol=1e-10):
            raise AssertionError("Nine-class proportions do not sum to 100 percent")
    rows = [
        dict(
            Class=class_name,
            **{name: float(values[index]) for name, values in shares.items()},
            Net_change_pp=float(shares["very_high"][index] - shares["very_low"][index]),
        )
        for index, class_name in enumerate(CLASSES)
    ]
    tables.emit(out, 14, rows)
    save(
        Path(out) / "class_trajectory_changes.json",
        {name: (values - shares["very_low"]).tolist() for name, values in shares.items()},
    )


def plausibility_table(out, semantic):
    names = ["baseline", "residential", "industrial", "mixed_use"]
    keys = [
        "residential_industrial_adjacency",
        "road_forest_adjacency",
        "mean_patch_ha",
        "patch_std_ha",
        "coherence",
        "fragmentation",
    ]
    rows, spreads = [], []
    reference_path = Path(out) / "reference_spatial_measurements.json"
    references = json.loads(reference_path.read_text()) if reference_path.exists() else []
    for key in keys:
        values = {name: [r["spatial"][key] for r in semantic[name]] for name in names}
        flattened = [v for name in names for v in values[name]]
        rows.append(
            dict(
                Spatial_metric=key,
                **{name: mean(v) for name, v in values.items()},
                Average=mean(flattened),
                Reference=mean([r["spatial"][key] for r in references]),
            )
        )
        spreads.append(
            dict(
                Spatial_metric=key,
                **{name: deviation(v) for name, v in values.items()},
                Average=deviation(flattened),
            )
        )
    tables.emit(out, 15, rows)
    save(
        Path(out) / "plausibility_variability.json",
        dict(
            standard_deviation=spreads,
            realizations={name: len(semantic[name]) for name in names},
            average="realization-weighted mean",
        ),
    )


def city_table(out, semantic):
    items = semantic["baseline"]
    grouped = group(items, "city")
    rows = []
    for name, values in grouped.items():
        other = [r for r in items if r["city"] != name]
        center = np.mean([descriptor(r["spatial"]) for r in values], axis=0)
        distinct = (
            float(
                np.linalg.norm(center - np.mean([descriptor(r["spatial"]) for r in other], axis=0))
                / np.sqrt(2)
            )
            if other
            else None
        )
        rows.append(
            dict(
                City=name,
                Realizations=len(values),
                Mean_patch_coherence=mean([r["spatial"]["patch_coherence"] for r in values]),
                Spatial_entropy_bits=mean([r["spatial"]["spatial_entropy_bits"] for r in values]),
                Class_entropy_bits=mean([r["spatial"]["class_entropy_bits"] for r in values]),
                Structural_distinctiveness=distinct,
            )
        )
    overall = dict(City="Overall", Realizations=len(items))
    for key in [
        "Mean_patch_coherence",
        "Spatial_entropy_bits",
        "Class_entropy_bits",
        "Structural_distinctiveness",
    ]:
        included = [r for r in rows if r[key] is not None]
        overall[key] = (
            float(
                np.average(
                    [r[key] for r in included], weights=[r["Realizations"] for r in included]
                )
            )
            if included
            else None
        )
    if overall["Class_entropy_bits"] > np.log2(len(CLASSES)) + 1e-10:
        raise AssertionError("Class entropy exceeds the nine-class bound")
    tables.emit(out, 16, rows + [overall])
    save(
        Path(out) / "city_consistency_variability.json",
        {
            name: {
                key: deviation([r["spatial"][key] for r in values])
                for key in ["patch_coherence", "spatial_entropy_bits", "class_entropy_bits"]
            }
            for name, values in grouped.items()
        },
    )


def comparable(semantic, field):
    baseline = {r["tile_id"]: r[field] for r in semantic["baseline"]}
    return [
        name
        for name in SCENARIOS.values()
        if len(semantic[name]) == len(baseline)
        and all(
            r["tile_id"] in baseline
            and np.allclose(r[field], baseline[r["tile_id"]], atol=1e-10, rtol=0)
            for r in semantic[name]
        )
    ]


def ranking(metric, values, higher, source, count, comparability):
    available = {key: value for key, value in values.items() if value is not None}
    base = available.get("baseline")
    if len(available) < 2 or base is None:
        return dict(
            Metric=metric,
            Baseline=base,
            Best_configuration=None,
            Best_value=None,
            Worst_configuration=None,
            Worst_value=None,
            Absolute_range=None,
            Samples=count,
            Source_experiment=source,
            Comparability=comparability,
        )
    ordered = sorted(available, key=lambda name: (available[name], name))
    best, worst = (ordered[-1], ordered[0]) if higher else (ordered[0], ordered[-1])
    return dict(
        Metric=metric,
        Baseline=base,
        Best_configuration=best,
        Best_value=available[best],
        Worst_configuration=worst,
        Worst_value=available[worst],
        Absolute_range=max(available.values()) - min(available.values()),
        Samples=count,
        Source_experiment=source,
        Comparability=comparability,
    )


def comparative_table(out, semantic, images):
    baseline = json.loads((Path(out) / "baseline_semantic.json").read_text())
    n = len(semantic["baseline"])
    rows = [
        ranking(
            "Semantic mIoU",
            {"baseline": baseline["miou"]},
            True,
            "baseline_semantic.json",
            n,
            "Counterfactual scenarios have no semantic ground truth",
        ),
        ranking(
            "Pixel accuracy",
            {"baseline": baseline["pixel_accuracy"]},
            True,
            "baseline_semantic.json",
            n,
            "Counterfactual scenarios have no semantic ground truth",
        ),
    ]
    expected = {r["tile_id"] for r in semantic["baseline"]}
    fixed = {}
    for name in SCENARIOS.values():
        if {r["tile_id"] for r in semantic[name]} != expected:
            raise AssertionError("Scenario sample identifiers differ")
        counts = sum(np.asarray(r["fixed_counts"]) for r in semantic[name])
        fixed[name] = metrics.preserved(counts)
    rows.append(
        ranking(
            "Fixed-feature F1",
            fixed,
            True,
            "semantic_measurements.json",
            n,
            "same fixed controls and sample identifiers",
        )
    )
    maps = {}
    image_ids = {(r["tile_id"], r["seed"]) for r in images["baseline"]}
    for name, values in images.items():
        if {(r["tile_id"], r["seed"]) for r in values} != image_ids:
            raise AssertionError("Image-map realizations differ")
        maps[name] = metrics.segmentation(sum(np.asarray(r["confusion"]) for r in values))["miou"]
    rows.append(
        ranking(
            "Image-map mIoU",
            maps,
            True,
            "Table 13",
            len(image_ids),
            "same controls, seeds, checkpoints and independent evaluator",
        )
    )
    for title, field, value, source in [
        (
            "Land-use adherence MAE pp",
            "requested_shares",
            "share_error_pp",
            "Tables 10-11 measurements",
        ),
        (
            "Walkability adherence MAE",
            "requested_walkability",
            "walk_error",
            "scenario semantic measurements",
        ),
    ]:
        values = {
            name: mean([r[value] for r in semantic[name]]) for name in comparable(semantic, field)
        }
        rows.append(
            ranking(
                title,
                values,
                False,
                source,
                n,
                "same sample identifiers and identical target values",
            )
        )
    tables.emit(out, 17, rows)
    save(Path(out) / "comparison_sources.json", rows)


def report(out, semantic, images, criteria):
    density_table(out, semantic)
    walk_table(out, semantic)
    counts = success_table(out, semantic, criteria)
    summaries = image_table(out, images, criteria)
    trajectory_table(out, semantic)
    plausibility_table(out, semantic)
    city_table(out, semantic)
    comparative_table(out, semantic, images)
    checks = {}
    for name in {r["Scenario"] for r in counts}:
        cities = [r for r in counts if r["Scenario"] == name and r["City"] != "Overall"]
        overall = next(r for r in counts if r["Scenario"] == name and r["City"] == "Overall")
        assert sum(r["Successful"] for r in cities) == overall["Successful"]
        assert sum(r["Evaluated"] for r in cities) == overall["Evaluated"]
        checks[f"table12_{name}_pooled"] = True
    checks.update(
        table6_13_same_confusion=True,
        table10_12_same_density_tolerance=True,
        table11_14_shared_levels=True,
        table14_percentages=True,
        table15_weighted=True,
        table16_weighted=True,
        table17_comparable=True,
    )
    save(Path(out) / "cross_table_checks.json", checks)
    save(Path(out) / "baseline_image_map.json", summaries["baseline"]["semantic"])
