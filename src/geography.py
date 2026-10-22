import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform, unary_union


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def city(row):
    value = row.get("city")
    if not value:
        raise ValueError("Each record requires a city")
    return str(value)


def footprint(row, target="EPSG:3035"):
    bounds = row.get("bounds")
    if isinstance(bounds, str):
        bounds = json.loads(bounds)
    source = row.get("bounds_crs") or row.get("crs")
    if not source or bounds is None or len(bounds) != 4:
        raise ValueError("Each record requires bounds and their CRS")
    left, bottom, right, top = map(float, bounds)
    if not np.isfinite([left, bottom, right, top]).all() or left >= right or bottom >= top:
        raise ValueError("Invalid geographic bounds")
    crs = CRS.from_user_input(target)
    if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-8 for a in crs.axis_info):
        raise ValueError("Use a projected CRS in meters")
    converter = Transformer.from_crs(source, crs, always_xy=True)
    fractions = np.linspace(0, 1, 17)
    points = [(left + (right - left) * v, bottom) for v in fractions]
    points += [(right, bottom + (top - bottom) * v) for v in fractions]
    points += [(right - (right - left) * v, top) for v in fractions]
    points += [(left, top - (top - bottom) * v) for v in fractions]
    geometry = transform(converter.transform, Polygon(points))
    if not geometry.is_valid or not math.isfinite(geometry.area) or geometry.area <= 0:
        raise ValueError("Invalid projected footprint")
    return geometry


def area_km2(row):
    return footprint(row).area / 1e6


def assignments(rows, seed, block_km=0, buffer_km=0, crs="EPSG:3035"):
    rng = np.random.default_rng(seed)
    shapes = {r["tile_id"]: footprint(r, crs) for r in rows}
    groups, blocks, excluded = {}, {}, {}
    for row in sorted(rows, key=lambda r: r["tile_id"]):
        sid, name = row["tile_id"], city(row)
        if block_km:
            width = block_km * 1000
            geometry = shapes[sid]
            x, y = math.floor(geometry.centroid.x / width), math.floor(geometry.centroid.y / width)
            key = f"{name}:{x}:{y}"
            block = box(x * width, y * width, (x + 1) * width, (y + 1) * width)
            if not block.covers(geometry):
                excluded[sid] = "crosses_block_boundary"
                continue
            blocks[key] = block
        else:
            key = sid
        groups.setdefault(name, {}).setdefault(key, []).append(sid)
    split, block_split = {}, {}
    for name, grouped in sorted(groups.items()):
        keys = sorted(grouped)
        rng.shuffle(keys)
        if len(keys) < 3:
            raise ValueError(f"Insufficient independent units in {name}")
        nv, nt = max(1, round(len(keys) * 0.2)), max(1, round(len(keys) * 0.2))
        for i, key in enumerate(keys):
            label = "test" if i < nt else "val" if i < nt + nv else "train"
            block_split[key] = label
            split.update({sid: label for sid in grouped[key]})
    if block_km:
        for candidate, protected in [("val", ["test"]), ("train", ["val", "test"])]:
            protected_shapes = [shapes[sid] for sid, label in split.items() if label in protected]
            region = unary_union(protected_shapes)
            for sid, label in list(split.items()):
                if label == candidate and (
                    shapes[sid].intersection(region).area > 1e-6
                    or shapes[sid].distance(region) < buffer_km * 1000
                ):
                    excluded[sid] = "buffer_exclusion"
                    del split[sid]
    counts = {label: list(split.values()).count(label) for label in ["train", "val", "test"]}
    if not all(counts.values()):
        raise ValueError("No usable train, validation or test observations after buffering")
    records = [dict(r, split=split[r["tile_id"]]) for r in rows if r["tile_id"] in split]
    features = [
        dict(
            type="Feature",
            geometry=mapping(geometry),
            properties=dict(block=key, split=block_split[key]),
        )
        for key, geometry in blocks.items()
    ]
    return records, dict(
        assignments=split,
        excluded=excluded,
        counts=counts,
        crs=crs,
        blocks=dict(
            type="FeatureCollection",
            features=features,
            crs=dict(type="name", properties=dict(name=crs)),
        ),
    )


def transfer_assignments(rows, target, test_ids, seed):
    remaining = [r for r in rows if city(r) != target]
    rng = np.random.default_rng(seed)
    split = {}
    for name in sorted({city(r) for r in remaining}):
        values = sorted(r["tile_id"] for r in remaining if city(r) == name)
        rng.shuffle(values)
        if len(values) < 2:
            raise ValueError("Each source city requires training and validation observations")
        n = max(1, round(len(values) * 0.2))
        split.update({sid: "val" if i < n else "train" for i, sid in enumerate(values)})
    split.update(
        {r["tile_id"]: "test" for r in rows if city(r) == target and r["tile_id"] in test_ids}
    )
    records = []
    for row in rows:
        if row["tile_id"] in split:
            text = "; ".join(
                part.strip()
                for part in row["prompt_structured"].split(";")
                if part.partition("=")[0].strip() not in ["city", "location"]
            )
            records.append(dict(row, split=split[row["tile_id"]], prompt_structured=text))
    return records
