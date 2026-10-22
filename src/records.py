import csv
import json
import math
from pathlib import Path
import random
import re
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from . import core
from .config import NAMES


def read(path, split=None):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f)) if path.suffix.lower() == ".csv" else json.load(f)
    rows = [dict(r, tile_id=str(r.get("tile_id", r.get("sample_id", "")))) for r in rows]
    ids = [r["tile_id"] for r in rows]
    if len(set(ids)) != len(ids) or any(
        not x or "/" in x or chr(92) in x or ":" in x or x in [".", ".."] for x in ids
    ):
        raise ValueError("Invalid or duplicate identifiers")
    rows = [r for r in rows if split is None or r.get("split") == split]
    if not rows:
        raise ValueError("No matching records")
    return rows


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def prompt(text, variant):
    text = text.replace("water_wetland=", "water=")
    text = "; ".join(
        part.strip()
        for part in text.split(";")
        if not ("%" in part and "=" in part and part.partition("=")[0].strip().lower() not in NAMES)
    )
    if variant == "without_text":
        return ""
    field = {
        "without_population": "population_density",
        "without_walkability_condition": "walkability",
        "without_city": "city",
        "without_country": "country",
    }.get(variant)
    if field:
        text = "; ".join(
            part.strip()
            for part in text.split(";")
            if part.strip().partition("=")[0].strip() != field
        )
    return text


class Records:
    def __init__(self, rows, root, config, stage=0, targets=True):
        self.rows, self.root, self.c, self.stage = rows, Path(root), config, stage
        self.targets = targets

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r, c = self.rows[index], self.c
        s = {"tile_id": r["tile_id"], "prompt": prompt(r["prompt_structured"], c["variant"])}
        if self.targets:
            s["semantic"] = core.read_labels(
                self.path(r.get("semantic_id_path") or r["semantic_map_path"]),
                c["semantic_palette"],
                c["resolution"],
                c["num_classes"],
                None if r.get("semantic_id_path") else c["semantic_id_map"],
                c["ignore_index"],
                c["semantic_ignore_colors"],
            )
        if self.stage != 2:
            labels = core.read_labels(
                self.path(r["control_map_path"]),
                c["control_palette"],
                c["resolution"],
                c["num_controls"],
            )
            s["control"] = F.one_hot(labels, c["num_controls"]).permute(2, 0, 1).float()
            if c["variant"] == "without_control":
                s["control"].zero_()
            if self.targets:
                s["walkability"] = torch.tensor(walk_target(r), dtype=torch.float32)
        if self.stage != 1 and self.targets:
            with Image.open(self.path(r["satellite_path"])) as image:
                image = image.convert("RGB").resize(
                    (c["resolution"], c["resolution"]), Image.Resampling.BILINEAR
                )
                s["satellite"] = (
                    torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 127.5
                    - 1
                )
        return s


def walk_target(row):
    match = re.search(r"\bwalkability=([0-9.]+)", row["prompt_structured"])
    value = float(match[1]) if match else float(row["walkability_score"])
    if not math.isfinite(value) or not 1 <= value <= 20:
        raise ValueError("Invalid walkability value")
    return value


def density_target(row):
    match = re.search(r"\bpopulation_density=([0-9.]+)", row["prompt_structured"])
    value = float(match[1]) if match else float(row["population_density"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid population density")
    return value


@torch.no_grad()
def walk_stats(samples, c):
    values = []
    for sample in samples:
        labels = sample["semantic"]
        valid = labels != c["ignore_index"]
        if not valid.any():
            continue
        probabilities = (
            F.one_hot(labels.masked_fill(~valid, 0), c["num_classes"])
            .permute(2, 0, 1)[None]
            .float()
        )
        probabilities *= valid[None, None]
        a, b = core.semantic_indicators(
            probabilities, c["road_class"], c["landuse_classes"], c["tile_area_km2"]
        )
        values.append([a.item(), b.item()])
    if not values:
        raise ValueError("No valid semantic pixels")
    values = np.array(values)
    mean, std = values.mean(0), np.maximum(values.std(0), 1e-8)
    z = ((values - mean) / std).sum(1)
    return core.WalkStats(*[float(v) for v in [mean[0], std[0], mean[1], std[1], z.min(), z.max()]])
