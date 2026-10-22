import copy
import json

CLASSES = ["Residential", "Farmland", "Forest", "Park", "Commercial", "Industrial", "Road", "Rail", "Water"]
NAMES = [name.lower() for name in CLASSES]
IDS = list(range(len(CLASSES)))
ROAD_ID = NAMES.index("road")
WATER_ID = NAMES.index("water")
RAIL_ID = NAMES.index("rail")
FIXED_IDS = [ROAD_ID, WATER_ID]
EDITABLE_IDS = [i for i in IDS if i not in FIXED_IDS]
CONTROL_TO_SEMANTIC = {1: ROAD_ID, 2: WATER_ID}
DEFAULTS = {
    "pretrained_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "resolution": 512,
    "num_classes": len(CLASSES),
    "class_names": CLASSES,
    "num_controls": 3,
    "base_channels": 64,
    "text_dim": 512,
    "tile_area_km2": 1.0,
    "road_class": ROAD_ID,
    "landuse_classes": [i for i in IDS if i != ROAD_ID],
    "ignore_index": 255,
    "semantic_ignore_colors": [[241, 240, 235]],
    "semantic_id_map": {0: 255, 1: 6, 2: 7, 3: 8, 4: 0, 5: 4, 6: 5, 7: 3, 8: 2, 9: 1},
    "seed": 42,
    "validation_seed": 1234,
    "max_steps": 50000,
    "warmup_steps": 2000,
    "validate_every": 1000,
    "patience": 10,
    "min_delta": 0.0,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_stage1": 1e-4,
    "lr_stage2": 1e-5,
    "effective_batch_stage1": 32,
    "effective_batch_stage2": 8,
    "semantic_weights": [1.0, 0.5, 0.3],
    "satellite_weights": [1.0, 1.0, 0.1],
    "semantic_palette": {
        0: [255, 193, 130],
        1: [205, 215, 140],
        2: [45, 155, 45],
        3: [100, 195, 100],
        4: [90, 185, 170],
        5: [165, 130, 195],
        6: [252, 252, 249],
        7: [65, 65, 65],
        8: [80, 155, 220],
    },
    "control_palette": {0: [241, 240, 235], 1: [252, 252, 249], 2: [80, 155, 220]},
    "inference_steps": 30,
    "guidance_scale": 3.0,
    "control_scale": 1.0,
    "variant": "full",
}


def load(path=None, variant=None):
    c = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            c.update(json.load(f))
    for key in ["semantic_palette", "control_palette"]:
        c[key] = {int(k): v for k, v in c[key].items()}
    c["semantic_id_map"] = {int(k): int(v) for k, v in c["semantic_id_map"].items()}
    if variant:
        c["variant"] = variant
    v = c["variant"]
    for name, key, index in [
        ("without_dice", "semantic_weights", 1),
        ("without_walkability_loss", "semantic_weights", 2),
        ("without_reconstruction", "satellite_weights", 1),
        ("without_perceptual", "satellite_weights", 2),
    ]:
        if v == name:
            c[key][index] = 0.0
    if c["resolution"] % 8 or c["max_steps"] <= 0 or c["validate_every"] <= 0:
        raise ValueError("Invalid resolution or optimizer settings")
    validate(c)
    return c


def validate(c):
    if (
        c["num_classes"] != len(CLASSES)
        or c["class_names"] != CLASSES
        or set(c["semantic_palette"]) != set(IDS)
        or c["road_class"] != ROAD_ID
        or c["landuse_classes"] != [i for i in IDS if i != ROAD_ID]
        or c["ignore_index"] != 255
        or c["num_controls"] != 3
        or set(c["control_palette"]) != {0, 1, 2}
        or any(v not in IDS + [255] for v in c["semantic_id_map"].values())
    ):
        raise ValueError("The semantic configuration must use the nine-class order and three controls")
