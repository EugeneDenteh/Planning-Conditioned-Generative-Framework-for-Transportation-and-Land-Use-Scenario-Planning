import math

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .config import NAMES, ROAD_ID


DEFINITIONS = {
    "patch_connectivity": 4,
    "adjacency_percent": "100 * unlike shared edges of the class pair / unlike edges touching either class",
    "coherence": "same-class shared edges / all shared edges",
    "fragmentation": "(patch count - present class count) / (pixel count - present class count)",
    "patch_coherence": "sum patch n*(n-1) / sum class n*(n-1)",
    "spatial_entropy_bits": "Shannon entropy of directed neighboring class pairs, log2, maximum log2(81)",
    "class_entropy_bits": "nine-class Shannon entropy, log2, maximum log2(9)",
    "network_intersections": "eight-connected clusters of skeleton pixels with degree at least three",
    "distortion": "symmetric mean nearest boundary distance divided by image diagonal",
}


def grid(values, allow_ignore=False):
    values = np.asarray(values, dtype=np.int64)
    invalid = (values < 0) | (values >= len(NAMES))
    if allow_ignore:
        invalid &= values != 255
    if values.ndim != 2 or invalid.any():
        raise ValueError("Expected a two-dimensional nine-class map")
    return values


def neighbors(values):
    return np.concatenate([values[:, :-1].ravel(), values[:-1, :].ravel()]), np.concatenate(
        [values[:, 1:].ravel(), values[1:, :].ravel()]
    )


def entropy(counts):
    counts = np.asarray(counts, dtype=float)
    probabilities = counts[counts > 0] / counts.sum() if counts.sum() else np.array([])
    return float(-(probabilities * np.log2(probabilities)).sum())


def measurements(values, area, allow_ignore=False):
    values = grid(values, allow_ignore)
    if not math.isfinite(area) or area <= 0:
        raise ValueError("A positive tile area in square kilometers is required")
    valid = values != 255
    if not valid.any():
        raise ValueError("No valid semantic pixels")
    counts = np.bincount(values[valid], minlength=len(NAMES))
    first, second = neighbors(values)
    usable = (first != 255) & (second != 255)
    first, second = first[usable], second[usable]
    unequal = first != second

    def adjacency(a, b):
        numerator = ((first == a) & (second == b)) | ((first == b) & (second == a))
        denominator = unequal & (np.isin(first, [a, b]) | np.isin(second, [a, b]))
        return 100 * numerator.sum() / denominator.sum() if denominator.any() else 0.0

    patches = []
    for class_id in range(len(NAMES)):
        components, number = label(values == class_id)
        patches.extend(np.bincount(components.ravel(), minlength=number + 1)[1:].tolist())
    patches = np.asarray(patches, dtype=float)
    hectares = patches * area * 100 / values.size
    present = np.count_nonzero(counts)
    denominator = np.sum(counts * (counts - 1))
    size = len(NAMES)
    pairs = np.bincount(first * size + second, minlength=size**2).reshape(size, size)
    directed = pairs + pairs.T
    road = skeletonize(values == ROAD_ID)
    from scipy.ndimage import convolve

    kernel = np.ones((3, 3), int)
    kernel[1, 1] = 0
    junctions = road & (convolve(road.astype(int), kernel, mode="constant") >= 3)
    junctions[[0, -1], :] = False
    junctions[:, [0, -1]] = False
    _, intersections = label(junctions, structure=np.ones((3, 3)))
    return dict(
        counts=counts.tolist(),
        shares_pp=(100 * counts / counts.sum()).tolist(),
        residential_industrial_adjacency=float(adjacency(NAMES.index("residential"), NAMES.index("industrial"))),
        road_forest_adjacency=float(adjacency(ROAD_ID, NAMES.index("forest"))),
        mean_patch_ha=float(hectares.mean()),
        patch_std_ha=float(hectares.std()),
        coherence=float((first == second).mean()) if len(first) else None,
        fragmentation=float((len(patches) - present) / max(valid.sum() - present, 1)),
        patch_coherence=(
            float(np.sum(patches * (patches - 1)) / denominator) if denominator else 1.0
        ),
        spatial_entropy_bits=entropy(directed.ravel()),
        class_entropy_bits=entropy(counts),
        normalized_class_entropy=entropy(counts) / math.log2(len(NAMES)),
        intersections=int(intersections),
        intersection_density=float(intersections / area),
        area_km2=float(area),
    )


def boundaries(values):
    result = np.zeros(values.shape, bool)
    change = values[:, :-1] != values[:, 1:]
    result[:, :-1] |= change
    result[:, 1:] |= change
    change = values[:-1, :] != values[1:, :]
    result[:-1, :] |= change
    result[1:, :] |= change
    return np.argwhere(result)


def distortion(first, second):
    first, second = grid(first), grid(second)
    if first.shape != second.shape:
        raise ValueError("Semantic grids differ")
    a, b = boundaries(first), boundaries(second)
    if not len(a) and not len(b):
        return 0.0
    if not len(a) or not len(b):
        return 1.0
    return float(
        (cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean())
        / (2 * np.hypot(*first.shape))
    )


def descriptor(measured):
    return np.asarray(measured["shares_pp"]) / 100
