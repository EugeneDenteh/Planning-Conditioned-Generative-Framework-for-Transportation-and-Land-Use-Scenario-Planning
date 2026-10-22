import numpy as np
from .config import CLASSES, IDS, NAMES, CONTROL_TO_SEMANTIC


def confusion(target, prediction, classes=len(NAMES), valid_mask=None):
    a, b = np.asarray(target, dtype=np.int64), np.asarray(prediction, dtype=np.int64)
    if classes != len(NAMES) or a.shape != b.shape:
        raise ValueError("Incompatible semantic grids")
    valid = a != 255
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != a.shape or mask.dtype != np.bool_:
            raise ValueError("Expected a boolean validity mask matching the semantic grid")
        valid &= mask
    a, b = a[valid], b[valid]
    if ((a < 0) | (a >= classes) | (b < 0) | (b >= classes)).any():
        raise ValueError("Incompatible semantic labels")
    return np.bincount((a * classes + b).ravel(), minlength=classes * classes).reshape(
        classes, classes
    )


def segmentation(raw):
    raw = np.asarray(raw, dtype=np.int64)
    if raw.shape != (len(NAMES), len(NAMES)) or (raw < 0).any():
        raise ValueError("Expected a nonnegative nine-class confusion matrix")
    selected = raw[IDS]
    cm = selected[:, IDS]
    support, predicted, tp = selected.sum(1), cm.sum(0), cm.diagonal()
    records = []
    for i, name in enumerate(CLASSES):
        t, p, x = int(support[i]), int(predicted[i]), int(tp[i])
        records.append(
            dict(
                Class=name,
                TP=x,
                FP=p - x,
                FN=t - x,
                IoU=x / (t + p - x) if t + p - x else 0.0,
                Recall=x / t if t else 0.0,
                Precision=x / p if p else 0.0,
                F1=2 * x / (t + p) if t + p else 0.0,
            )
        )
    macro = {}
    for key in ["IoU", "Recall", "Precision", "F1"]:
        values = [r[key] for r in records if r[key] is not None]
        macro[key] = float(np.mean(values)) if support.sum() else None
    return dict(
        classes=records,
        macro=macro,
        miou=macro["IoU"],
        pixel_accuracy=float(tp.sum() / support.sum()) if support.sum() else None,
        confusion=cm.tolist(),
        outside_predictions=(support - cm.sum(1)).tolist(),
        scored_pixels=int(support.sum()),
        zero_division=0,
    )


def regression(target, achieved):
    a, b = np.asarray(target, float), np.asarray(achieved, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid scalar observations")
    if not len(a):
        return dict(MAE=None, Pearson_r=None, R2=None, Bias=None)
    delta = b - a
    denominator = np.square(a - a.mean()).sum()
    return dict(
        MAE=float(np.abs(delta).mean()),
        Bias=float(delta.mean()),
        Pearson_r=float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None,
        R2=float(1 - np.square(delta).sum() / denominator) if denominator else None,
    )


def preservation(control, prediction, valid_mask=None):
    control, prediction = np.asarray(control), np.asarray(prediction)
    if control.shape != prediction.shape:
        raise ValueError("Incompatible control and prediction grids")
    valid = np.ones(control.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if valid.shape != control.shape or valid.dtype != np.bool_:
        raise ValueError("Expected a boolean validity mask matching the control grid")
    result = []
    for control_id, semantic_id in CONTROL_TO_SEMANTIC.items():
        a, b = (control == control_id) & valid, (prediction == semantic_id) & valid
        result.append([int((a & b).sum()), int(a.sum()), int(b.sum())])
    return np.array(result, dtype=np.int64)


def preserved(counts):
    values = [2 * t / (a + b) for t, a, b in counts if a + b]
    return float(np.mean(values)) if values else None


def rgb_metrics(a, b):
    from skimage.metrics import structural_similarity

    a, b = np.asarray(a, float) / 255, np.asarray(b, float) / 255
    delta = a - b
    mse = float(np.square(delta).mean())
    return dict(
        MSE=mse,
        MAE=float(np.abs(delta).mean() * 255),
        PSNR=float(-10 * np.log10(mse)) if mse else None,
        SSIM=float(structural_similarity(a, b, data_range=1.0, channel_axis=-1, win_size=7)),
    )


def distribution(first, second, device, seed):
    import torch_fidelity

    result = torch_fidelity.calculate_metrics(
        input1=str(first),
        input2=str(second),
        cuda=str(device).startswith("cuda"),
        fid=True,
        kid=True,
        kid_subset_size=min(100, len(list(first.glob("*.png")))),
        rng_seed=seed,
        verbose=False,
    )
    return dict(
        FID=float(result["frechet_inception_distance"]),
        KID=float(result["kernel_inception_distance_mean"]),
        KID_std=float(result["kernel_inception_distance_std"]),
    )
