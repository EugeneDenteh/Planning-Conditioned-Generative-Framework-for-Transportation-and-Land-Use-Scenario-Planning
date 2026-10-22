import csv
from pathlib import Path
from .config import CLASSES
from .records import save


def emit(out, number, rows):
    with (Path(out) / f"table{number}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_file(path, summary, row_label="Truth / prediction"):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row_label] + CLASSES)
        w.writerows([name] + row for name, row in zip(CLASSES, summary["confusion"]))


def primary(out, result):
    overall = result["overall"]
    rows = [
        {k: r[k] for k in ["Class", "IoU", "Recall", "Precision", "F1"]}
        for r in overall["semantic"]["classes"]
    ]
    emit(out, 1, rows + [dict(Class="Macro average", **overall["semantic"]["macro"])])
    emit(
        out,
        2,
        [
            dict(
                City=city,
                Tiles=r["tiles"],
                mIoU=r["semantic"]["miou"],
                Pixel_accuracy=r["semantic"]["pixel_accuracy"],
            )
            for city, r in result["cities"].items()
        ]
        + [
            dict(
                City="Overall",
                Tiles=overall["tiles"],
                mIoU=overall["semantic"]["miou"],
                Pixel_accuracy=overall["semantic"]["pixel_accuracy"],
            )
        ],
    )
    emit(
        out,
        3,
        [
            dict(City=city, Tiles=r["tiles"], **r["walkability"])
            for city, r in result["cities"].items()
        ]
        + [dict(City="Overall", Tiles=overall["tiles"], **overall["walkability"])],
    )
    consistency = overall["image_map"]
    if consistency is not None:
        rows = [
            {k: r[k] for k in ["Class", "IoU", "Recall", "Precision"]}
            for r in consistency["classes"]
        ]
        rows.append(
            dict(
                Class="Macro average",
                **{k: consistency["macro"][k] for k in ["IoU", "Recall", "Precision"]},
            )
        )
        emit(out, 6, rows)
        matrix_file(Path(out) / "image_map_confusion.csv", consistency, "Map / image label")
    matrix_file(Path(out) / "confusion.csv", overall["semantic"])
    save(Path(out) / "confusion.json", overall["semantic"])
    save(
        Path(out) / "city_confusions.json",
        {city: r["semantic"] for city, r in result["cities"].items()},
    )
    with (Path(out) / "class_frequencies.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["Class", "Pixels", "Frequency", "Predicted_pixels", "Outside_prediction_errors"]
        )
        for r, outside in zip(
            overall["semantic"]["classes"], overall["semantic"]["outside_predictions"]
        ):
            support = r["TP"] + r["FN"]
            frequency = (
                support / overall["semantic"]["scored_pixels"]
                if overall["semantic"]["scored_pixels"]
                else None
            )
            w.writerow([r["Class"], support, frequency, r["TP"] + r["FP"], outside])


def ablations(out, results):
    def row(name):
        r = results[name]["overall"]
        return dict(
            Variant=name,
            Semantic_mIoU=r["semantic"]["miou"],
            Pixel_accuracy=r["semantic"]["pixel_accuracy"],
            Walkability_MAE=r["walkability"]["MAE"],
            Mean_tile_share_MAE_pp=r["share_mae_pp"],
            Pooled_share_MAE_pp=r.get("pooled_share_mae_pp"),
            Population_density_MAE=r.get("density_mae"),
            Fixed_feature_preservation=r["fixed_feature_preservation"],
            Image_map_consistency=r["image_map"]["miou"] if r["image_map"] else None,
            FID=r["images"].get("FID"),
            RGB_MSE=r["images"].get("MSE"),
            RGB_MAE_01=r["images"]["MAE"] / 255 if r["images"].get("MAE") is not None else None,
            PSNR_dB=r["images"].get("PSNR"),
            SSIM=r["images"].get("SSIM"),
            Pearson_r=r["walkability"]["Pearson_r"],
            Semantic_matrix="semantic_full" if name in ["without_semantics", "semantic_only", "without_perceptual", "without_reconstruction"] else "semantic_" + name,
            Image_map_matrix="image_" + name if r["image_map"] else None,
            Changed=results[name].get("changed", True),
        )

    components = [
        "full",
        "without_text",
        "without_control",
        "without_population",
        "without_walkability_condition",
        "without_city",
        "without_walkability_loss",
        "without_semantics",
        "semantic_only",
    ]
    rows = []
    for name in components:
        r = row(name)
        r.pop("Pearson_r")
        rows.append(r)
    emit(out, 4, rows)
    keep = [
        "Variant",
        "Semantic_mIoU",
        "FID",
        "Walkability_MAE",
        "Pearson_r",
        "Mean_tile_share_MAE_pp",
        "Pooled_share_MAE_pp",
        "RGB_MSE",
        "RGB_MAE_01",
        "PSNR_dB",
        "SSIM",
        "Semantic_matrix",
    ]
    emit(
        out,
        5,
        [
            {k: row(name)[k] for k in keep}
            for name in ["full", "without_dice", "without_perceptual", "without_reconstruction"]
        ],
    )
    for name in [
        "without_semantics",
        "without_perceptual",
        "without_reconstruction",
        "semantic_only",
    ]:
        a, b = results["full"]["overall"], results[name]["overall"]
        assert (
            a["semantic"] == b["semantic"]
            and a["walkability"] == b["walkability"]
            and a["share_mae_pp"] == b["share_mae_pp"]
            and a.get("pooled_share_mae_pp") == b.get("pooled_share_mae_pp")
            and a.get("density_mae") == b.get("density_mae")
        )
