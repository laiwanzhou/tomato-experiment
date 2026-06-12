from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "tomato"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "seg_dataset"
PREVIEW_DIR = PROJECT_ROOT / "outputs" / "seg_dataset_preview"

TOMATO_CLASSES = {
    "l_green",
    "b_green",
    "l_fully_ripened",
    "l_half_ripened",
    "b_half_ripened",
    "b_fully_ripened",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread_unicode(path: Path) -> np.ndarray:
    """Read an image from a Windows path that may contain non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """Write an image to a Windows path that may contain non-ASCII characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: p.name)


def points_to_contour(points: Any) -> np.ndarray | None:
    """Convert a JSON polygon point list to an OpenCV contour."""
    if not isinstance(points, list) or len(points) < 3:
        return None
    contour = np.array(points, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2:
        return None
    return np.round(contour).astype(np.int32).reshape((-1, 1, 2))


def convert_split(split: str) -> tuple[pd.DataFrame, list[dict[str, Any]], Counter]:
    img_dir = DATASET_ROOT / split / "img"
    ann_dir = DATASET_ROOT / split / "ann"
    mask_dir = OUTPUT_ROOT / split / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    class_counter: Counter = Counter()

    image_paths = list_images(img_dir)
    image_stems = {p.name for p in image_paths}
    json_paths = sorted(ann_dir.glob("*.json"), key=lambda p: p.name)
    expected_json_names = {f"{p.name}.json" for p in image_paths}

    for json_path in json_paths:
        image_name = json_path.name.removesuffix(".json")
        if image_name not in image_stems:
            warning_rows.append(
                {
                    "split": split,
                    "image_path": "",
                    "json_path": str(json_path),
                    "warning_type": "missing_image",
                    "message": f"No image found for annotation {json_path.name}",
                }
            )

    for image_path in tqdm(image_paths, desc=f"Convert {split}", unit="img"):
        json_path = ann_dir / f"{image_path.name}.json"
        if not json_path.exists():
            warning_rows.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                    "warning_type": "missing_json",
                    "message": f"No annotation found for image {image_path.name}",
                }
            )
            continue

        data = read_json(json_path)
        size = data.get("size", {})
        width = int(size.get("width", 0) or 0)
        height = int(size.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            warning_rows.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                    "warning_type": "invalid_json_size",
                    "message": f"Invalid JSON size: {size}",
                }
            )
            try:
                image = imread_unicode(image_path)
                height, width = image.shape[:2]
            except ValueError:
                continue

        try:
            image = imread_unicode(image_path)
            image_h, image_w = image.shape[:2]
            if (image_w, image_h) != (width, height):
                warning_rows.append(
                    {
                        "split": split,
                        "image_path": str(image_path),
                        "json_path": str(json_path),
                        "warning_type": "size_mismatch",
                        "message": f"JSON size {width}x{height}, image size {image_w}x{image_h}",
                    }
                )
        except ValueError as exc:
            warning_rows.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                    "warning_type": "image_read_failed",
                    "message": str(exc),
                }
            )

        mask = np.zeros((height, width), dtype=np.uint8)
        object_count = 0
        for object_index, obj in enumerate(data.get("objects", [])):
            class_title = obj.get("classTitle", "")
            geometry_type = obj.get("geometryType", "")
            if class_title:
                class_counter[class_title] += 1

            if geometry_type != "polygon" or class_title not in TOMATO_CLASSES:
                continue

            points = obj.get("points", {})
            exterior = points.get("exterior", [])
            exterior_contour = points_to_contour(exterior)
            if exterior_contour is None:
                warning_rows.append(
                    {
                        "split": split,
                        "image_path": str(image_path),
                        "json_path": str(json_path),
                        "warning_type": "polygon_points_insufficient",
                        "message": f"object_index={object_index}, classTitle={class_title}, exterior point count={len(exterior) if isinstance(exterior, list) else 'invalid'}",
                    }
                )
                continue

            cv2.fillPoly(mask, [exterior_contour], 1)
            object_count += 1

            interiors = points.get("interior", []) or []
            for interior_index, interior in enumerate(interiors):
                interior_contour = points_to_contour(interior)
                if interior_contour is None:
                    warning_rows.append(
                        {
                            "split": split,
                            "image_path": str(image_path),
                            "json_path": str(json_path),
                            "warning_type": "interior_points_insufficient",
                            "message": f"object_index={object_index}, interior_index={interior_index}, point count={len(interior) if isinstance(interior, list) else 'invalid'}",
                        }
                    )
                    continue
                cv2.fillPoly(mask, [interior_contour], 0)

        mask_path = mask_dir / f"{image_path.stem}.png"
        imwrite_unicode(mask_path, mask)
        tomato_area_px = int(mask.sum())
        tomato_area_ratio = tomato_area_px / float(width * height) if width > 0 and height > 0 else 0.0
        manifest_rows.append(
            {
                "idx": len(manifest_rows) + 1,
                "split": split,
                "image_path": str(image_path),
                "json_path": str(json_path),
                "mask_path": str(mask_path),
                "width": width,
                "height": height,
                "object_count": object_count,
                "tomato_area_px": tomato_area_px,
                "tomato_area_ratio": tomato_area_ratio,
            }
        )

    for missing_json in sorted(expected_json_names - {p.name for p in json_paths}):
        warning_rows.append(
            {
                "split": split,
                "image_path": str(img_dir / missing_json.removesuffix(".json")),
                "json_path": str(ann_dir / missing_json),
                "warning_type": "missing_json",
                "message": f"No annotation found for image {missing_json.removesuffix('.json')}",
            }
        )

    return pd.DataFrame(manifest_rows), warning_rows, class_counter


def draw_text(image: np.ndarray, lines: list[str]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 64), (255, 255, 255), -1)
    for i, line in enumerate(lines[:3]):
        cv2.putText(image, line, (10, 22 + i * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def make_preview(split: str, manifest_df: pd.DataFrame, output_path: Path, sample_count: int = 12) -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = manifest_df.sort_values("idx").head(sample_count)
    if rows.empty:
        return

    panel_w, panel_h = 260, 220
    sample_h = panel_h
    sample_w = panel_w * 3
    canvas = np.full((sample_h * len(rows), sample_w, 3), 255, dtype=np.uint8)

    for sample_i, row in enumerate(rows.itertuples(index=False)):
        image = imread_unicode(Path(row.image_path))
        mask = imread_unicode(Path(row.mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_vis = (mask > 0).astype(np.uint8) * 255
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        overlay = image.copy()
        color = np.zeros_like(overlay)
        color[:, :, 2] = 255
        overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.55, color, 0.45, 0), overlay)

        original_panel = fit_panel(image, panel_w, panel_h)
        mask_panel = fit_panel(mask_bgr, panel_w, panel_h)
        overlay_panel = fit_panel(overlay, panel_w, panel_h)

        label_lines = [
            f"{split}: {Path(row.image_path).name}",
            f"objects={int(row.object_count)}",
            f"area_ratio={float(row.tomato_area_ratio):.4f}",
        ]
        draw_text(original_panel, label_lines)
        draw_text(mask_panel, ["mask 0/1 shown as white", "", ""])
        draw_text(overlay_panel, ["overlay", "", ""])

        y = sample_i * sample_h
        canvas[y : y + sample_h, 0:panel_w] = original_panel
        canvas[y : y + sample_h, panel_w : panel_w * 2] = mask_panel
        canvas[y : y + sample_h, panel_w * 2 : panel_w * 3] = overlay_panel

    imwrite_unicode(output_path, canvas)


def main() -> None:
    (OUTPUT_ROOT / "Train" / "mask").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "Test" / "mask").mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    all_warnings: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for split in ["Train", "Test"]:
        manifest_df, warning_rows, class_counter = convert_split(split)
        all_warnings.extend(warning_rows)
        manifest_path = OUTPUT_ROOT / f"{split.lower()}_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        make_preview(split, manifest_df, PREVIEW_DIR / f"{split.lower()}_mask_preview.jpg")

        for class_title, count in sorted(class_counter.items()):
            stats_rows.append({"split": split, "classTitle": class_title, "count": count})

    pd.DataFrame(stats_rows).to_csv(OUTPUT_ROOT / "class_stats.csv", index=False, encoding="utf-8-sig")
    warning_df = pd.DataFrame(
        all_warnings,
        columns=["split", "image_path", "json_path", "warning_type", "message"],
    )
    warning_df.to_csv(OUTPUT_ROOT / "convert_warnings.csv", index=False, encoding="utf-8-sig")

    print("Wrote: outputs/seg_dataset/Train/mask")
    print("Wrote: outputs/seg_dataset/Test/mask")
    print("Wrote: outputs/seg_dataset/train_manifest.csv")
    print("Wrote: outputs/seg_dataset/test_manifest.csv")
    print("Wrote: outputs/seg_dataset/class_stats.csv")
    print("Wrote: outputs/seg_dataset/convert_warnings.csv")
    print("Wrote: outputs/seg_dataset_preview/train_mask_preview.jpg")
    print("Wrote: outputs/seg_dataset_preview/test_mask_preview.jpg")


if __name__ == "__main__":
    main()
