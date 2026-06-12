from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_CSV = OUTPUT_DIR / "weights.csv"
ROI_DIR = OUTPUT_DIR / "roi_tomato_images"
DEBUG_MASKS_DIR = OUTPUT_DIR / "debug_masks"
DEBUG_SCALE_DIR = OUTPUT_DIR / "debug_scale"
VOLUME_RESULTS_DIR = OUTPUT_DIR / "volume_results"
SCALE_CONFIG_JSON = OUTPUT_DIR / "scale_config.json"
SUMMARY_XLSX = OUTPUT_DIR / "summary.xlsx"

CUBE_EDGE_CM = 5.0
TRAIN_COUNT = 20
VOLUME_COLUMNS = [
    "V1_area_sphere",
    "V2_axis_mean_sphere",
    "V3_ellipsoid_H_eq_W",
    "V4_ellipsoid_H_eq_mean",
    "V5_ellipsoid_H_eq_geom",
]
DENSITY_METHODS = ["mean", "median", "lstsq"]


def mode_paths(mask_mode: str) -> dict[str, Path]:
    """Return mode-specific output paths without overwriting the original red run."""
    if mask_mode == "warm":
        return {
            "debug_dir": OUTPUT_DIR / "debug_masks_warm",
            "results_dir": OUTPUT_DIR / "volume_results_warm",
            "summary_xlsx": OUTPUT_DIR / "summary_warm.xlsx",
            "contact_sheet": OUTPUT_DIR / "debug_masks_contact_sheet_warm.jpg",
        }
    if mask_mode == "red":
        return {
            "debug_dir": OUTPUT_DIR / "debug_masks_red",
            "results_dir": OUTPUT_DIR / "volume_results_red",
            "summary_xlsx": OUTPUT_DIR / "summary_red.xlsx",
            "contact_sheet": OUTPUT_DIR / "debug_masks_contact_sheet_red.jpg",
        }
    raise ValueError(f"Unknown mask_mode: {mask_mode}")


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"Could not encode image for: {path}")
    encoded.tofile(str(path))


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Read user-edited CSV files that may be saved as UTF-8 or local ANSI/GBK."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            df.columns = [str(col).strip() for col in df.columns]
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Could not decode {path}; tried {errors}")


def parse_roi_idx(path: Path) -> int:
    stem = path.stem
    # Expected: idx_001_roi
    parts = stem.split("_")
    if len(parts) < 2 or parts[0] != "idx":
        raise ValueError(f"Unexpected ROI filename format: {path.name}")
    return int(parts[1])


def load_weights() -> pd.DataFrame:
    if not WEIGHTS_CSV.exists():
        raise FileNotFoundError("Missing outputs/weights.csv. Fill weights_template.csv and save it as weights.csv first.")

    df = read_csv_with_fallback(WEIGHTS_CSV)
    required = {"idx", "weight_g"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"outputs/weights.csv is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
    df["weight_g"] = pd.to_numeric(df["weight_g"], errors="coerce")
    bad = df[df["idx"].isna() | df["weight_g"].isna()]
    if not bad.empty:
        bad_rows = ", ".join(str(i + 2) for i in bad.index.tolist())
        raise ValueError(
            "outputs/weights.csv has blank or non-numeric idx/weight_g values. "
            f"Please fill complete numeric weights first. Problem CSV rows: {bad_rows}"
        )

    df["idx"] = df["idx"].astype(int)
    df = df.sort_values("idx").reset_index(drop=True)
    return df


def load_roi_table(weights_df: pd.DataFrame) -> pd.DataFrame:
    roi_paths = sorted(ROI_DIR.glob("idx_*_roi.*"), key=parse_roi_idx)
    if not roi_paths:
        raise FileNotFoundError("No ROI images found in outputs/roi_tomato_images. Run phase 1 ROI crop first.")

    roi_df = pd.DataFrame({"idx": [parse_roi_idx(p) for p in roi_paths], "roi_image": [str(p) for p in roi_paths]})
    merged = weights_df.merge(roi_df, on="idx", how="left")
    missing = merged[merged["roi_image"].isna()]
    if not missing.empty:
        raise ValueError(f"Missing ROI images for idx: {missing['idx'].tolist()}")

    extra = sorted(set(roi_df["idx"]) - set(weights_df["idx"]))
    if extra:
        print(f"WARNING: ROI images without weights will be ignored: {extra}")

    return merged.sort_values("idx").reset_index(drop=True)


def select_scale_points(image: np.ndarray) -> dict[str, float]:
    """Collect two mouse clicks on a clear 5 cm cube edge."""
    max_display_w = 1400
    max_display_h = 900
    image_h, image_w = image.shape[:2]
    display_scale = min(1.0, max_display_w / image_w, max_display_h / image_h)
    display = cv2.resize(
        image,
        (int(round(image_w * display_scale)), int(round(image_h * display_scale))),
        interpolation=cv2.INTER_AREA,
    )
    shown = display.copy()
    points: list[tuple[int, int]] = []
    window_name = "Click two endpoints of one 5cm front cube edge, then press ENTER"

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(shown, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(shown, points[0], points[1], (0, 255, 255), 2)
            cv2.imshow(window_name, shown)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, shown)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32) and len(points) == 2:
            break
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Scale selection cancelled. Re-run and click two endpoints.")

    cv2.destroyWindow(window_name)
    (x1, y1), (x2, y2) = points
    full_x1 = x1 / display_scale
    full_y1 = y1 / display_scale
    full_x2 = x2 / display_scale
    full_y2 = y2 / display_scale
    edge_px = math.hypot(full_x2 - full_x1, full_y2 - full_y1)
    if edge_px <= 0:
        raise ValueError("Selected scale edge has zero length.")

    return {
        "cube_x1": float(full_x1),
        "cube_y1": float(full_y1),
        "cube_x2": float(full_x2),
        "cube_y2": float(full_y2),
        "cube_edge_px": float(edge_px),
        "scale_cm_per_px": float(CUBE_EDGE_CM / edge_px),
    }


def draw_scale_reference(image: np.ndarray, config: dict[str, float]) -> None:
    debug = image.copy()
    p1 = (int(round(config["cube_x1"])), int(round(config["cube_y1"])))
    p2 = (int(round(config["cube_x2"])), int(round(config["cube_y2"])))
    cv2.line(debug, p1, p2, (0, 255, 255), 4)
    cv2.circle(debug, p1, 8, (0, 0, 255), -1)
    cv2.circle(debug, p2, 8, (0, 0, 255), -1)
    label = f"edge_px={config['cube_edge_px']:.2f}, scale={config['scale_cm_per_px']:.5f} cm/px"
    cv2.putText(debug, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(debug, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(DEBUG_SCALE_DIR / "scale_reference.jpg", debug)


def load_or_select_scale(first_idx: int, first_image: np.ndarray, reset_scale: bool) -> dict[str, float]:
    if SCALE_CONFIG_JSON.exists() and not reset_scale:
        with SCALE_CONFIG_JSON.open("r", encoding="utf-8") as f:
            config = json.load(f)
        print("Using existing scale config: outputs/scale_config.json")
        return config

    config = {"reference_idx": int(first_idx), **select_scale_points(first_image)}
    with SCALE_CONFIG_JSON.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    draw_scale_reference(first_image, config)
    print("Wrote scale config: outputs/scale_config.json")
    print("Wrote scale debug image: outputs/debug_scale/scale_reference.jpg")
    return config


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside a binary mask using flood fill from the image border."""
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def make_color_mask(image: np.ndarray, mask_mode: str) -> np.ndarray:
    """Build the initial HSV mask for either strict red or broader warm colors."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask_mode == "red":
        lower1 = np.array([0, 55, 45], dtype=np.uint8)
        upper1 = np.array([12, 255, 255], dtype=np.uint8)
        lower2 = np.array([165, 45, 40], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
    elif mask_mode == "warm":
        # Warm mode expands hue to orange/yellow tomatoes but keeps saturation high
        # enough to avoid white cube/platform edges.
        lower1 = np.array([0, 90, 40], dtype=np.uint8)
        upper1 = np.array([35, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 90, 40], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(mask1, mask2)


def segment_tomato(image: np.ndarray, mask_mode: str = "red") -> dict[str, Any]:
    raw_mask = make_color_mask(image, mask_mode)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = fill_holes(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "failed", "failure_reason": "no_color_contour", "mask": mask, "mask_mode": mask_mode}

    contour = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    if area_px < 100:
        return {"status": "failed", "failure_reason": "largest_contour_too_small", "mask": mask, "area_px": area_px, "mask_mode": mask_mode}

    final_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)

    equivalent_diameter_px = float(math.sqrt(4.0 * area_px / math.pi))
    ellipse = None
    ellipse_area_px = np.nan
    axis_source = "fitEllipse"
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (axis_a, axis_b), _ = ellipse
            long_axis_px = float(max(axis_a, axis_b))
            short_axis_px = float(min(axis_a, axis_b))
            ellipse_area_px = float(math.pi * (long_axis_px / 2.0) * (short_axis_px / 2.0))
        except cv2.error:
            ellipse = None
    if ellipse is None:
        rect = cv2.minAreaRect(contour)
        (_, _), (side_a, side_b), _ = rect
        long_axis_px = float(max(side_a, side_b, equivalent_diameter_px))
        short_axis_px = float(min(max(min(side_a, side_b), 1.0), long_axis_px))
        ellipse_area_px = float(math.pi * (long_axis_px / 2.0) * (short_axis_px / 2.0))
        axis_source = "minAreaRect"
    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)

    return {
        "status": "ok",
        "failure_reason": "",
        "mask_mode": mask_mode,
        "mask": final_mask,
        "contour": contour,
        "ellipse": ellipse,
        "area_px": area_px,
        "contour_area_px": area_px,
        "long_axis_px": long_axis_px,
        "short_axis_px": short_axis_px,
        "equivalent_diameter_px": equivalent_diameter_px,
        "axis_source": axis_source,
        "bbox_x": int(bbox_x),
        "bbox_y": int(bbox_y),
        "bbox_w": int(bbox_w),
        "bbox_h": int(bbox_h),
        "ellipse_area_px": ellipse_area_px,
        "area_to_ellipse_ratio": float(area_px / ellipse_area_px) if ellipse_area_px and not np.isnan(ellipse_area_px) else np.nan,
    }


def compute_volumes(area_px: float, long_px: float, short_px: float, d_eq_px: float, scale: float) -> dict[str, float]:
    area_cm2 = area_px * scale**2
    l_cm = long_px * scale
    w_cm = short_px * scale
    d_eq_cm = d_eq_px * scale
    d_area = 2.0 * math.sqrt(area_cm2 / math.pi)
    d_mean = (l_cm + w_cm) / 2.0
    h_mean = d_mean
    h_geom = math.sqrt(max(l_cm * w_cm, 0.0))

    return {
        "A_cm2": area_cm2,
        "L_cm": l_cm,
        "W_cm": w_cm,
        "D_eq_cm": d_eq_cm,
        "V1_area_sphere": math.pi / 6.0 * d_area**3,
        "V2_axis_mean_sphere": math.pi / 6.0 * d_mean**3,
        "V3_ellipsoid_H_eq_W": math.pi / 6.0 * l_cm * w_cm * w_cm,
        "V4_ellipsoid_H_eq_mean": math.pi / 6.0 * l_cm * w_cm * h_mean,
        "V5_ellipsoid_H_eq_geom": math.pi / 6.0 * l_cm * w_cm * h_geom,
    }


def draw_panel_label(image: np.ndarray, label: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)


def fit_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    max_w, max_h = size
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((max_h, max_w, 3), 245, dtype=np.uint8)
    y = (max_h - resized.shape[0]) // 2
    x = (max_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_debug_image(row: dict[str, Any], image: np.ndarray, seg: dict[str, Any]) -> np.ndarray:
    overlay = image.copy()
    if seg.get("contour") is not None:
        cv2.drawContours(overlay, [seg["contour"]], -1, (0, 255, 0), 3)
    if seg.get("ellipse") is not None:
        cv2.ellipse(overlay, seg["ellipse"], (255, 0, 255), 2)

    mask_bgr = cv2.cvtColor(seg.get("mask", np.zeros(image.shape[:2], dtype=np.uint8)), cv2.COLOR_GRAY2BGR)
    original_panel = fit_image(image, (360, 270))
    mask_panel = fit_image(mask_bgr, (360, 270))
    overlay_panel = fit_image(overlay, (360, 270))
    draw_panel_label(original_panel, "ROI")
    draw_panel_label(mask_panel, f"{row.get('mask_mode', 'red')} mask")
    draw_panel_label(overlay_panel, "contour + ellipse")

    text_panel = np.full((270, 360, 3), 255, dtype=np.uint8)
    lines = [
        f"idx: {int(row['idx']):03d}",
        f"mode: {row.get('mask_mode', '')}",
        f"status: {row['status']}",
        f"weight_g: {row['weight_g']:.2f}",
        f"area_px: {row.get('area_px', np.nan):.1f}",
        f"L_cm: {row.get('L_cm', np.nan):.2f}",
        f"W_cm: {row.get('W_cm', np.nan):.2f}",
        f"D_eq_cm: {row.get('D_eq_cm', np.nan):.2f}",
        f"V1: {row.get('V1_area_sphere', np.nan):.1f}",
        f"V2: {row.get('V2_axis_mean_sphere', np.nan):.1f}",
        f"V3: {row.get('V3_ellipsoid_H_eq_W', np.nan):.1f}",
        f"V4: {row.get('V4_ellipsoid_H_eq_mean', np.nan):.1f}",
        f"V5: {row.get('V5_ellipsoid_H_eq_geom', np.nan):.1f}",
    ]
    if row["status"] != "ok":
        lines.append(f"reason: {row.get('failure_reason', '')}")
    for i, line in enumerate(lines[:13]):
        cv2.putText(text_panel, line, (10, 24 + i * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    top = np.hstack([original_panel, mask_panel])
    bottom = np.hstack([overlay_panel, text_panel])
    return np.vstack([top, bottom])


def make_contact_sheet(debug_paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for path in debug_paths:
        image = imread_unicode(path)
        thumbs.append(fit_image(image, (360, 270)))
    if not thumbs:
        return

    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * 270, cols * 360, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * 270 : (r + 1) * 270, c * 360 : (c + 1) * 360] = thumb
    imwrite_unicode(output_path, sheet)


def run_segmentation(table: pd.DataFrame, scale: float, mask_mode: str, debug_dir: Path, contact_sheet_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    debug_paths: list[Path] = []
    for record in table.itertuples(index=False):
        idx = int(record.idx)
        image = imread_unicode(Path(record.roi_image))
        seg = segment_tomato(image, mask_mode)
        roi_h, roi_w = image.shape[:2]
        row: dict[str, Any] = {
            "idx": idx,
            "mask_mode": mask_mode,
            "roi_image": record.roi_image,
            "weight_g": float(record.weight_g),
            "status": seg.get("status", "failed"),
            "failure_reason": seg.get("failure_reason", ""),
            "axis_source": seg.get("axis_source", ""),
            "roi_width": roi_w,
            "roi_height": roi_h,
        }
        if row["status"] == "ok":
            row.update(
                {
                    "area_px": seg["area_px"],
                    "mask_area_ratio": seg["area_px"] / float(roi_w * roi_h),
                    "bbox_x": seg["bbox_x"],
                    "bbox_y": seg["bbox_y"],
                    "bbox_w": seg["bbox_w"],
                    "bbox_h": seg["bbox_h"],
                    "contour_area_px": seg["contour_area_px"],
                    "ellipse_area_px": seg["ellipse_area_px"],
                    "area_to_ellipse_ratio": seg["area_to_ellipse_ratio"],
                    "long_axis_px": seg["long_axis_px"],
                    "short_axis_px": seg["short_axis_px"],
                    "equivalent_diameter_px": seg["equivalent_diameter_px"],
                }
            )
            row.update(compute_volumes(seg["area_px"], seg["long_axis_px"], seg["short_axis_px"], seg["equivalent_diameter_px"], scale))

        debug = make_debug_image(row, image, seg)
        debug_path = debug_dir / f"idx_{idx:03d}_debug.jpg"
        imwrite_unicode(debug_path, debug)
        debug_paths.append(debug_path)
        rows.append(row)

    make_contact_sheet(debug_paths, contact_sheet_path)
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def density_values(train_df: pd.DataFrame, volume_col: str) -> dict[str, float]:
    valid = train_df[(train_df["status"] == "ok") & (train_df[volume_col] > 0)].copy()
    if valid.empty:
        return {key: np.nan for key in ["mean", "median", "lstsq", "density_std", "density_cv", "train_n"]}
    density = valid["weight_g"] / valid[volume_col]
    mean_rho = float(density.mean())
    density_std = float(density.std(ddof=1)) if len(density) > 1 else 0.0
    return {
        "mean": mean_rho,
        "median": float(density.median()),
        "lstsq": float((valid[volume_col] * valid["weight_g"]).sum() / (valid[volume_col] ** 2).sum()),
        "density_std": density_std,
        "density_cv": float(density_std / mean_rho) if mean_rho else np.nan,
        "train_n": int(len(valid)),
    }


def error_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100.0) if nonzero.any() else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_holdout_experiment(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = features_df[["idx", "mask_mode", "weight_g", "status", "failure_reason"]].copy()
    results: list[dict[str, Any]] = []
    sorted_df = features_df.sort_values("idx").reset_index(drop=True)
    train_df = sorted_df.iloc[:TRAIN_COUNT]
    val_df = sorted_df.iloc[TRAIN_COUNT:]

    for volume_col in VOLUME_COLUMNS:
        densities = density_values(train_df, volume_col)
        for method in DENSITY_METHODS:
            rho = densities[method]
            pred_col = f"pred_{volume_col}_{method}"
            err_col = f"err_{volume_col}_{method}"
            abs_err_col = f"abs_err_{volume_col}_{method}"
            predictions[pred_col] = np.where(features_df["status"] == "ok", features_df[volume_col] * rho, np.nan)
            predictions[err_col] = predictions[pred_col] - predictions["weight_g"]
            predictions[abs_err_col] = predictions[err_col].abs()

            val_pred = predictions.loc[val_df.index, pred_col]
            val_true = predictions.loc[val_df.index, "weight_g"]
            valid_eval = val_pred.notna() & val_true.notna()
            metrics = error_metrics(val_true[valid_eval], val_pred[valid_eval]) if valid_eval.any() else {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
            results.append(
                {
                    "volume_formula": volume_col,
                    "density_method": method,
                    "rho": rho,
                    "mean_rho": densities["mean"],
                    "median_rho": densities["median"],
                    "rho_lstsq": densities["lstsq"],
                    "density_std": densities["density_std"],
                    "density_cv": densities["density_cv"],
                    "train_n": densities["train_n"],
                    "val_n": int(valid_eval.sum()),
                    **metrics,
                }
            )

    return predictions, pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)


def run_cross_validation(features_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = features_df[features_df["status"] == "ok"].sort_values("idx").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    if len(valid_df) < 5:
        return pd.DataFrame(rows)

    kf = KFold(n_splits=5, shuffle=False)
    for volume_col in VOLUME_COLUMNS:
        for method in DENSITY_METHODS:
            fold_metrics = []
            fold_density_std = []
            fold_density_cv = []
            for fold, (train_index, val_index) in enumerate(kf.split(valid_df), start=1):
                train_df = valid_df.iloc[train_index]
                val_df = valid_df.iloc[val_index]
                densities = density_values(train_df, volume_col)
                rho = densities[method]
                pred = val_df[volume_col] * rho
                metrics = error_metrics(val_df["weight_g"], pred)
                fold_metrics.append(metrics)
                fold_density_std.append(densities["density_std"])
                fold_density_cv.append(densities["density_cv"])
                rows.append(
                    {
                        "volume_formula": volume_col,
                        "density_method": method,
                        "fold": fold,
                        "rho": rho,
                        "val_idx": ",".join(str(v) for v in val_df["idx"].tolist()),
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "MAPE": metrics["MAPE"],
                        "density_std": densities["density_std"],
                        "density_cv": densities["density_cv"],
                    }
                )
            rows.append(
                {
                    "volume_formula": volume_col,
                    "density_method": method,
                    "fold": "mean",
                    "rho": np.nan,
                    "val_idx": "",
                    "MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
                    "RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
                    "MAPE": float(np.mean([m["MAPE"] for m in fold_metrics])),
                    "density_std": float(np.mean(fold_density_std)),
                    "density_cv": float(np.mean(fold_density_cv)),
                }
            )
    return pd.DataFrame(rows)


def write_summary(features_df: pd.DataFrame, predictions: pd.DataFrame, results: pd.DataFrame, cv_results: pd.DataFrame, output_path: Path) -> None:
    best = results.sort_values("MAE").head(1).copy()
    if not best.empty:
        best["recommendation"] = best.apply(
            lambda r: f"Best holdout MAE: {r['volume_formula']} with {r['density_method']} density, MAE={r['MAE']:.3f} g",
            axis=1,
        )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        features_df.to_excel(writer, sheet_name="volume_features", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)
        results.to_excel(writer, sheet_name="experiment_results", index=False)
        cv_results.to_excel(writer, sheet_name="cv_results", index=False)
        best.to_excel(writer, sheet_name="recommendation", index=False)


def best_prediction_columns(results: pd.DataFrame) -> tuple[str, str, str, str]:
    best = results.sort_values("MAE").iloc[0]
    volume_formula = str(best["volume_formula"])
    density_method = str(best["density_method"])
    pred_col = f"pred_{volume_formula}_{density_method}"
    abs_err_col = f"abs_err_{volume_formula}_{density_method}"
    return volume_formula, density_method, pred_col, abs_err_col


def load_mode_outputs(mask_mode: str) -> dict[str, pd.DataFrame] | None:
    paths = mode_paths(mask_mode)
    features_path = paths["results_dir"] / "volume_features.csv"
    predictions_path = paths["results_dir"] / "predictions.csv"
    results_path = paths["results_dir"] / "experiment_results.csv"
    cv_path = paths["results_dir"] / "cv_results.csv"
    if not all(p.exists() for p in [features_path, predictions_path, results_path, cv_path]):
        return None
    return {
        "features": pd.read_csv(features_path),
        "predictions": pd.read_csv(predictions_path),
        "results": pd.read_csv(results_path),
        "cv": pd.read_csv(cv_path),
    }


def write_mask_mode_comparison_if_possible() -> None:
    red = load_mode_outputs("red")
    warm = load_mode_outputs("warm")
    if red is None or warm is None:
        return

    comparison: pd.DataFrame | None = None
    for mode, data in [("red", red), ("warm", warm)]:
        volume_formula, density_method, pred_col, abs_err_col = best_prediction_columns(data["results"])
        mode_df = data["features"][
            ["idx", "area_px", "L_cm", "W_cm", "V3_ellipsoid_H_eq_W", "area_to_ellipse_ratio", "mask_area_ratio"]
        ].copy()
        mode_df = mode_df.merge(data["predictions"][["idx", pred_col, abs_err_col]], on="idx", how="left")
        mode_df = mode_df.rename(
            columns={
                "area_px": f"{mode}_area_px",
                "L_cm": f"{mode}_L_cm",
                "W_cm": f"{mode}_W_cm",
                "V3_ellipsoid_H_eq_W": f"{mode}_V3_ellipsoid_H_eq_W",
                "area_to_ellipse_ratio": f"{mode}_area_to_ellipse_ratio",
                "mask_area_ratio": f"{mode}_mask_area_ratio",
                pred_col: f"{mode}_predicted_weight_best",
                abs_err_col: f"{mode}_absolute_error_best",
            }
        )
        mode_df[f"{mode}_best_volume_formula"] = volume_formula
        mode_df[f"{mode}_best_density_method"] = density_method
        comparison = mode_df if comparison is None else comparison.merge(mode_df, on="idx", how="outer")

    comparison = comparison.sort_values("idx").reset_index(drop=True)
    comparison.to_csv(OUTPUT_DIR / "mask_mode_comparison.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    for mode, data in [("red", red), ("warm", warm)]:
        best = data["results"].sort_values("MAE").iloc[0]
        cv_mean = data["cv"][
            (data["cv"]["fold"].astype(str) == "mean")
            & (data["cv"]["volume_formula"] == best["volume_formula"])
            & (data["cv"]["density_method"] == best["density_method"])
        ]
        summary_rows.append(
            {
                "mask_mode": mode,
                "best_volume_formula": best["volume_formula"],
                "best_density_method": best["density_method"],
                "MAE": best["MAE"],
                "RMSE": best["RMSE"],
                "MAPE": best["MAPE"],
                "CV_MAE_mean": float(cv_mean["MAE"].iloc[0]) if not cv_mean.empty else np.nan,
                "CV_MAPE_mean": float(cv_mean["MAPE"].iloc[0]) if not cv_mean.empty else np.nan,
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "mask_mode_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tomato ROI segmentation, volume, density, and validation experiments.")
    parser.add_argument("--reset-scale", action="store_true", help="Re-click the 5cm cube edge even if scale_config.json exists.")
    parser.add_argument(
        "--mask-mode",
        choices=["red", "warm"],
        default="red",
        help="HSV segmentation mode. red keeps the original strict red mask; warm also includes orange/yellow tomato pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = mode_paths(args.mask_mode)
    paths["debug_dir"].mkdir(parents=True, exist_ok=True)
    DEBUG_SCALE_DIR.mkdir(parents=True, exist_ok=True)
    paths["results_dir"].mkdir(parents=True, exist_ok=True)

    weights_df = load_weights()
    table = load_roi_table(weights_df)
    first = table.iloc[0]
    first_image = imread_unicode(Path(first["roi_image"]))
    scale_config = load_or_select_scale(int(first["idx"]), first_image, args.reset_scale)
    scale = float(scale_config["scale_cm_per_px"])
    if SCALE_CONFIG_JSON.exists() and (DEBUG_SCALE_DIR / "scale_reference.jpg").exists() is False:
        draw_scale_reference(first_image, scale_config)

    features_df = run_segmentation(table, scale, args.mask_mode, paths["debug_dir"], paths["contact_sheet"])
    features_df.to_csv(paths["results_dir"] / "volume_features.csv", index=False, encoding="utf-8-sig")

    predictions, results = run_holdout_experiment(features_df)
    cv_results = run_cross_validation(features_df)
    predictions.to_csv(paths["results_dir"] / "predictions.csv", index=False, encoding="utf-8-sig")
    results.to_csv(paths["results_dir"] / "experiment_results.csv", index=False, encoding="utf-8-sig")
    cv_results.to_csv(paths["results_dir"] / "cv_results.csv", index=False, encoding="utf-8-sig")
    write_summary(features_df, predictions, results, cv_results, paths["summary_xlsx"])
    write_mask_mode_comparison_if_possible()

    failed_n = int((features_df["status"] != "ok").sum())
    print(f"Processed {len(features_df)} ROI images with mask_mode={args.mask_mode}. Segmentation failures: {failed_n}.")
    print(f"Wrote: {paths['results_dir'].relative_to(PROJECT_ROOT)}\\volume_features.csv")
    print(f"Wrote: {paths['results_dir'].relative_to(PROJECT_ROOT)}\\predictions.csv")
    print(f"Wrote: {paths['results_dir'].relative_to(PROJECT_ROOT)}\\experiment_results.csv")
    print(f"Wrote: {paths['results_dir'].relative_to(PROJECT_ROOT)}\\cv_results.csv")
    print(f"Wrote: {paths['summary_xlsx'].relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {paths['contact_sheet'].relative_to(PROJECT_ROOT)}")
    if (OUTPUT_DIR / "mask_mode_comparison.csv").exists() and (OUTPUT_DIR / "mask_mode_summary.csv").exists():
        print("Wrote: outputs\\mask_mode_comparison.csv")
        print("Wrote: outputs\\mask_mode_summary.csv")


if __name__ == "__main__":
    main()
