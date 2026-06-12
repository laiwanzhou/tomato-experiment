from __future__ import annotations

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
SEG_PRED_DIR = OUTPUT_DIR / "seg_roi_predictions"
SCALE_CONFIG_JSON = OUTPUT_DIR / "scale_config.json"
SUMMARY_RED_XLSX = OUTPUT_DIR / "summary_red.xlsx"

DEBUG_DIR = OUTPUT_DIR / "debug_masks_seg"
RESULTS_DIR = OUTPUT_DIR / "volume_results_seg"
CONTACT_SHEET = OUTPUT_DIR / "debug_masks_contact_sheet_seg.jpg"
SUMMARY_SEG_XLSX = OUTPUT_DIR / "summary_seg.xlsx"

TRAIN_COUNT = 20
VOLUME_COLUMNS = [
    "V1_area_sphere",
    "V2_axis_mean_sphere",
    "V3_ellipsoid_H_eq_W",
    "V4_ellipsoid_H_eq_mean",
    "V5_ellipsoid_H_eq_geom",
]
DENSITY_METHODS = ["mean", "median", "lstsq"]


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
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
    """Read CSV/TSV files that may be UTF-8, GBK, comma separated, or tab separated."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            df.columns = [str(col).strip() for col in df.columns]
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Could not decode {path}; tried {errors}")


def parse_idx_from_name(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 2 or parts[0] != "idx":
        raise ValueError(f"Unexpected idx filename format: {path.name}")
    return int(parts[1])


def load_weights() -> pd.DataFrame:
    if not WEIGHTS_CSV.exists():
        raise FileNotFoundError("Missing outputs/weights.csv.")
    df = read_csv_with_fallback(WEIGHTS_CSV)
    missing = {"idx", "weight_g"} - set(df.columns)
    if missing:
        raise ValueError(f"outputs/weights.csv is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
    df["weight_g"] = pd.to_numeric(df["weight_g"], errors="coerce")
    bad = df[df["idx"].isna() | df["weight_g"].isna()]
    if not bad.empty:
        rows = ", ".join(str(i + 2) for i in bad.index.tolist())
        raise ValueError(f"weights.csv has blank or non-numeric idx/weight_g values. Problem CSV rows: {rows}")
    df["idx"] = df["idx"].astype(int)
    return df.sort_values("idx").reset_index(drop=True)


def load_input_table(weights_df: pd.DataFrame) -> pd.DataFrame:
    roi_paths = sorted(ROI_DIR.glob("idx_*_roi.*"), key=parse_idx_from_name)
    if not roi_paths:
        raise FileNotFoundError("No ROI images found in outputs/roi_tomato_images.")

    rows = []
    for roi_path in roi_paths:
        idx = parse_idx_from_name(roi_path)
        rows.append(
            {
                "idx": idx,
                "roi_image": str(roi_path),
                "seg_mask": str(SEG_PRED_DIR / f"idx_{idx:03d}_pred_mask.png"),
            }
        )
    table = weights_df.merge(pd.DataFrame(rows), on="idx", how="left")
    missing_roi = table[table["roi_image"].isna()]
    if not missing_roi.empty:
        raise ValueError(f"Missing ROI images for idx: {missing_roi['idx'].tolist()}")
    return table.sort_values("idx").reset_index(drop=True)


def load_scale() -> float:
    if not SCALE_CONFIG_JSON.exists():
        raise FileNotFoundError("Missing outputs/scale_config.json. Run the HSV volume script scale calibration first.")
    with SCALE_CONFIG_JSON.open("r", encoding="utf-8") as f:
        config = json.load(f)
    scale = float(config["scale_cm_per_px"])
    if scale <= 0:
        raise ValueError("scale_cm_per_px must be positive.")
    return scale


def load_binary_mask(mask_path: Path, roi_shape: tuple[int, int], idx: int, warnings: list[dict[str, Any]]) -> np.ndarray | None:
    if not mask_path.exists():
        warnings.append({"idx": idx, "warning_type": "missing_mask", "message": str(mask_path)})
        return None
    mask = imread_unicode(mask_path, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
        warnings.append({"idx": idx, "warning_type": "mask_three_channel", "message": "Used first channel."})
    mask = (mask > 0).astype(np.uint8)
    roi_h, roi_w = roi_shape
    if mask.shape[:2] != (roi_h, roi_w):
        warnings.append(
            {
                "idx": idx,
                "warning_type": "mask_size_mismatch",
                "message": f"mask {mask.shape[:2]} resized to ROI {(roi_h, roi_w)}",
            }
        )
        mask = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
    return mask


def largest_component(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas) + 1)
    component = (labels == best_label).astype(np.uint8)
    return component, float(stats[best_label, cv2.CC_STAT_AREA])


def extract_features_from_mask(mask: np.ndarray) -> dict[str, Any]:
    component_mask, component_area_px = largest_component(mask)
    if component_mask is None or component_area_px <= 0:
        return {"status": "failed", "failure_reason": "no_connected_component", "mask": np.zeros_like(mask)}

    contours, _ = cv2.findContours(component_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "failed", "failure_reason": "no_component_contour", "mask": component_mask * 255}
    contour = max(contours, key=cv2.contourArea)
    contour_area_px = float(cv2.contourArea(contour))
    if component_area_px < 20:
        return {
            "status": "failed",
            "failure_reason": "component_too_small",
            "mask": component_mask * 255,
            "component_area_px": component_area_px,
            "contour_area_px": contour_area_px,
        }

    equivalent_diameter_px = float(math.sqrt(4.0 * component_area_px / math.pi))
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
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
        long_axis_px = float(max(bbox_w, bbox_h, equivalent_diameter_px))
        short_axis_px = float(max(1.0, min(bbox_w, bbox_h, long_axis_px)))
        ellipse_area_px = float(math.pi * (long_axis_px / 2.0) * (short_axis_px / 2.0))
        axis_source = "boundingRect"
    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)

    return {
        "status": "ok",
        "failure_reason": "",
        "mask": component_mask * 255,
        "contour": contour,
        "ellipse": ellipse,
        "axis_source": axis_source,
        "area_px": component_area_px,
        "component_area_px": component_area_px,
        "contour_area_px": contour_area_px,
        "bbox_x": int(bbox_x),
        "bbox_y": int(bbox_y),
        "bbox_w": int(bbox_w),
        "bbox_h": int(bbox_h),
        "long_axis_px": long_axis_px,
        "short_axis_px": short_axis_px,
        "equivalent_diameter_px": equivalent_diameter_px,
        "ellipse_area_px": ellipse_area_px,
        "area_to_ellipse_ratio": float(component_area_px / ellipse_area_px) if ellipse_area_px else np.nan,
    }


def compute_volumes(area_px: float, long_px: float, short_px: float, d_eq_px: float, scale: float) -> dict[str, float]:
    area_cm2 = area_px * scale**2
    l_cm = long_px * scale
    w_cm = short_px * scale
    d_eq_cm = d_eq_px * scale
    d_area = 2.0 * math.sqrt(area_cm2 / math.pi)
    d_mean = (l_cm + w_cm) / 2.0
    h_geom = math.sqrt(max(l_cm * w_cm, 0.0))
    return {
        "A_cm2": area_cm2,
        "L_cm": l_cm,
        "W_cm": w_cm,
        "D_eq_cm": d_eq_cm,
        "V1_area_sphere": math.pi / 6.0 * d_area**3,
        "V2_axis_mean_sphere": math.pi / 6.0 * d_mean**3,
        "V3_ellipsoid_H_eq_W": math.pi / 6.0 * l_cm * w_cm * w_cm,
        "V4_ellipsoid_H_eq_mean": math.pi / 6.0 * l_cm * w_cm * d_mean,
        "V5_ellipsoid_H_eq_geom": math.pi / 6.0 * l_cm * w_cm * h_geom,
    }


def fit_image(image: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    max_w, max_h = size
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interpolation)
    canvas = np.full((max_h, max_w, 3), 245, dtype=np.uint8)
    y = (max_h - resized.shape[0]) // 2
    x = (max_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def draw_label(image: np.ndarray, label: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)


def make_debug_image(row: dict[str, Any], roi: np.ndarray, raw_mask: np.ndarray | None, features: dict[str, Any]) -> np.ndarray:
    if raw_mask is None:
        raw_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    raw_mask_vis = (raw_mask > 0).astype(np.uint8) * 255
    component_vis = features.get("mask", np.zeros(roi.shape[:2], dtype=np.uint8))

    overlay = roi.copy()
    color_mask = np.zeros_like(roi)
    color_mask[:, :, 1] = component_vis
    overlay = cv2.addWeighted(overlay, 0.72, color_mask, 0.28, 0)
    if features.get("contour") is not None:
        cv2.drawContours(overlay, [features["contour"]], -1, (0, 255, 0), 3)
    if features.get("ellipse") is not None:
        cv2.ellipse(overlay, features["ellipse"], (255, 0, 255), 2)
    if row.get("bbox_w") and row.get("bbox_h"):
        x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 255), 2)

    panels = [
        fit_image(roi, (330, 250)),
        fit_image(cv2.cvtColor(raw_mask_vis, cv2.COLOR_GRAY2BGR), (330, 250), cv2.INTER_NEAREST),
        fit_image(cv2.cvtColor(component_vis, cv2.COLOR_GRAY2BGR), (330, 250), cv2.INTER_NEAREST),
        fit_image(overlay, (330, 250)),
    ]
    for panel, label in zip(panels, ["ROI", "pred mask x255", "largest component", "overlay"]):
        draw_label(panel, label)

    text_panel = np.full((250, 330, 3), 255, dtype=np.uint8)
    lines = [
        f"idx: {int(row['idx']):03d}",
        f"status: {row['status']}",
        f"weight_g: {row['weight_g']:.2f}",
        f"area_px: {row.get('area_px', np.nan):.1f}",
        f"L_cm: {row.get('L_cm', np.nan):.3f}",
        f"W_cm: {row.get('W_cm', np.nan):.3f}",
        f"D_eq_cm: {row.get('D_eq_cm', np.nan):.3f}",
        f"V3: {row.get('V3_ellipsoid_H_eq_W', np.nan):.3f}",
        f"ratio: {row.get('area_to_ellipse_ratio', np.nan):.3f}",
    ]
    if row["status"] != "ok":
        lines.append(f"reason: {row.get('failure_reason', '')}")
    for i, line in enumerate(lines[:11]):
        cv2.putText(text_panel, line, (10, 24 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    draw_label(text_panel, "features")

    top = np.hstack(panels[:3])
    bottom = np.hstack([panels[3], text_panel, np.full((250, 330, 3), 245, dtype=np.uint8)])
    return np.vstack([top, bottom])


def make_contact_sheet(debug_paths: list[Path], output_path: Path) -> None:
    thumbs = []
    for path in debug_paths:
        image = imread_unicode(path)
        thumb = fit_image(image, (360, 240))
        draw_label(thumb, path.stem)
        thumbs.append(thumb)
    if not thumbs:
        return
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * 240, cols * 360, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * 240 : (r + 1) * 240, c * 360 : (c + 1) * 360] = thumb
    imwrite_unicode(output_path, sheet)


def build_features(table: pd.DataFrame, scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    debug_paths: list[Path] = []

    for record in table.itertuples(index=False):
        idx = int(record.idx)
        roi_path = Path(record.roi_image)
        mask_path = Path(record.seg_mask)
        roi = imread_unicode(roi_path, cv2.IMREAD_COLOR)
        roi_h, roi_w = roi.shape[:2]
        raw_mask = load_binary_mask(mask_path, (roi_h, roi_w), idx, warnings)
        features = (
            {"status": "failed", "failure_reason": "missing_mask", "mask": np.zeros((roi_h, roi_w), dtype=np.uint8)}
            if raw_mask is None
            else extract_features_from_mask(raw_mask)
        )

        row: dict[str, Any] = {
            "idx": idx,
            "mask_source": "seg-mask",
            "roi_image": str(roi_path),
            "seg_mask": str(mask_path),
            "weight_g": float(record.weight_g),
            "status": features.get("status", "failed"),
            "failure_reason": features.get("failure_reason", ""),
            "axis_source": features.get("axis_source", ""),
            "roi_width": roi_w,
            "roi_height": roi_h,
        }
        if row["status"] == "ok":
            row.update(
                {
                    "area_px": features["area_px"],
                    "component_area_px": features["component_area_px"],
                    "mask_area_ratio": features["area_px"] / float(roi_w * roi_h),
                    "bbox_x": features["bbox_x"],
                    "bbox_y": features["bbox_y"],
                    "bbox_w": features["bbox_w"],
                    "bbox_h": features["bbox_h"],
                    "contour_area_px": features["contour_area_px"],
                    "ellipse_area_px": features["ellipse_area_px"],
                    "area_to_ellipse_ratio": features["area_to_ellipse_ratio"],
                    "long_axis_px": features["long_axis_px"],
                    "short_axis_px": features["short_axis_px"],
                    "equivalent_diameter_px": features["equivalent_diameter_px"],
                }
            )
            row.update(
                compute_volumes(
                    features["area_px"],
                    features["long_axis_px"],
                    features["short_axis_px"],
                    features["equivalent_diameter_px"],
                    scale,
                )
            )
        else:
            warnings.append({"idx": idx, "warning_type": "feature_failed", "message": row["failure_reason"]})

        debug_path = DEBUG_DIR / f"idx_{idx:03d}_seg_debug.jpg"
        imwrite_unicode(debug_path, make_debug_image(row, roi, raw_mask, features))
        debug_paths.append(debug_path)
        rows.append(row)

    make_contact_sheet(debug_paths, CONTACT_SHEET)
    warnings_df = pd.DataFrame(warnings, columns=["idx", "warning_type", "message"])
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True), warnings_df


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
    sorted_df = features_df.sort_values("idx").reset_index(drop=True)
    train_df = sorted_df.iloc[:TRAIN_COUNT]
    val_df = sorted_df.iloc[TRAIN_COUNT:]
    prediction_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for volume_col in VOLUME_COLUMNS:
        densities = density_values(train_df, volume_col)
        for method in DENSITY_METHODS:
            rho = densities[method]
            for row in sorted_df.itertuples(index=False):
                volume = getattr(row, volume_col, np.nan)
                pred = float(volume * rho) if row.status == "ok" and pd.notna(volume) and pd.notna(rho) else np.nan
                err = pred - float(row.weight_g) if pd.notna(pred) else np.nan
                split = "train" if int(row.idx) <= TRAIN_COUNT else "val"
                prediction_rows.append(
                    {
                        "idx": int(row.idx),
                        "split": split,
                        "weight_g": float(row.weight_g),
                        "status": row.status,
                        "volume_formula": volume_col,
                        "density_method": method,
                        "volume_cm3": volume,
                        "rho": rho,
                        "pred_weight_g": pred,
                        "error_g": err,
                        "abs_error_g": abs(err) if pd.notna(err) else np.nan,
                        "abs_pct_error": abs(err) / float(row.weight_g) * 100.0 if pd.notna(err) and row.weight_g else np.nan,
                    }
                )

            method_val = pd.DataFrame(prediction_rows)
            method_val = method_val[
                (method_val["split"] == "val")
                & (method_val["volume_formula"] == volume_col)
                & (method_val["density_method"] == method)
                & method_val["pred_weight_g"].notna()
            ]
            metrics = (
                error_metrics(method_val["weight_g"], method_val["pred_weight_g"])
                if not method_val.empty
                else {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
            )
            result_rows.append(
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
                    "val_n": int(len(method_val)),
                    **metrics,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    results = pd.DataFrame(result_rows).sort_values("MAE").reset_index(drop=True)
    return predictions, results


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


def read_hsv_red_outputs(warnings: list[dict[str, Any]]) -> dict[str, pd.DataFrame] | None:
    if not SUMMARY_RED_XLSX.exists():
        warnings.append({"idx": "", "warning_type": "missing_summary_red", "message": str(SUMMARY_RED_XLSX)})
        return None
    try:
        return {
            "experiment_results": pd.read_excel(SUMMARY_RED_XLSX, sheet_name="experiment_results"),
            "predictions": pd.read_excel(SUMMARY_RED_XLSX, sheet_name="predictions"),
        }
    except Exception as exc:
        warnings.append({"idx": "", "warning_type": "read_summary_red_failed", "message": str(exc)})
        return None


def best_row(results: pd.DataFrame) -> pd.Series:
    return results.sort_values("MAE", na_position="last").iloc[0]


def build_seg_vs_red_comparison(
    seg_results: pd.DataFrame,
    seg_predictions: pd.DataFrame,
    red_outputs: dict[str, pd.DataFrame] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg_best = best_row(seg_results)
    method_rows = [
        {
            "method_source": "Seg-mask",
            "volume_formula": seg_best["volume_formula"],
            "density_method": seg_best["density_method"],
            "MAE": seg_best["MAE"],
            "RMSE": seg_best["RMSE"],
            "MAPE": seg_best["MAPE"],
            "rho": seg_best["rho"],
        }
    ]

    seg_sample = seg_predictions[
        (seg_predictions["split"] == "val")
        & (seg_predictions["volume_formula"] == seg_best["volume_formula"])
        & (seg_predictions["density_method"] == seg_best["density_method"])
    ][["idx", "weight_g", "pred_weight_g", "abs_error_g", "abs_pct_error"]].copy()
    seg_sample = seg_sample.rename(
        columns={
            "pred_weight_g": "seg_pred_weight_g",
            "abs_error_g": "seg_abs_error_g",
            "abs_pct_error": "seg_abs_pct_error",
        }
    )

    if red_outputs is None:
        sample_compare = seg_sample.copy()
        for col in ["hsv_pred_weight_g", "hsv_abs_error_g", "hsv_abs_pct_error", "seg_minus_hsv_abs_error"]:
            sample_compare[col] = np.nan
        return pd.DataFrame(method_rows), sample_compare

    red_best = best_row(red_outputs["experiment_results"])
    method_rows.insert(
        0,
        {
            "method_source": "HSV-red",
            "volume_formula": red_best["volume_formula"],
            "density_method": red_best["density_method"],
            "MAE": red_best["MAE"],
            "RMSE": red_best["RMSE"],
            "MAPE": red_best["MAPE"],
            "rho": red_best["rho"],
        },
    )

    red_pred_col = f"pred_{red_best['volume_formula']}_{red_best['density_method']}"
    red_abs_col = f"abs_err_{red_best['volume_formula']}_{red_best['density_method']}"
    red_predictions = red_outputs["predictions"].copy()
    red_sample = red_predictions[red_predictions["idx"] > TRAIN_COUNT][["idx", "weight_g", red_pred_col, red_abs_col]].copy()
    red_sample = red_sample.rename(columns={red_pred_col: "hsv_pred_weight_g", red_abs_col: "hsv_abs_error_g"})
    red_sample["hsv_abs_pct_error"] = red_sample["hsv_abs_error_g"] / red_sample["weight_g"] * 100.0
    sample_compare = red_sample.merge(seg_sample, on=["idx", "weight_g"], how="outer")
    sample_compare["seg_minus_hsv_abs_error"] = sample_compare["seg_abs_error_g"] - sample_compare["hsv_abs_error_g"]
    sample_compare = sample_compare[
        [
            "idx",
            "weight_g",
            "hsv_pred_weight_g",
            "seg_pred_weight_g",
            "hsv_abs_error_g",
            "seg_abs_error_g",
            "hsv_abs_pct_error",
            "seg_abs_pct_error",
            "seg_minus_hsv_abs_error",
        ]
    ].sort_values("idx")
    return pd.DataFrame(method_rows), sample_compare


def build_recommendation(method_compare: pd.DataFrame, sample_compare: pd.DataFrame) -> pd.DataFrame:
    seg = method_compare[method_compare["method_source"] == "Seg-mask"].iloc[0]
    hsv = method_compare[method_compare["method_source"] == "HSV-red"]
    hsv_text = "HSV-red summary unavailable"
    seg_better = np.nan
    if not hsv.empty:
        hsv_row = hsv.iloc[0]
        hsv_text = (
            f"{hsv_row['volume_formula']} + {hsv_row['density_method']}, "
            f"MAE={hsv_row['MAE']:.3f} g, RMSE={hsv_row['RMSE']:.3f} g, MAPE={hsv_row['MAPE']:.3f}%"
        )
        seg_better = bool(seg["MAE"] < hsv_row["MAE"])

    focus_lines = []
    for idx in (30, 32):
        row = sample_compare[sample_compare["idx"] == idx]
        if row.empty:
            focus_lines.append(f"idx{idx}: not found in validation comparison")
            continue
        item = row.iloc[0]
        improved = item["seg_abs_error_g"] < item["hsv_abs_error_g"] if pd.notna(item.get("hsv_abs_error_g")) else np.nan
        focus_lines.append(
            f"idx{idx}: HSV abs={item.get('hsv_abs_error_g', np.nan):.3f} g, "
            f"Seg abs={item.get('seg_abs_error_g', np.nan):.3f} g, improved={improved}"
        )

    conclusion = "Seg-mask holdout MAE is lower than HSV-red." if seg_better is True else "HSV-red holdout MAE is still lower than Seg-mask."
    if pd.isna(seg_better):
        conclusion = "Seg-mask experiment completed; HSV-red comparison was unavailable."
    return pd.DataFrame(
        [
            {"item": "Seg-mask best", "value": f"{seg['volume_formula']} + {seg['density_method']}, MAE={seg['MAE']:.3f} g, RMSE={seg['RMSE']:.3f} g, MAPE={seg['MAPE']:.3f}%"},
            {"item": "HSV-red best", "value": hsv_text},
            {"item": "Seg better than HSV-red", "value": seg_better},
            {"item": "idx30 / idx32", "value": " | ".join(focus_lines)},
            {"item": "Conclusion", "value": conclusion},
        ]
    )


def write_outputs(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    cv_results: pd.DataFrame,
    warnings_df: pd.DataFrame,
    method_compare: pd.DataFrame,
    sample_compare: pd.DataFrame,
    recommendation: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(RESULTS_DIR / "volume_features_seg.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / "predictions_seg.csv", index=False, encoding="utf-8-sig")
    results.to_csv(RESULTS_DIR / "experiment_results_seg.csv", index=False, encoding="utf-8-sig")
    cv_results.to_csv(RESULTS_DIR / "cv_results_seg.csv", index=False, encoding="utf-8-sig")
    sample_compare.to_csv(RESULTS_DIR / "seg_vs_red_comparison.csv", index=False, encoding="utf-8-sig")
    warnings_df.to_csv(RESULTS_DIR / "warnings_seg.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(SUMMARY_SEG_XLSX, engine="openpyxl") as writer:
        features.to_excel(writer, sheet_name="volume_features", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)
        results.to_excel(writer, sheet_name="experiment_results", index=False)
        cv_results.to_excel(writer, sheet_name="cv_results", index=False)
        recommendation.to_excel(writer, sheet_name="recommendation", index=False)
        method_compare.to_excel(writer, sheet_name="method_comparison", index=False)
        sample_compare.to_excel(writer, sheet_name="seg_vs_red_comparison", index=False)
        warnings_df.to_excel(writer, sheet_name="warnings", index=False)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    weights = load_weights()
    table = load_input_table(weights)
    scale = load_scale()
    warning_rows: list[dict[str, Any]] = []

    features, warnings_df = build_features(table, scale)
    if not warnings_df.empty:
        warning_rows.extend(warnings_df.to_dict("records"))

    predictions, results = run_holdout_experiment(features)
    cv_results = run_cross_validation(features)
    red_outputs = read_hsv_red_outputs(warning_rows)
    method_compare, sample_compare = build_seg_vs_red_comparison(results, predictions, red_outputs)
    recommendation = build_recommendation(method_compare, sample_compare)
    all_warnings_df = pd.DataFrame(warning_rows, columns=["idx", "warning_type", "message"])

    write_outputs(features, predictions, results, cv_results, all_warnings_df, method_compare, sample_compare, recommendation)

    seg_best = method_compare[method_compare["method_source"] == "Seg-mask"].iloc[0]
    hsv_best = method_compare[method_compare["method_source"] == "HSV-red"]
    print(
        "Seg best: "
        f"{seg_best['volume_formula']} + {seg_best['density_method']}, "
        f"MAE={seg_best['MAE']:.3f} g, RMSE={seg_best['RMSE']:.3f} g, MAPE={seg_best['MAPE']:.3f}%"
    )
    if not hsv_best.empty:
        hsv = hsv_best.iloc[0]
        better = "Seg-mask" if float(seg_best["MAE"]) < float(hsv["MAE"]) else "HSV-red"
        print(
            "HSV-red best: "
            f"{hsv['volume_formula']} + {hsv['density_method']}, "
            f"MAE={hsv['MAE']:.3f} g, RMSE={hsv['RMSE']:.3f} g, MAPE={hsv['MAPE']:.3f}%"
        )
        print(f"Better holdout MAE: {better}")
    else:
        print("HSV-red best: unavailable; see warnings_seg.csv")

    for idx in (30, 32):
        row = sample_compare[sample_compare["idx"] == idx]
        if row.empty:
            print(f"idx{idx}: not found in validation comparison")
        else:
            item = row.iloc[0]
            print(
                f"idx{idx}: HSV abs={item.get('hsv_abs_error_g', np.nan):.3f} g, "
                f"Seg abs={item.get('seg_abs_error_g', np.nan):.3f} g, "
                f"delta={item.get('seg_minus_hsv_abs_error', np.nan):.3f} g"
            )

    print(f"Warnings: {len(all_warnings_df)}")
    if all_warnings_df.empty:
        print("No warnings recorded.")
    print("Wrote:")
    for path in [
        SUMMARY_SEG_XLSX,
        CONTACT_SHEET,
        RESULTS_DIR / "volume_features_seg.csv",
        RESULTS_DIR / "predictions_seg.csv",
        RESULTS_DIR / "experiment_results_seg.csv",
        RESULTS_DIR / "cv_results_seg.csv",
        RESULTS_DIR / "seg_vs_red_comparison.csv",
        RESULTS_DIR / "warnings_seg.csv",
    ]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
