from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = PACKAGE_DIR / "images"
DEFAULT_WEIGHTS_CSV = PACKAGE_DIR / "weights.csv"
DEFAULT_CHECKPOINT = PACKAGE_DIR / "models" / "best_model.pth"
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "calibration_config.json"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"

FALLBACK_CONFIG = {
    "model": "fcn_resnet50",
    "image_size": 512,
    "volume_formula": "V3_ellipsoid_H_eq_W",
    "density_method": "mean",
    "cube_edge_cm": 5.0,
}


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
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    package_relative = (PACKAGE_DIR / path).resolve()
    if package_relative.exists():
        return package_relative
    return (Path.cwd() / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = FALLBACK_CONFIG.copy()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        config.update({key: value for key, value in loaded.items() if value is not None})
    return config


def read_weights(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing weights csv: {path}")
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    missing = {"image_name", "weight_g"} - set(df.columns)
    if missing:
        raise ValueError(f"weights.csv missing columns: {sorted(missing)}")
    df = df.copy()
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["weight_g"] = pd.to_numeric(df["weight_g"], errors="coerce")
    bad = df[df["image_name"].eq("") | df["weight_g"].isna()]
    if not bad.empty:
        rows = ", ".join(str(i + 2) for i in bad.index.tolist())
        raise ValueError(f"weights.csv has blank image_name or non-numeric weight_g at rows: {rows}")
    return df


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device_arg)


def set_num_classes(model: nn.Module, model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name == "fcn_resnet50":
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    elif model_name == "deeplabv3_resnet50":
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def build_model(model_name: str) -> nn.Module:
    if model_name == "fcn_resnet50":
        model = fcn_resnet50(weights=None, weights_backbone=None, num_classes=21)
    elif model_name == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=21)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return set_num_classes(model, model_name, num_classes=2)


def load_checkpoint(checkpoint_path: Path, fallback_model: str, fallback_image_size: int, device: torch.device) -> tuple[nn.Module, str, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    try:
        checkpoint: Any = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_name = str(checkpoint.get("model_name", fallback_model))
        image_size = int(checkpoint.get("image_size", fallback_image_size))
        state = checkpoint["model_state"]
    else:
        model_name = fallback_model
        image_size = int(fallback_image_size)
        state = checkpoint

    model = build_model(model_name)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, model_name, image_size


@torch.no_grad()
def predict_tomato_mask(model: nn.Module, image_bgr: np.ndarray, image_size: int, device: torch.device) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
    logits = model(tensor.to(device))["out"]
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
    return (pred > 0).astype(np.uint8)


def largest_component(mask01: np.ndarray) -> tuple[np.ndarray | None, float]:
    mask01 = (mask01 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 1:
        return None, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas) + 1)
    component = (labels == best_label).astype(np.uint8)
    return component, float(stats[best_label, cv2.CC_STAT_AREA])


def extract_component_features(mask01: np.ndarray) -> dict[str, Any]:
    component, component_area_px = largest_component(mask01)
    if component is None or component_area_px <= 0:
        return {"status": "failed", "failure_reason": "no_tomato_component", "component_mask": np.zeros_like(mask01)}

    contours, _ = cv2.findContours(component * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "failed", "failure_reason": "no_component_contour", "component_mask": component}

    contour = max(contours, key=cv2.contourArea)
    contour_area_px = float(cv2.contourArea(contour))
    equivalent_diameter_px = float(math.sqrt(4.0 * component_area_px / math.pi))
    ellipse = None
    ellipse_area_px = float("nan")
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
    if long_axis_px < short_axis_px:
        long_axis_px, short_axis_px = short_axis_px, long_axis_px
    return {
        "status": "ok",
        "failure_reason": "",
        "component_mask": component,
        "contour": contour,
        "ellipse": ellipse,
        "axis_source": axis_source,
        "area_px": component_area_px,
        "contour_area_px": contour_area_px,
        "bbox_x": int(bbox_x),
        "bbox_y": int(bbox_y),
        "bbox_w": int(bbox_w),
        "bbox_h": int(bbox_h),
        "long_axis_px": long_axis_px,
        "short_axis_px": short_axis_px,
        "equivalent_diameter_px": equivalent_diameter_px,
        "ellipse_area_px": ellipse_area_px,
        "area_to_ellipse_ratio": float(component_area_px / ellipse_area_px) if ellipse_area_px else float("nan"),
    }


def compute_v3(features: dict[str, Any], cm_per_px: float) -> dict[str, float]:
    l_cm = float(features["long_axis_px"]) * cm_per_px
    w_cm = float(features["short_axis_px"]) * cm_per_px
    volume_cm3 = math.pi / 6.0 * l_cm * w_cm * w_cm
    return {
        "A_cm2": float(features["area_px"]) * cm_per_px**2,
        "L_cm": l_cm,
        "W_cm": w_cm,
        "D_eq_cm": float(features["equivalent_diameter_px"]) * cm_per_px,
        "volume_cm3": volume_cm3,
    }


def click_calibrate_scale(image_bgr: np.ndarray, cube_edge_cm: float, image_name: str) -> dict[str, Any]:
    max_display_w = 1400
    max_display_h = 900
    height, width = image_bgr.shape[:2]
    display_scale = min(1.0, max_display_w / width, max_display_h / height)
    display = cv2.resize(
        image_bgr,
        (int(round(width * display_scale)), int(round(height * display_scale))),
        interpolation=cv2.INTER_AREA,
    )
    shown = display.copy()
    points: list[tuple[int, int]] = []
    window_name = f"{image_name}: click two endpoints of the white 5cm cube edge, then press ENTER"

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
            raise RuntimeError(f"Calibration cancelled for {image_name}. Please rerun and click two endpoints.")
    cv2.destroyWindow(window_name)

    (x1, y1), (x2, y2) = points
    full_x1 = x1 / display_scale
    full_y1 = y1 / display_scale
    full_x2 = x2 / display_scale
    full_y2 = y2 / display_scale
    edge_px = math.hypot(full_x2 - full_x1, full_y2 - full_y1)
    if edge_px <= 0:
        raise ValueError(f"Clicked calibration edge has zero length for {image_name}.")
    return {
        "image_name": image_name,
        "x1": full_x1,
        "y1": full_y1,
        "x2": full_x2,
        "y2": full_y2,
        "edge_px": edge_px,
        "cube_edge_cm": cube_edge_cm,
        "cm_per_px": cube_edge_cm / edge_px,
    }


def load_calibration_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration points: {path}")
    return pd.read_csv(path)


def prepare_calibrations(
    images: list[tuple[str, Path]],
    mode: str,
    cube_edge_cm: float,
    output_dir: Path,
    reuse: bool,
) -> pd.DataFrame:
    points_path = output_dir / "calibration_points.csv"
    if reuse:
        points_df = load_calibration_points(points_path)
        if mode == "group_once":
            if points_df.empty:
                raise ValueError("calibration_points.csv is empty.")
            return points_df.head(1).copy()
        missing = sorted(set(name for name, _ in images) - set(points_df["image_name"].astype(str)))
        if missing:
            raise ValueError(f"Missing per-image calibration points for: {missing}")
        return points_df.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if mode == "group_once":
        image_name, image_path = images[0]
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        row = click_calibrate_scale(image, cube_edge_cm, image_name)
        row["calibration_mode"] = "group_once"
        rows.append(row)
    else:
        for image_name, image_path in images:
            image = imread_unicode(image_path, cv2.IMREAD_COLOR)
            row = click_calibrate_scale(image, cube_edge_cm, image_name)
            row["calibration_mode"] = "per_image"
            rows.append(row)

    points_df = pd.DataFrame(rows)
    points_df.to_csv(points_path, index=False, encoding="utf-8-sig")
    return points_df


def calibration_for_image(points_df: pd.DataFrame, image_name: str, mode: str) -> dict[str, Any]:
    if mode == "group_once":
        row = points_df.iloc[0].to_dict()
        row["applied_from_image_name"] = row.get("image_name", "")
        return row
    matches = points_df[points_df["image_name"].astype(str) == image_name]
    if matches.empty:
        raise ValueError(f"No calibration points for {image_name}")
    return matches.iloc[0].to_dict()


def make_overlay(image_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[:, :, 1] = 255
    blended = cv2.addWeighted(image_bgr, 0.72, color, 0.28, 0)
    overlay[mask01 > 0] = blended[mask01 > 0]
    return overlay


def fit_panel(image: np.ndarray, width: int, height: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=interpolation)
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def draw_label(panel: np.ndarray, lines: list[str]) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 78), (255, 255, 255), -1)
    for i, line in enumerate(lines[:4]):
        cv2.putText(panel, line, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


def make_debug_image(image_bgr: np.ndarray, pred_mask: np.ndarray, features: dict[str, Any], row: dict[str, Any], calibration: dict[str, Any]) -> np.ndarray:
    component = features.get("component_mask", np.zeros(image_bgr.shape[:2], dtype=np.uint8)).astype(np.uint8)
    overlay = make_overlay(image_bgr, component)
    contour_view = overlay.copy()
    if features.get("contour") is not None:
        cv2.drawContours(contour_view, [features["contour"]], -1, (0, 255, 0), 3)
    if features.get("ellipse") is not None:
        cv2.ellipse(contour_view, features["ellipse"], (255, 0, 255), 2)
    if row.get("bbox_w") and row.get("bbox_h"):
        x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        cv2.rectangle(contour_view, (x, y), (x + w, y + h), (0, 200, 255), 2)
    if row["image_name"] == calibration.get("image_name") or row.get("calibration_mode") == "per_image":
        p1 = (int(round(float(calibration["x1"]))), int(round(float(calibration["y1"]))))
        p2 = (int(round(float(calibration["x2"]))), int(round(float(calibration["y2"]))))
        cv2.line(contour_view, p1, p2, (0, 255, 255), 3)
        cv2.circle(contour_view, p1, 6, (0, 0, 255), -1)
        cv2.circle(contour_view, p2, 6, (0, 0, 255), -1)

    mask_bgr = cv2.cvtColor(pred_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    component_bgr = cv2.cvtColor(component * 255, cv2.COLOR_GRAY2BGR)
    panels = [
        fit_panel(image_bgr, 360, 270),
        fit_panel(mask_bgr, 360, 270, cv2.INTER_NEAREST),
        fit_panel(component_bgr, 360, 270, cv2.INTER_NEAREST),
        fit_panel(contour_view, 360, 270),
    ]
    labels = [
        ["input", row["image_name"]],
        ["pred mask", f"mask px={int(pred_mask.sum())}"],
        ["largest component", f"area={row.get('area_px', 0):.1f}"],
        ["debug", f"weight={row['weight_g']:.3f}g"],
    ]
    for panel, label in zip(panels, labels):
        draw_label(panel, label)

    text_panel = np.full((270, 720, 3), 255, dtype=np.uint8)
    lines = [
        f"mode: {row['calibration_mode']}  cm_per_px: {row['cm_per_px']:.8f}",
        f"L={row.get('L_cm', 0):.4f} cm  W={row.get('W_cm', 0):.4f} cm  V={row.get('volume_cm3', 0):.4f} cm3",
        f"weight={row['weight_g']:.4f} g",
        f"edge_px={row.get('edge_px', float('nan')):.3f}  cube_edge_cm={row.get('cube_edge_cm', float('nan')):.2f}",
    ]
    if row["calibration_mode"] == "group_once" and row["image_name"] != calibration.get("image_name"):
        lines.append(f"calibration reused from: {calibration.get('image_name')}")
    for i, line in enumerate(lines):
        cv2.putText(text_panel, line, (16, 36 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)

    return np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:]), text_panel])


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({key: json_safe(value) for key, value in data.items()}, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate tomato density rho from known-weight images.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--weights-csv", default=str(DEFAULT_WEIGHTS_CSV))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", choices=["fcn_resnet50", "deeplabv3_resnet50"], default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--cube-edge-cm", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--reuse-calibration-points", action="store_true")
    parser.add_argument("--calibration-mode", choices=["group_once", "per_image"], default="group_once")
    parser.add_argument("--save-updated-predictor-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    model_name = args.model or str(config.get("model", FALLBACK_CONFIG["model"]))
    image_size = int(args.image_size or config.get("image_size", FALLBACK_CONFIG["image_size"]))
    cube_edge_cm = float(args.cube_edge_cm or config.get("cube_edge_cm", FALLBACK_CONFIG["cube_edge_cm"]))
    volume_formula = str(config.get("volume_formula", FALLBACK_CONFIG["volume_formula"]))
    density_method = str(config.get("density_method", FALLBACK_CONFIG["density_method"]))

    images_dir = resolve_path(args.images_dir)
    weights_csv = resolve_path(args.weights_csv)
    checkpoint = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    weights_df = read_weights(weights_csv)
    images: list[tuple[str, Path]] = []
    for rec in weights_df.itertuples(index=False):
        image_name = str(rec.image_name)
        image_path = images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in weights.csv was not found: {image_path}")
        images.append((image_name, image_path))

    calibrations = prepare_calibrations(images, args.calibration_mode, cube_edge_cm, output_dir, args.reuse_calibration_points)
    calibrations.to_csv(output_dir / "calibration_points.csv", index=False, encoding="utf-8-sig")

    device = choose_device(args.device)
    model, loaded_model, loaded_size = load_checkpoint(checkpoint, model_name, image_size, device)
    rows: list[dict[str, Any]] = []
    rho_values: list[float] = []
    for i, rec in enumerate(weights_df.itertuples(index=False), start=1):
        image_name = str(rec.image_name)
        image_path = images_dir / image_name
        weight_g = float(rec.weight_g)
        calibration = calibration_for_image(calibrations, image_name, args.calibration_mode)
        cm_per_px = float(calibration["cm_per_px"])
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        pred_mask = predict_tomato_mask(model, image, loaded_size, device)
        features = extract_component_features(pred_mask)
        if features["status"] != "ok":
            raise RuntimeError(f"Failed to segment tomato in {image_name}: {features.get('failure_reason')}")
        v3 = compute_v3(features, cm_per_px)
        rho_i = weight_g / v3["volume_cm3"]
        rho_values.append(rho_i)
        row = {
            "image_name": image_name,
            "image_path": str(image_path),
            "weight_g": weight_g,
            "cm_per_px": cm_per_px,
            "edge_px": float(calibration["edge_px"]),
            "cube_edge_cm": float(calibration["cube_edge_cm"]),
            "calibration_mode": args.calibration_mode,
            "calibration_source_image": str(calibration.get("image_name", "")),
            "area_px": features["area_px"],
            "contour_area_px": features["contour_area_px"],
            "long_axis_px": features["long_axis_px"],
            "short_axis_px": features["short_axis_px"],
            "L_cm": v3["L_cm"],
            "W_cm": v3["W_cm"],
            "D_eq_cm": v3["D_eq_cm"],
            "volume_cm3": v3["volume_cm3"],
            # rho_i is used internally to compute rho_mean, but is not written to outputs.
            "bbox_x": features["bbox_x"],
            "bbox_y": features["bbox_y"],
            "bbox_w": features["bbox_w"],
            "bbox_h": features["bbox_h"],
            "area_to_ellipse_ratio": features["area_to_ellipse_ratio"],
            "model": loaded_model,
            "image_size": loaded_size,
            "volume_formula": volume_formula,
        }
        rows.append(row)
        debug = make_debug_image(image, pred_mask, features, row, calibration)
        imwrite_unicode(debug_dir / f"{Path(image_name).stem}_debug.jpg", debug)
        print(f"[{i}/{len(weights_df)}] {image_name}")
        print(f"weight: {weight_g:.4f} g")
        print(f"volume: {v3['volume_cm3']:.4f} cm^3")
        # Only rho_mean is reported as the final density output.

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "density_results.csv", index=False, encoding="utf-8-sig")
    if not rho_values:
        raise RuntimeError("No valid rho values were computed.")
    summary = {
        "n_samples": int(len(results_df)),
        "rho_mean": float(np.mean(rho_values)),
        "volume_formula": volume_formula,
        "density_method": density_method,
        "calibration_mode": args.calibration_mode,
        "model": loaded_model,
        "image_size": loaded_size,
        "cube_edge_cm": cube_edge_cm,
    }
    write_json(output_dir / "density_summary.json", summary)
    pd.DataFrame([summary]).to_csv(output_dir / "density_summary.csv", index=False, encoding="utf-8-sig")

    updated_config_path = ""
    if args.save_updated_predictor_config:
        updated_config = {
            "model": loaded_model,
            "image_size": loaded_size,
            "rho": summary["rho_mean"],
            "volume_formula": volume_formula,
            "density_method": density_method,
        }
        updated_config_path = str(output_dir / "predictor_config_updated.json")
        write_json(output_dir / "predictor_config_updated.json", updated_config)

    print("Density calibration complete")
    print(f"Samples: {summary['n_samples']}")
    print(f"Calibration mode: {args.calibration_mode}")
    print(f"rho mean: {summary['rho_mean']:.6f}")
    print(f"Output dir: {output_dir}")
    print(f"Updated predictor config: {updated_config_path or 'not requested'}")


if __name__ == "__main__":
    main()
