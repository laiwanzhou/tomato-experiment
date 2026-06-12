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

from utils.feature_utils import compute_v3_prediction, extract_component_features
from utils.scale_utils import click_calibrate_scale
from utils.segmentation_utils import (
    choose_device,
    imread_unicode,
    imwrite_unicode,
    load_checkpoint,
    make_overlay,
    predict_tomato_mask,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PACKAGE_DIR / "models" / "best_model.pth"
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "predictor_config.json"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"

FALLBACK_CONFIG = {
    "rho": 1.0468,
    "volume_formula": "V3_ellipsoid_H_eq_W",
    "density_method": "mean",
    "model": "fcn_resnet50",
    "image_size": 512,
    "cube_edge_cm": 5.0,
}


def load_predictor_config(path: Path) -> dict[str, Any]:
    config = FALLBACK_CONFIG.copy()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            config.update({k: v for k, v in loaded.items() if v is not None})
        except Exception as exc:
            print(f"WARNING: Could not read predictor config, using fallback defaults. Reason: {exc}", file=sys.stderr)
    return config


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    package_relative = (PACKAGE_DIR / path).resolve()
    if package_relative.exists():
        return package_relative
    return (Path.cwd() / path).resolve()


def resolve_scale(args: argparse.Namespace, image_bgr: np.ndarray) -> tuple[float, str, dict[str, Any]]:
    if args.calibrate_click:
        cm_per_px, click_info = click_calibrate_scale(image_bgr, args.cube_edge_cm)
        return cm_per_px, "interactive: --calibrate-click", click_info

    raise ValueError(
        "当前版本必须通过点击标定物进行尺度标定，请使用 --calibrate-click，或在脚本底部设置 DIRECT_RUN_CALIBRATE_CLICK = True。"
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_result_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {key: json_safe(value) for key, value in result.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)


def write_result_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow({key: json_safe(value) for key, value in result.items()})


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
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 74), (255, 255, 255), -1)
    for i, line in enumerate(lines[:4]):
        cv2.putText(panel, line, (10, 20 + i * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (20, 20, 20), 1, cv2.LINE_AA)


def make_debug_image(
    image_bgr: np.ndarray,
    pred_mask01: np.ndarray,
    component_mask01: np.ndarray,
    overlay_bgr: np.ndarray,
    features: dict[str, Any],
    result: dict[str, Any],
) -> np.ndarray:
    contour_view = overlay_bgr.copy()
    if features.get("contour") is not None:
        cv2.drawContours(contour_view, [features["contour"]], -1, (0, 255, 0), 3)
    if features.get("ellipse") is not None:
        cv2.ellipse(contour_view, features["ellipse"], (255, 0, 255), 2)
    if result.get("bbox_w") and result.get("bbox_h"):
        x, y, w, h = int(result["bbox_x"]), int(result["bbox_y"]), int(result["bbox_w"]), int(result["bbox_h"])
        cv2.rectangle(contour_view, (x, y), (x + w, y + h), (0, 200, 255), 2)

    pred_mask_bgr = cv2.cvtColor(pred_mask01.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    component_bgr = cv2.cvtColor(component_mask01.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    panels = [
        fit_panel(image_bgr, 360, 270),
        fit_panel(pred_mask_bgr, 360, 270, cv2.INTER_NEAREST),
        fit_panel(component_bgr, 360, 270, cv2.INTER_NEAREST),
        fit_panel(contour_view, 360, 270),
    ]
    labels = [
        ["input image", str(result["image_path"])[:45]],
        ["pred mask x255", f"mask area px={int(pred_mask01.sum())}"],
        ["largest component", f"area px={result.get('area_px', 0):.1f}"],
        [
            "prediction",
            f"L={result.get('L_cm', 0):.3f} cm W={result.get('W_cm', 0):.3f} cm",
            f"V={result.get('volume_cm3', 0):.3f} cm3",
            f"mass={result.get('pred_weight_g', 0):.3f} g",
        ],
    ]
    for panel, label in zip(panels, labels):
        draw_label(panel, label)

    text_panel = np.full((270, 720, 3), 255, dtype=np.uint8)
    text_lines = [
        f"cm_per_px: {result['cm_per_px']:.8f}",
        f"rho: {result['rho']:.4f} g/cm3",
        f"formula: V3 = pi/6 * L * W^2",
        f"model: {result['model']} image_size={result['image_size']} device={result['device']}",
        f"area_to_ellipse_ratio: {result.get('area_to_ellipse_ratio', float('nan')):.4f}",
        f"bbox: x={result.get('bbox_x')} y={result.get('bbox_y')} w={result.get('bbox_w')} h={result.get('bbox_h')}",
    ]
    for i, line in enumerate(text_lines):
        cv2.putText(text_panel, line, (16, 34 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 1, cv2.LINE_AA)

    top = np.hstack(panels[:2])
    middle = np.hstack(panels[2:])
    return np.vstack([top, middle, text_panel])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the weight of one tomato image using a trained segmentation model.")
    parser.add_argument("--image", default=None, help="Path to one image containing one tomato and one 5cm calibration cube.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to best_model.pth.")
    parser.add_argument("--model", choices=["fcn_resnet50", "deeplabv3_resnet50"], default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None, help="Density in g/cm^3. Defaults to predictor_config.json.")
    parser.add_argument("--calibrate-click", action="store_true", help="Click two endpoints of a 5cm cube edge on the image.")
    parser.add_argument("--cube-edge-cm", type=float, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-mask-01", action="store_true", help="Keep pred_mask_01.png. It is also written by default for reproducibility.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def apply_direct_run_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Allow PyCharm/direct execution without command-line arguments."""
    if args.image:
        return args

    if not DIRECT_RUN_IMAGE:
        raise ValueError(
            "No image was provided. Pass --image on the command line or fill DIRECT_RUN_IMAGE near the bottom of this script."
        )

    args.image = DIRECT_RUN_IMAGE
    args.calibrate_click = bool(DIRECT_RUN_CALIBRATE_CLICK)
    args.cube_edge_cm = float(DIRECT_RUN_CUBE_EDGE_CM)
    return args


def main() -> None:
    args = apply_direct_run_defaults(parse_args())
    config = load_predictor_config(DEFAULT_CONFIG)
    model_arg = args.model or str(config.get("model", FALLBACK_CONFIG["model"]))
    image_size_arg = int(args.image_size or config.get("image_size", FALLBACK_CONFIG["image_size"]))
    rho = float(args.rho if args.rho is not None else config.get("rho", FALLBACK_CONFIG["rho"]))
    volume_formula = str(config.get("volume_formula", FALLBACK_CONFIG["volume_formula"]))
    density_method = str(config.get("density_method", FALLBACK_CONFIG["density_method"]))
    if args.cube_edge_cm is None:
        args.cube_edge_cm = float(config.get("cube_edge_cm", FALLBACK_CONFIG["cube_edge_cm"]))

    image_path = resolve_path(args.image)
    checkpoint_path = resolve_path(args.checkpoint)
    output_root = resolve_path(args.output_dir)
    sample_dir = output_root / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
    cm_per_px, scale_source, click_info = resolve_scale(args, image_bgr)
    device = choose_device(args.device)
    model, loaded_model_name, loaded_image_size = load_checkpoint(checkpoint_path, model_arg, image_size_arg, device)

    pred_mask01 = predict_tomato_mask(model, image_bgr, loaded_image_size, device)
    features = extract_component_features(pred_mask01)
    if features["status"] != "ok":
        empty = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        imwrite_unicode(sample_dir / "pred_mask_01.png", pred_mask01.astype(np.uint8))
        imwrite_unicode(sample_dir / "pred_mask_vis.png", pred_mask01.astype(np.uint8) * 255)
        imwrite_unicode(sample_dir / "tomato_component_mask_vis.png", empty)
        debug = make_debug_image(image_bgr, pred_mask01, empty, image_bgr.copy(), features, {"image_path": str(image_path), "cm_per_px": cm_per_px, "rho": rho, "model": loaded_model_name, "image_size": loaded_image_size, "device": str(device), "area_px": 0, "L_cm": 0, "W_cm": 0, "volume_cm3": 0, "pred_weight_g": 0})
        imwrite_unicode(sample_dir / "debug.jpg", debug)
        raise RuntimeError(f"No tomato component found: {features.get('failure_reason')}")

    prediction = compute_v3_prediction(features, cm_per_px, rho)
    component_mask01 = features["component_mask"].astype(np.uint8)
    overlay_bgr = make_overlay(image_bgr, component_mask01)

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "model": loaded_model_name,
        "image_size": loaded_image_size,
        "device": str(device),
        "scale_source": scale_source,
        "cm_per_px": cm_per_px,
        "cube_edge_cm": float(args.cube_edge_cm),
        "rho": rho,
        "volume_formula": volume_formula,
        "density_method": density_method,
        "area_px": features["area_px"],
        "contour_area_px": features["contour_area_px"],
        "long_axis_px": features["long_axis_px"],
        "short_axis_px": features["short_axis_px"],
        "L_cm": prediction["L_cm"],
        "W_cm": prediction["W_cm"],
        "D_eq_cm": prediction["D_eq_cm"],
        "volume_cm3": prediction["volume_cm3"],
        "pred_weight_g": prediction["pred_weight_g"],
        "bbox_x": features["bbox_x"],
        "bbox_y": features["bbox_y"],
        "bbox_w": features["bbox_w"],
        "bbox_h": features["bbox_h"],
        "ellipse_area_px": features["ellipse_area_px"],
        "area_to_ellipse_ratio": features["area_to_ellipse_ratio"],
        "axis_source": features["axis_source"],
    }
    result.update({f"calibration_{key}": value for key, value in click_info.items()})

    imwrite_unicode(sample_dir / "pred_mask_01.png", pred_mask01.astype(np.uint8))
    imwrite_unicode(sample_dir / "pred_mask_vis.png", pred_mask01.astype(np.uint8) * 255)
    imwrite_unicode(sample_dir / "tomato_component_mask_vis.png", component_mask01 * 255)
    imwrite_unicode(sample_dir / "overlay.jpg", overlay_bgr)
    debug = make_debug_image(image_bgr, pred_mask01, component_mask01, overlay_bgr, features, result)
    imwrite_unicode(sample_dir / "debug.jpg", debug)
    write_result_json(sample_dir / "result.json", result)
    write_result_csv(sample_dir / "result.csv", result)

    print(f"Input image: {image_path}")
    print(f"Scale source: {scale_source}")
    print(f"cm_per_px: {cm_per_px:.8f}")
    print(f"L_cm: {prediction['L_cm']:.4f}")
    print(f"W_cm: {prediction['W_cm']:.4f}")
    print("Volume formula: V3 = pi/6 * L * W^2")
    print(f"Volume: {prediction['volume_cm3']:.4f} cm^3")
    print(f"rho: {rho:.4f} g/cm^3")
    print(f"Predicted weight: {prediction['pred_weight_g']:.4f} g")
    print(f"Output dir: {sample_dir}")
    print("==============================")
    print(f"预测番茄重量：{prediction['pred_weight_g']:.4f} g")
    print("==============================")


if __name__ == "__main__":
    # =========================
    # USER CONFIG FOR DIRECT RUN
    # =========================
    # PyCharm 直接运行脚本且没有传入 --image 时，会自动使用这里的配置。
    # 相对路径优先按 single_predict 目录解析，因此 .\examples\sample.jpg
    # 指向 single_predict\examples\sample.jpg。
    DIRECT_RUN_IMAGE = r".\examples\sample.jpg"
    DIRECT_RUN_CALIBRATE_CLICK = True
    DIRECT_RUN_CUBE_EDGE_CM = 5.0

    main()
