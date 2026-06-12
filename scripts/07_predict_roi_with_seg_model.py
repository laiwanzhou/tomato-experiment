from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = PROJECT_ROOT / "outputs" / "roi_tomato_images"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "seg_checkpoints" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "seg_roi_predictions"


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def parse_roi_idx(path: Path) -> int:
    parts = path.stem.split("_")
    return int(parts[1]) if len(parts) >= 2 and parts[0] == "idx" else 0


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


def load_model(checkpoint_path: Path, fallback_model: str, fallback_image_size: int, device: torch.device) -> tuple[nn.Module, int, str]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}. Train a model first.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", fallback_model)
    image_size = int(checkpoint.get("image_size", fallback_image_size))
    model = build_model(model_name)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, image_size, model_name


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(image_rgb)
    color[:, :, 0] = 255
    overlay = image_rgb.copy()
    overlay[mask > 0] = cv2.addWeighted(image_rgb, 0.55, color, 0.45, 0)[mask > 0]
    return overlay


def largest_component_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def draw_label(panel: np.ndarray, lines: list[str]) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 56), (255, 255, 255), -1)
    for i, line in enumerate(lines[:3]):
        cv2.putText(panel, line, (8, 19 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


@torch.no_grad()
def predict_mask(model: nn.Module, image_bgr: np.ndarray, image_size: int, device: torch.device) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
    logits = model(tensor.to(device))["out"]
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def make_sample_debug(idx: int, path: Path, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
    overlay = make_overlay(image_rgb, mask)
    contour_view = overlay.copy()
    contour = largest_component_contour(mask)
    if contour is not None:
        cv2.drawContours(contour_view, [contour], -1, (0, 255, 0), 3)

    panels = [image_rgb, mask_vis, overlay, contour_view]
    labels = [
        [f"idx={idx:03d}", path.name, "ROI"],
        ["pred mask", f"area={int(mask.sum())}", ""],
        ["overlay", "", ""],
        ["largest component", "", ""],
    ]
    panel_w, panel_h = 260, 240
    canvas = np.full((panel_h, panel_w * 4, 3), 255, dtype=np.uint8)
    for i, panel in enumerate(panels):
        fitted = fit_panel(panel, panel_w, panel_h)
        fitted_bgr = cv2.cvtColor(fitted, cv2.COLOR_RGB2BGR)
        draw_label(fitted_bgr, labels[i])
        x = i * panel_w
        canvas[:, x : x + panel_w] = fitted_bgr
    return canvas


def make_contact_sheet(debug_images: list[np.ndarray], output_path: Path) -> None:
    if not debug_images:
        return
    thumb_w, thumb_h = 520, 120
    cols = 2
    rows = int(np.ceil(len(debug_images) / cols))
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype=np.uint8)
    for i, image in enumerate(debug_images):
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        r = i // cols
        c = i % cols
        sheet[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = thumb
    imwrite_unicode(output_path, sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tomato masks for the 33 ROI images using a trained segmentation model.")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--model", choices=["fcn_resnet50", "deeplabv3_resnet50"], default="fcn_resnet50")
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size, model_name = load_model(Path(args.checkpoint), args.model, args.image_size, device)
    print(f"Loaded {model_name} checkpoint with image_size={image_size} on {device}")

    roi_paths = sorted(ROI_DIR.glob("idx_*_roi.*"), key=parse_roi_idx)
    debug_images: list[np.ndarray] = []
    for roi_path in roi_paths:
        idx = parse_roi_idx(roi_path)
        image_bgr = imread_unicode(roi_path, cv2.IMREAD_COLOR)
        mask = predict_mask(model, image_bgr, image_size, device)
        imwrite_unicode(OUTPUT_DIR / f"idx_{idx:03d}_pred_mask.png", mask.astype(np.uint8))
        debug = make_sample_debug(idx, roi_path, image_bgr, mask)
        imwrite_unicode(OUTPUT_DIR / f"idx_{idx:03d}_segmentation.jpg", debug)
        debug_images.append(debug)

    make_contact_sheet(debug_images, OUTPUT_DIR / "roi_segmentation_preview.jpg")
    print(f"Processed {len(roi_paths)} ROI images.")
    print("Wrote: outputs/seg_roi_predictions")
    print("Wrote: outputs/seg_roi_predictions/roi_segmentation_preview.jpg")


if __name__ == "__main__":
    main()
