from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50


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


def make_overlay(image_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[:, :, 1] = 255
    blended = cv2.addWeighted(image_bgr, 0.72, color, 0.28, 0)
    overlay[mask01 > 0] = blended[mask01 > 0]
    return overlay
