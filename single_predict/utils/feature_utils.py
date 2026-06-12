from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


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
        return {
            "status": "failed",
            "failure_reason": "no_tomato_component",
            "component_mask": np.zeros(mask01.shape[:2], dtype=np.uint8),
        }

    contours, _ = cv2.findContours(component * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "status": "failed",
            "failure_reason": "no_component_contour",
            "component_mask": component,
        }

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
        "area_to_ellipse_ratio": float(component_area_px / ellipse_area_px) if ellipse_area_px else float("nan"),
    }


def compute_v3_prediction(features: dict[str, Any], cm_per_px: float, rho: float) -> dict[str, float]:
    area_px = float(features["area_px"])
    long_axis_px = float(features["long_axis_px"])
    short_axis_px = float(features["short_axis_px"])
    equivalent_diameter_px = float(features["equivalent_diameter_px"])

    a_cm2 = area_px * cm_per_px**2
    l_cm = long_axis_px * cm_per_px
    w_cm = short_axis_px * cm_per_px
    d_eq_cm = equivalent_diameter_px * cm_per_px
    volume_cm3 = math.pi / 6.0 * l_cm * w_cm * w_cm
    pred_weight_g = rho * volume_cm3
    return {
        "A_cm2": a_cm2,
        "L_cm": l_cm,
        "W_cm": w_cm,
        "D_eq_cm": d_eq_cm,
        "volume_cm3": volume_cm3,
        "pred_weight_g": pred_weight_g,
    }
