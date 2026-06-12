from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def click_calibrate_scale(image_bgr: np.ndarray, cube_edge_cm: float) -> tuple[float, dict[str, Any]]:
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
    window_name = "Click two endpoints of one 5cm cube edge, then press ENTER"

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
            raise RuntimeError("Scale calibration cancelled.")
    cv2.destroyWindow(window_name)

    (x1, y1), (x2, y2) = points
    full_x1 = x1 / display_scale
    full_y1 = y1 / display_scale
    full_x2 = x2 / display_scale
    full_y2 = y2 / display_scale
    edge_px = math.hypot(full_x2 - full_x1, full_y2 - full_y1)
    if edge_px <= 0:
        raise ValueError("Clicked edge has zero length.")
    return cube_edge_cm / edge_px, {
        "cube_x1": full_x1,
        "cube_y1": full_y1,
        "cube_x2": full_x2,
        "cube_y2": full_y2,
        "cube_edge_px": edge_px,
    }
