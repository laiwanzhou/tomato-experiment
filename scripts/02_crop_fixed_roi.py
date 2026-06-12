from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PAIRS_CSV = OUTPUT_DIR / "pairs.csv"
ROI_CONFIG_JSON = OUTPUT_DIR / "roi_config.json"
SIZE_CHECK_CSV = OUTPUT_DIR / "image_size_check.csv"
ROI_OUTPUT_DIR = OUTPUT_DIR / "roi_tomato_images"
REVIEW_OUTPUT_DIR = OUTPUT_DIR / "review_pairs"
PADDING = 100


def imread_unicode(path: Path) -> np.ndarray:
    """Read an image from a Windows path that may contain non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 92) -> None:
    """Write a JPEG/PNG image to a Windows path that may contain non-ASCII characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"Could not encode image for: {path}")
    encoded.tofile(str(path))


def fit_image(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Resize image to fit inside a box while preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_text_lines(
    canvas: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    font_scale: float = 0.55,
    color: tuple[int, int, int] = (20, 20, 20),
) -> None:
    """Draw short labels onto a review image."""
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )


def select_roi_on_first_image(first_image: np.ndarray) -> dict[str, int]:
    """Open a scaled interactive window, then map the selected ROI back to full resolution."""
    image_h, image_w = first_image.shape[:2]
    max_display_w = 1400
    max_display_h = 900
    display_scale = min(1.0, max_display_w / image_w, max_display_h / image_h)
    display = cv2.resize(
        first_image,
        (int(round(image_w * display_scale)), int(round(image_h * display_scale))),
        interpolation=cv2.INTER_AREA,
    )

    window_name = "Select ROI: target tomato + white cube, then press ENTER/SPACE"
    roi = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise RuntimeError("No ROI selected. Re-run the script and select a rectangle.")

    raw_x1 = int(round(x / display_scale))
    raw_y1 = int(round(y / display_scale))
    raw_x2 = int(round((x + w) / display_scale))
    raw_y2 = int(round((y + h) / display_scale))

    roi_x1 = max(0, raw_x1 - PADDING)
    roi_y1 = max(0, raw_y1 - PADDING)
    roi_x2 = min(image_w, raw_x2 + PADDING)
    roi_y2 = min(image_h, raw_y2 + PADDING)

    return {
        "image_width": image_w,
        "image_height": image_h,
        "raw_roi_x1": raw_x1,
        "raw_roi_y1": raw_y1,
        "raw_roi_x2": raw_x2,
        "raw_roi_y2": raw_y2,
        "roi_x1": roi_x1,
        "roi_y1": roi_y1,
        "roi_x2": roi_x2,
        "roi_y2": roi_y2,
        "padding": PADDING,
    }


def load_or_select_roi(first_image: np.ndarray, reset_roi: bool) -> dict[str, int]:
    """Use saved ROI config unless reset is requested."""
    if ROI_CONFIG_JSON.exists() and not reset_roi:
        with ROI_CONFIG_JSON.open("r", encoding="utf-8") as f:
            config = json.load(f)
        print("Using existing ROI config: outputs/roi_config.json")
        return config

    config = select_roi_on_first_image(first_image)
    with ROI_CONFIG_JSON.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("Wrote ROI config: outputs/roi_config.json")
    return config


def check_image_sizes(pairs_df: pd.DataFrame, first_w: int, first_h: int) -> None:
    """Record tomato image dimensions and warn if any differ from the first image."""
    rows = []
    mismatches = []

    for row in pairs_df.itertuples(index=False):
        tomato_path = Path(row.tomato_image)
        image = imread_unicode(tomato_path)
        h, w = image.shape[:2]
        matches_first = (w == first_w and h == first_h)
        rows.append(
            {
                "idx": int(row.idx),
                "tomato_image": str(tomato_path),
                "image_width": w,
                "image_height": h,
                "matches_first": matches_first,
            }
        )
        if not matches_first:
            mismatches.append((int(row.idx), tomato_path.name, w, h))

    pd.DataFrame(rows).to_csv(SIZE_CHECK_CSV, index=False, encoding="utf-8-sig")

    if mismatches:
        print("WARNING: Some tomato images do not match the first image resolution:")
        for idx, name, w, h in mismatches:
            print(f"  idx {idx:03d}: {name} -> {w}x{h}, expected {first_w}x{first_h}")
        print("Details written to: outputs/image_size_check.csv")
    else:
        print("All tomato image sizes match the first image. Wrote: outputs/image_size_check.csv")


def make_review_image(
    idx: int,
    roi_image: np.ndarray,
    weight_image: np.ndarray,
    tomato_name: str,
    weight_name: str,
    output_path: Path,
) -> None:
    """Create one side-by-side review image for a single pair."""
    thumb_w = 460
    thumb_h = 340
    label_h = 80
    margin = 18
    gap = 24
    canvas_w = margin * 2 + thumb_w * 2 + gap
    canvas_h = margin * 2 + label_h + thumb_h
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    left = fit_image(roi_image, thumb_w, thumb_h)
    right = fit_image(weight_image, thumb_w, thumb_h)
    left_x = margin
    right_x = margin + thumb_w + gap
    image_y = margin + label_h

    draw_text_lines(canvas, [f"idx: {idx:03d}", f"tomato: {tomato_name}", f"weight: {weight_name}"], margin, 30)
    canvas[image_y : image_y + left.shape[0], left_x : left_x + left.shape[1]] = left
    canvas[image_y : image_y + right.shape[0], right_x : right_x + right.shape[1]] = right
    cv2.rectangle(canvas, (left_x, image_y), (left_x + thumb_w, image_y + thumb_h), (210, 210, 210), 1)
    cv2.rectangle(canvas, (right_x, image_y), (right_x + thumb_w, image_y + thumb_h), (210, 210, 210), 1)

    imwrite_unicode(output_path, canvas)


def make_contact_sheet(review_items: list[dict[str, object]], output_path: Path) -> None:
    """Create a total review contact sheet from ROI and weight images."""
    thumb_w = 340
    thumb_h = 250
    label_h = 78
    margin = 18
    gap = 22
    row_h = thumb_h + label_h + margin
    sheet_w = margin * 2 + thumb_w * 2 + gap
    sheet_h = max(row_h, row_h * len(review_items))
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    for item in review_items:
        idx = int(item["idx"])
        row_y = (idx - 1) * row_h
        roi_image = fit_image(item["roi_image"], thumb_w, thumb_h)  # type: ignore[arg-type]
        weight_image = fit_image(item["weight_image"], thumb_w, thumb_h)  # type: ignore[arg-type]
        left_x = margin
        right_x = margin + thumb_w + gap
        image_y = row_y + label_h

        draw_text_lines(
            sheet,
            [
                f"idx: {idx:03d}",
                f"tomato: {item['tomato_name']}",
                f"weight: {item['weight_name']}",
            ],
            margin,
            row_y + 24,
        )
        sheet[image_y : image_y + roi_image.shape[0], left_x : left_x + roi_image.shape[1]] = roi_image
        sheet[image_y : image_y + weight_image.shape[0], right_x : right_x + weight_image.shape[1]] = weight_image
        cv2.rectangle(sheet, (left_x, image_y), (left_x + thumb_w, image_y + thumb_h), (210, 210, 210), 1)
        cv2.rectangle(sheet, (right_x, image_y), (right_x + thumb_w, image_y + thumb_h), (210, 210, 210), 1)
        cv2.line(sheet, (0, row_y + row_h - 1), (sheet_w, row_y + row_h - 1), (230, 230, 230), 1)

    imwrite_unicode(output_path, sheet)


def crop_all_pairs(pairs_df: pd.DataFrame, roi_config: dict[str, int]) -> None:
    """Apply the fixed ROI to every tomato image and write review outputs."""
    ROI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x1 = int(roi_config["roi_x1"])
    y1 = int(roi_config["roi_y1"])
    x2 = int(roi_config["roi_x2"])
    y2 = int(roi_config["roi_y2"])
    review_items: list[dict[str, object]] = []

    for row in pairs_df.itertuples(index=False):
        idx = int(row.idx)
        tomato_path = Path(row.tomato_image)
        weight_path = Path(row.weight_image)

        tomato_image = imread_unicode(tomato_path)
        weight_image = imread_unicode(weight_path)
        h, w = tomato_image.shape[:2]
        crop_x1 = max(0, min(x1, w))
        crop_y1 = max(0, min(y1, h))
        crop_x2 = max(0, min(x2, w))
        crop_y2 = max(0, min(y2, h))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError(f"Invalid ROI for idx {idx:03d}: {tomato_path}")

        roi_image = tomato_image[crop_y1:crop_y2, crop_x1:crop_x2]
        roi_output_path = ROI_OUTPUT_DIR / f"idx_{idx:03d}_roi.jpg"
        review_output_path = REVIEW_OUTPUT_DIR / f"idx_{idx:03d}_review.jpg"

        imwrite_unicode(roi_output_path, roi_image)
        make_review_image(idx, roi_image, weight_image, tomato_path.name, weight_path.name, review_output_path)

        review_items.append(
            {
                "idx": idx,
                "roi_image": roi_image,
                "weight_image": weight_image,
                "tomato_name": tomato_path.name,
                "weight_name": weight_path.name,
            }
        )

    make_contact_sheet(review_items, OUTPUT_DIR / "review_contact_sheet.jpg")
    print("Wrote ROI crops to: outputs/roi_tomato_images")
    print("Wrote review pairs to: outputs/review_pairs")
    print("Wrote: outputs/review_contact_sheet.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop a fixed ROI from all tomato images.")
    parser.add_argument(
        "--reset-roi",
        action="store_true",
        help="Ignore existing outputs/roi_config.json and select a new fixed ROI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not PAIRS_CSV.exists():
        raise FileNotFoundError(f"Missing {PAIRS_CSV}. Run scripts/01_make_pairs.py first.")

    pairs_df = pd.read_csv(PAIRS_CSV)
    required_columns = {"idx", "tomato_image", "weight_image"}
    missing = required_columns - set(pairs_df.columns)
    if missing:
        raise ValueError(f"{PAIRS_CSV} is missing required columns: {sorted(missing)}")

    first_image = imread_unicode(Path(pairs_df.iloc[0]["tomato_image"]))
    first_h, first_w = first_image.shape[:2]

    roi_config = load_or_select_roi(first_image, reset_roi=args.reset_roi)
    check_image_sizes(pairs_df, first_w, first_h)
    crop_all_pairs(pairs_df, roi_config)


if __name__ == "__main__":
    main()
