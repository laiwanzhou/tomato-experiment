from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "番茄数据"
TOMATO_DIR = DATA_ROOT / "番茄图像数据"
WEIGHT_DIR = DATA_ROOT / "番茄实际重量"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> list[Path]:
    """Return image files in natural filename order."""
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return natsorted(images, key=lambda p: p.name)


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
    """Draw short ASCII-friendly labels onto the contact sheet."""
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


def make_pair_contact_sheet(pairs: list[tuple[int, Path, Path]], output_path: Path) -> None:
    """Create a vertical contact sheet with tomato image and weight image side by side."""
    thumb_w = 360
    thumb_h = 260
    label_h = 70
    margin = 18
    gap = 24
    row_h = thumb_h + label_h + margin
    sheet_w = margin * 2 + thumb_w * 2 + gap
    sheet_h = max(row_h, row_h * len(pairs))
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    for idx, tomato_path, weight_path in pairs:
        row_y = (idx - 1) * row_h
        tomato = fit_image(imread_unicode(tomato_path), thumb_w, thumb_h)
        weight = fit_image(imread_unicode(weight_path), thumb_w, thumb_h)

        left_x = margin
        right_x = margin + thumb_w + gap
        image_y = row_y + label_h

        draw_text_lines(
            sheet,
            [f"idx: {idx:03d}", f"tomato: {tomato_path.name}", f"weight: {weight_path.name}"],
            margin,
            row_y + 24,
        )
        sheet[image_y : image_y + tomato.shape[0], left_x : left_x + tomato.shape[1]] = tomato
        sheet[image_y : image_y + weight.shape[0], right_x : right_x + weight.shape[1]] = weight

        cv2.rectangle(sheet, (left_x, image_y), (left_x + thumb_w, image_y + thumb_h), (210, 210, 210), 1)
        cv2.rectangle(sheet, (right_x, image_y), (right_x + thumb_w, image_y + thumb_h), (210, 210, 210), 1)
        cv2.line(sheet, (0, row_y + row_h - 1), (sheet_w, row_y + row_h - 1), (230, 230, 230), 1)

    imwrite_unicode(output_path, sheet)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tomato_images = list_images(TOMATO_DIR)
    weight_images = list_images(WEIGHT_DIR)

    if len(tomato_images) != len(weight_images):
        raise ValueError(
            f"Image count mismatch: tomato={len(tomato_images)}, weight={len(weight_images)}. "
            "Please check the source folders before pairing."
        )

    if not tomato_images:
        raise ValueError("No images found in the data folders.")

    pairs = list(zip(range(1, len(tomato_images) + 1), tomato_images, weight_images))
    rows = [
        {
            "idx": idx,
            "tomato_image": str(tomato_path),
            "weight_image": str(weight_path),
        }
        for idx, tomato_path, weight_path in pairs
    ]

    pairs_df = pd.DataFrame(rows, columns=["idx", "tomato_image", "weight_image"])
    pairs_df.to_csv(OUTPUT_DIR / "pairs.csv", index=False, encoding="utf-8-sig")

    weights_df = pairs_df.copy()
    weights_df["weight_g"] = ""
    weights_df.to_csv(OUTPUT_DIR / "weights_template.csv", index=False, encoding="utf-8-sig")

    make_pair_contact_sheet(pairs, OUTPUT_DIR / "pair_contact_sheet.jpg")

    print(f"Paired {len(pairs)} tomato images with {len(pairs)} weight images.")
    print("Wrote: outputs/pairs.csv")
    print("Wrote: outputs/weights_template.csv")
    print("Wrote: outputs/pair_contact_sheet.jpg")


if __name__ == "__main__":
    main()
