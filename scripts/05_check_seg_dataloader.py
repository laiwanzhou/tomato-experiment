from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_DATASET_DIR = PROJECT_ROOT / "outputs" / "seg_dataset"
PREVIEW_DIR = PROJECT_ROOT / "outputs" / "seg_training_preview"


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


class TomatoSegDataset(Dataset):
    """Binary tomato segmentation dataset backed by a manifest CSV."""

    def __init__(self, manifest_csv: Path, image_size: int = 512) -> None:
        self.manifest_csv = manifest_csv
        self.df = pd.read_csv(manifest_csv)
        self.image_size = int(image_size)
        required = {"image_path", "mask_path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{manifest_csv} missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image_bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = imread_unicode(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)

        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.int64)

        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous().float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).contiguous().long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "filename": image_path.name,
        }


def tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(image_rgb)
    color[:, :, 0] = 255
    overlay = image_rgb.copy()
    overlay[mask > 0] = cv2.addWeighted(image_rgb, 0.55, color, 0.45, 0)[mask > 0]
    return overlay


def draw_label(panel: np.ndarray, lines: list[str]) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 76), (255, 255, 255), -1)
    for i, line in enumerate(lines[:4]):
        cv2.putText(panel, line, (8, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def make_preview(loader: DataLoader, output_path: Path, max_samples: int = 8) -> None:
    samples: list[dict[str, Any]] = []
    for batch in loader:
        batch_size = batch["image"].shape[0]
        for i in range(batch_size):
            samples.append(
                {
                    "image": batch["image"][i],
                    "mask": batch["mask"][i],
                    "filename": batch["filename"][i],
                }
            )
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    if not samples:
        return

    panel_w, panel_h = 260, 260
    row_w = panel_w * 3
    canvas = np.full((panel_h * len(samples), row_w, 3), 255, dtype=np.uint8)
    for row_i, sample in enumerate(samples):
        image_rgb = tensor_to_rgb_uint8(sample["image"])
        mask = sample["mask"].detach().cpu().numpy().astype(np.uint8)
        mask_vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
        overlay = make_overlay(image_rgb, mask)

        panels = [image_rgb, mask_vis, overlay]
        labels = [
            [
                f"image: {sample['filename']}",
                f"shape={tuple(sample['image'].shape)}",
                "",
                "",
            ],
            [
                "mask",
                f"unique={torch.unique(sample['mask']).tolist()}",
                f"shape={tuple(sample['mask'].shape)}",
                "",
            ],
            ["overlay", "", "", ""],
        ]

        y = row_i * panel_h
        for col_i, panel in enumerate(panels):
            fitted = fit_panel(panel, panel_w, panel_h)
            panel_bgr = cv2.cvtColor(fitted, cv2.COLOR_RGB2BGR)
            draw_label(panel_bgr, labels[col_i])
            x = col_i * panel_w
            canvas[y : y + panel_h, x : x + panel_w] = panel_bgr

    imwrite_unicode(output_path, canvas)


def build_loader(manifest_csv: Path, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = TomatoSegDataset(manifest_csv, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def inspect_loader(name: str, loader: DataLoader) -> None:
    batch = next(iter(loader))
    images = batch["image"]
    masks = batch["mask"]
    print(f"{name} samples: {len(loader.dataset)}")
    print(f"{name} batch image shape: {tuple(images.shape)}")
    print(f"{name} batch mask shape: {tuple(masks.shape)}")
    print(f"{name} mask unique values: {torch.unique(masks).tolist()}")
    print(f"{name} image min/max: {float(images.min()):.4f}/{float(images.max()):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LaboroTomato binary segmentation Dataset/DataLoader.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    train_manifest = SEG_DATASET_DIR / "train_manifest.csv"
    test_manifest = SEG_DATASET_DIR / "test_manifest.csv"
    train_loader = build_loader(train_manifest, args.image_size, args.batch_size, args.num_workers, shuffle=True)
    test_loader = build_loader(test_manifest, args.image_size, args.batch_size, args.num_workers, shuffle=False)

    inspect_loader("train", train_loader)
    inspect_loader("test", test_loader)

    make_preview(train_loader, PREVIEW_DIR / "dataloader_train_preview.jpg", max_samples=args.preview_samples)
    make_preview(test_loader, PREVIEW_DIR / "dataloader_test_preview.jpg", max_samples=args.preview_samples)

    print("Wrote: outputs/seg_training_preview/dataloader_train_preview.jpg")
    print("Wrote: outputs/seg_training_preview/dataloader_test_preview.jpg")


if __name__ == "__main__":
    main()
