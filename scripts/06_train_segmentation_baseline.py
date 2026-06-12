from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, FCN_ResNet50_Weights, deeplabv3_resnet50, fcn_resnet50
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_DATASET_DIR = PROJECT_ROOT / "outputs" / "seg_dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "seg_checkpoints"
PRED_PREVIEW_DIR = PROJECT_ROOT / "outputs" / "seg_predictions_preview"


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
    def __init__(self, manifest_csv: Path, image_size: int = 512) -> None:
        self.df = pd.read_csv(manifest_csv)
        self.image_size = int(image_size)

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

        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        mask_tensor = torch.from_numpy((mask > 0).astype(np.int64)).contiguous().long()
        return {"image": image_tensor, "mask": mask_tensor, "filename": image_path.name}


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


def build_model(model_name: str, pretrained: bool = False) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = FCN_ResNet50_Weights.DEFAULT if model_name == "fcn_resnet50" else DeepLabV3_ResNet50_Weights.DEFAULT
        except Exception as exc:
            print(f"WARNING: Could not prepare pretrained weights, using weights=None. Reason: {exc}")
            weights = None

    try:
        if model_name == "fcn_resnet50":
            model = fcn_resnet50(weights=weights, weights_backbone=None, num_classes=21)
        elif model_name == "deeplabv3_resnet50":
            model = deeplabv3_resnet50(weights=weights, weights_backbone=None, num_classes=21)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as exc:
        print(f"WARNING: Could not load requested weights, falling back to weights=None. Reason: {exc}")
        if model_name == "fcn_resnet50":
            model = fcn_resnet50(weights=None, weights_backbone=None, num_classes=21)
        else:
            model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=21)

    return set_num_classes(model, model_name, num_classes=2)


def build_loaders(image_size: int, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_ds = TomatoSegDataset(SEG_DATASET_DIR / "train_manifest.csv", image_size=image_size)
    test_ds = TomatoSegDataset(SEG_DATASET_DIR / "test_manifest.csv", image_size=image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    total = target.numel()
    correct = (pred == target).sum().item()
    pixel_acc = correct / total if total else 0.0

    ious = []
    for cls in [0, 1]:
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union else float("nan"))

    tomato_inter = ((pred == 1) & (target == 1)).sum().item()
    tomato_pred = (pred == 1).sum().item()
    tomato_true = (target == 1).sum().item()
    dice = (2 * tomato_inter) / (tomato_pred + tomato_true) if (tomato_pred + tomato_true) else 1.0
    mean_iou = float(np.nanmean(ious))
    return {"pixel_acc": pixel_acc, "tomato_iou": ious[1], "mean_iou": mean_iou, "dice": dice}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"pixel_acc": 0.0, "tomato_iou": 0.0, "mean_iou": 0.0, "dice": 0.0}
    count = 0
    for batch in tqdm(loader, desc="eval", unit="batch", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)["out"]
        pred = logits.argmax(dim=1)
        metrics = compute_metrics(pred, masks)
        for key in totals:
            totals[key] += metrics[key] * images.shape[0]
        count += images.shape[0]
    return {key: value / count for key, value in totals.items()}


def tensor_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(image_rgb)
    color[:, :, 0] = 255
    overlay = image_rgb.copy()
    overlay[mask > 0] = cv2.addWeighted(image_rgb, 0.55, color, 0.45, 0)[mask > 0]
    return overlay


def draw_label(panel: np.ndarray, lines: list[str]) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 58), (255, 255, 255), -1)
    for i, line in enumerate(lines[:3]):
        cv2.putText(panel, line, (8, 19 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


def resize_preview_panel(panel_rgb: np.ndarray, panel_w: int, panel_h: int, interpolation: int) -> np.ndarray:
    """Resize a preview panel to the fixed canvas slot size."""
    resized = cv2.resize(panel_rgb, (panel_w, panel_h), interpolation=interpolation)
    return resized


def ensure_canvas_panel_size(panel_bgr: np.ndarray, panel_w: int, panel_h: int, interpolation: int) -> np.ndarray:
    """Final guard so no preview panel can be written into a mismatched slot."""
    if panel_bgr.shape[:2] != (panel_h, panel_w):
        panel_bgr = cv2.resize(panel_bgr, (panel_w, panel_h), interpolation=interpolation)
    return panel_bgr


@torch.no_grad()
def write_prediction_preview(model: nn.Module, loader: DataLoader, device: torch.device, epoch: int, max_samples: int = 4) -> None:
    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"]
    logits = model(images)["out"].argmax(dim=1).detach().cpu()

    sample_n = min(max_samples, images.shape[0])
    panel_w, panel_h = 220, 220
    canvas = np.full((panel_h * sample_n, panel_w * 4, 3), 255, dtype=np.uint8)
    for i in range(sample_n):
        image_rgb = tensor_rgb(batch["image"][i])
        true_mask = masks[i].numpy().astype(np.uint8)
        pred_mask = logits[i].numpy().astype(np.uint8)
        metrics = compute_metrics(torch.from_numpy(pred_mask), torch.from_numpy(true_mask))
        panels = [
            (image_rgb, cv2.INTER_AREA),
            (cv2.cvtColor(true_mask * 255, cv2.COLOR_GRAY2RGB), cv2.INTER_NEAREST),
            (cv2.cvtColor(pred_mask * 255, cv2.COLOR_GRAY2RGB), cv2.INTER_NEAREST),
            (make_overlay(image_rgb, pred_mask), cv2.INTER_AREA),
        ]
        labels = [
            [batch["filename"][i], "image", ""],
            ["true mask", "", ""],
            ["pred mask", f"IoU={metrics['tomato_iou']:.3f}", ""],
            ["overlay", "", ""],
        ]
        y = i * panel_h
        for j, (panel, interpolation) in enumerate(panels):
            panel_resized = resize_preview_panel(panel, panel_w, panel_h, interpolation)
            panel_bgr = cv2.cvtColor(panel_resized, cv2.COLOR_RGB2BGR)
            draw_label(panel_bgr, labels[j])
            panel_bgr = ensure_canvas_panel_size(panel_bgr, panel_w, panel_h, interpolation)
            x = j * panel_w
            canvas[y : y + panel_h, x : x + panel_w] = panel_bgr

    imwrite_unicode(PRED_PREVIEW_DIR / f"epoch_{epoch:03d}_preview.jpg", canvas)


def append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "pixel_acc", "tomato_iou", "mean_iou", "dice", "lr", "checkpoint_path"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, model_name: str, image_size: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "image_size": image_size,
            "num_classes": 2,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a torchvision binary tomato segmentation baseline.")
    parser.add_argument("--model", choices=["fcn_resnet50", "deeplabv3_resnet50"], default="fcn_resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained", action="store_true", help="Try torchvision pretrained weights; falls back to weights=None if unavailable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = build_loaders(args.image_size, args.batch_size, args.num_workers)
    model = build_model(args.model, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1
    best_miou = -math.inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_miou = float(checkpoint.get("metrics", {}).get("mean_iou", -math.inf))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            train_iter = tqdm(train_loader, desc=f"epoch {epoch:03d} train", unit="batch")
            for batch in train_iter:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs["out"], masks)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu()) * images.shape[0]
                seen += images.shape[0]
                train_iter.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}")

            train_loss = running_loss / max(seen, 1)
            metrics = evaluate(model, test_loader, device)
            last_path = CHECKPOINT_DIR / "last_model.pth"
            save_checkpoint(last_path, model, optimizer, epoch, args.model, args.image_size, metrics)
            checkpoint_path = str(last_path)
            if metrics["mean_iou"] > best_miou:
                best_miou = metrics["mean_iou"]
                best_path = CHECKPOINT_DIR / "best_model.pth"
                save_checkpoint(best_path, model, optimizer, epoch, args.model, args.image_size, metrics)
                checkpoint_path = str(best_path)

            write_prediction_preview(model, test_loader, device, epoch)
            append_log(
                CHECKPOINT_DIR / "train_log.csv",
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "pixel_acc": metrics["pixel_acc"],
                    "tomato_iou": metrics["tomato_iou"],
                    "mean_iou": metrics["mean_iou"],
                    "dice": metrics["dice"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "checkpoint_path": checkpoint_path,
                },
            )
            print(
                f"epoch={epoch:03d} loss={train_loss:.4f} "
                f"pixel_acc={metrics['pixel_acc']:.4f} tomato_iou={metrics['tomato_iou']:.4f} "
                f"mean_iou={metrics['mean_iou']:.4f} dice={metrics['dice']:.4f}"
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("CUDA out of memory. Try reducing memory use with:")
            print("  --batch-size 1")
            print("or:")
            print("  --image-size 384")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        raise


if __name__ == "__main__":
    main()
