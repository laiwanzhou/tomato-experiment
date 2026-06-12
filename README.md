# 番茄图像估重实验项目

本项目第一阶段只做数据整理、图像配对、重量录入模板生成、固定 ROI 裁剪和检查图生成。

当前阶段不做番茄分割、体积估算、OCR 或重量预测。

## 目录结构

```text
D:\work\2026.6.11番茄
├─ 番茄数据
│  ├─ 番茄图像数据
│  └─ 番茄实际重量
├─ scripts
├─ outputs
│  ├─ roi_tomato_images
│  └─ review_pairs
└─ README.md
```

## 安装依赖

建议使用独立的 `tomato` conda 环境，不修改 `base` 环境：

```powershell
conda create -y -n tomato python=3.10
conda activate tomato
pip install opencv-python numpy pandas matplotlib openpyxl natsort scikit-learn
```

如果已经创建并安装过依赖，后续只需要：

```powershell
conda activate tomato
```

项目依赖为：

```powershell
pip install opencv-python numpy pandas matplotlib openpyxl natsort scikit-learn
```

## 1. 生成图片配对和重量录入模板

```powershell
python .\scripts\01_make_pairs.py
```

脚本会自然排序读取：

```text
.\番茄数据\番茄图像数据
.\番茄数据\番茄实际重量
```

然后按排序后一一配对，生成：

```text
outputs\pairs.csv
outputs\weights_template.csv
outputs\pair_contact_sheet.jpg
```

请先检查：

```text
outputs\pair_contact_sheet.jpg
```

确认番茄图像和电子秤照片配对正确。

`weights_template.csv` 中的 `weight_g` 为空，后续人工读取电子秤照片后填写。

## 2. 固定 ROI 裁剪

第一次运行：

```powershell
python .\scripts\02_crop_fixed_roi.py
```

脚本会打开第一张番茄图的交互窗口。请用鼠标框选右侧平台上的“目标番茄 + 白色立方体”，然后按 `Enter` 或 `Space` 确认。

脚本会自动向四周扩充 100 像素，并保存固定 ROI 配置：

```text
outputs\roi_config.json
```

之后再次运行时，脚本默认直接读取 `outputs\roi_config.json`，不会重复框选。

如果需要重新框选 ROI：

```powershell
python .\scripts\02_crop_fixed_roi.py --reset-roi
```

## 3. 检查 ROI 裁剪结果

固定 ROI 脚本会生成：

```text
outputs\image_size_check.csv
outputs\roi_tomato_images\idx_001_roi.jpg
outputs\roi_tomato_images\idx_002_roi.jpg
...
outputs\review_pairs\idx_001_review.jpg
outputs\review_pairs\idx_002_review.jpg
...
outputs\review_contact_sheet.jpg
```

请重点检查：

```text
outputs\review_contact_sheet.jpg
outputs\review_pairs
```

确认每张裁剪图都包含右侧平台上的目标番茄和 5cm 白色立方体，并且对应的电子秤照片正确。

## 注意事项

- 脚本不会修改原始图片。
- 图片按文件名自然排序，避免 `1.jpg、10.jpg、2.jpg` 的排序错误。
- 如果某些番茄图分辨率和第一张不同，脚本会在终端提示，并记录到 `outputs\image_size_check.csv`。
- 第一阶段只准备人工核对和后续实验所需的数据，不进行自动重量读取或重量预测。

## 第二阶段：分割、尺度标定、体积和估重验证

第二阶段只使用第一阶段生成的固定 ROI 图像：

```text
outputs\roi_tomato_images
outputs\weights.csv
```

不会修改原始图片，也不会使用 OCR。

请先确认 `outputs\weights.csv` 已经填写完整的 `weight_g`，并且 `outputs\roi_tomato_images` 中有 33 张 ROI 图。

### 运行第二阶段脚本

```powershell
conda activate tomato
python .\scripts\03_run_volume_experiment.py
```

默认等价于严格红色分割：

```powershell
python .\scripts\03_run_volume_experiment.py --mask-mode red
```

如果偏黄色或橙色番茄的 red mask 覆盖不完整，可以运行 warm-color 分割版本：

```powershell
python .\scripts\03_run_volume_experiment.py --mask-mode warm
```

第一次运行时，脚本会显示第一张 ROI 图像。请用鼠标点击白色 5cm 立方体正面上一条清晰 5cm 边的两个端点，优先选正面水平边或竖直边，不要选侧面透视边。点击两个点后按 `Enter` 或 `Space` 确认。

尺度配置会保存到：

```text
outputs\scale_config.json
outputs\debug_scale\scale_reference.jpg
```

以后再次运行会默认读取 `outputs\scale_config.json`，不会重复点击。

如需重新标定尺度：

```powershell
python .\scripts\03_run_volume_experiment.py --reset-scale
```

### 第二阶段输出

```text
outputs\debug_masks_red
outputs\debug_masks_contact_sheet_red.jpg
outputs\debug_masks_warm
outputs\debug_masks_contact_sheet_warm.jpg
outputs\debug_scale\scale_reference.jpg
outputs\volume_results_red\volume_features.csv
outputs\volume_results_red\predictions.csv
outputs\volume_results_red\experiment_results.csv
outputs\volume_results_red\cv_results.csv
outputs\volume_results_warm\volume_features.csv
outputs\volume_results_warm\predictions.csv
outputs\volume_results_warm\experiment_results.csv
outputs\volume_results_warm\cv_results.csv
outputs\summary_red.xlsx
outputs\summary_warm.xlsx
outputs\mask_mode_comparison.csv
outputs\mask_mode_summary.csv
```

其中：

- `volume_features.csv`：每张 ROI 图的分割状态、像素特征、厘米换算特征和 5 种体积。
- `predictions.csv`：每张图的真实重量、各体积公式与密度方法的预测重量和误差。
- `experiment_results.csv`：前 20 张训练密度、后 13 张验证误差的汇总。
- `cv_results.csv`：5-fold cross validation 结果。
- `summary_red.xlsx` / `summary_warm.xlsx`：汇总特征、预测、验证结果、CV 结果和推荐组合。
- `debug_masks_contact_sheet_red.jpg` / `debug_masks_contact_sheet_warm.jpg` 和对应 `outputs\debug_masks_*`：用于人工检查番茄 HSV 分割是否合理。
- `mask_mode_comparison.csv`：逐个 `idx` 对比 red 和 warm 的面积、尺寸、V3 体积、最佳预测重量和绝对误差。
- `mask_mode_summary.csv`：汇总 red 和 warm 各自最佳体积公式、密度方法、验证误差和对应 CV 均值。

第二阶段暂时使用全局尺度标定，不自动逐张识别立方体。

### red / warm 分割模式说明

- `red`：保留原始红色 HSV 双区间分割逻辑，适合红色成熟番茄。
- `warm`：扩大到红色、橙色和偏黄色番茄，重点改善偏黄样本 mask 不完整的问题，例如 `idx_032` 附近。

`volume_features.csv` 中会额外记录 `mask_mode`、`mask_area_ratio`、`bbox_x`、`bbox_y`、`bbox_w`、`bbox_h`、`contour_area_px`、`ellipse_area_px`、`area_to_ellipse_ratio`。如果 `area_to_ellipse_ratio` 明显偏小，通常说明 mask 可能只覆盖了番茄的一部分。

## 第三阶段准备：LaboroTomato 标注转二值语义分割 mask

本步骤只把 LaboroTomato 的 polygon JSON 标注转换为二值语义分割 mask，不训练模型，不修改原始图片或原始 JSON。

输入目录：

```text
tomato\Train\img
tomato\Train\ann
tomato\Test\img
tomato\Test\ann
```

所有以下类别会合并为 `tomato`：

```text
l_green
b_green
l_fully_ripened
l_half_ripened
b_half_ripened
b_fully_ripened
```

输出 mask 像素值：

```text
0 = background
1 = tomato
```

运行环境使用已有的 `PyTorch2.7`，不要安装新包：

```powershell
conda activate PyTorch2.7
cd D:\work\2026.6.11番茄
python .\scripts\04_convert_laboro_to_masks.py
```

输出文件：

```text
outputs\seg_dataset\Train\mask
outputs\seg_dataset\Test\mask
outputs\seg_dataset\train_manifest.csv
outputs\seg_dataset\test_manifest.csv
outputs\seg_dataset\class_stats.csv
outputs\seg_dataset\convert_warnings.csv
outputs\seg_dataset_preview\train_mask_preview.jpg
outputs\seg_dataset_preview\test_mask_preview.jpg
```

检查建议：

- 打开 `outputs\seg_dataset_preview\train_mask_preview.jpg` 和 `outputs\seg_dataset_preview\test_mask_preview.jpg`，确认原图、mask、半透明叠加大致对齐。
- 查看 `outputs\seg_dataset\convert_warnings.csv`，确认是否有缺图、缺 JSON、点数不足或尺寸不一致。
- 随机检查 mask PNG，确认像素值只有 `0` 和 `1`，不是 `255`。

## 清理 outputs 中间结果

清理脚本只移动文件到归档目录，不删除文件，不修改原始数据目录 `番茄数据` 或 `tomato`，也不修改已有实验脚本。

默认只预览清理计划，不移动任何文件：

```powershell
python .\scripts\00_cleanup_outputs.py
```

确认 dry-run 输出无误后，再执行归档：

```powershell
python .\scripts\00_cleanup_outputs.py --apply
```

归档目录格式：

```text
outputs_archive_cleanup_YYYYMMDD_HHMMSS
```

每次运行都会生成清理日志：

```text
outputs\cleanup_report_YYYYMMDD_HHMMSS.csv
```

当前清理策略：

- 保留图片/重量配对、ROI 裁剪、HSV red 正式结果、LaboroTomato mask 数据集转换结果。
- 将旧版未命名 HSV 结果和 warm 对照结果移动到归档目录。

## 语义分割训练流程

本阶段使用 `PyTorch2.7` 环境，不安装新包，不修改原始 `tomato` 数据集、原始 JSON 或原始图片。

输入数据来自：

```text
outputs\seg_dataset\train_manifest.csv
outputs\seg_dataset\test_manifest.csv
outputs\seg_dataset\Train\mask
outputs\seg_dataset\Test\mask
```

mask 标签：

```text
0 = background
1 = tomato
```

### 1. 检查 DataLoader

```powershell
conda activate PyTorch2.7
cd D:\work\2026.6.11番茄
python .\scripts\05_check_seg_dataloader.py
```

输出：

```text
outputs\seg_training_preview\dataloader_train_preview.jpg
outputs\seg_training_preview\dataloader_test_preview.jpg
```

请先检查预览图，确认 image、mask、overlay 对齐，且 mask unique values 只有 `0` 和 `1`。

### 2. 训练 baseline

默认训练 FCN-ResNet50：

```powershell
python .\scripts\06_train_segmentation_baseline.py --model fcn_resnet50 --epochs 20 --batch-size 2 --image-size 512
```

也可以训练 DeepLabV3-ResNet50：

```powershell
python .\scripts\06_train_segmentation_baseline.py --model deeplabv3_resnet50 --epochs 20 --batch-size 2 --image-size 512
```

如果显存不足：

```powershell
python .\scripts\06_train_segmentation_baseline.py --model fcn_resnet50 --epochs 20 --batch-size 1 --image-size 384
```

训练输出：

```text
outputs\seg_checkpoints\best_model.pth
outputs\seg_checkpoints\last_model.pth
outputs\seg_checkpoints\train_log.csv
outputs\seg_predictions_preview\epoch_XXX_preview.jpg
```

训练时会显示 batch 级 tqdm 进度条。预测预览图会把原图、真实 mask、预测 mask 和 overlay 统一缩放到固定 panel 尺寸；mask 面板使用最近邻插值，避免标签边缘被插值成非类别值。

训练脚本默认从头训练，不主动联网下载权重。如需尝试 torchvision 预训练权重，可加 `--pretrained`；如果权重不可用，会自动退回 `weights=None`。

### 3. 用训练好的模型预测 33 张 ROI

```powershell
python .\scripts\07_predict_roi_with_seg_model.py
```

输出：

```text
outputs\seg_roi_predictions
outputs\seg_roi_predictions\roi_segmentation_preview.jpg
```

当前阶段只做 ROI 语义分割预测和可视化，暂不接入体积估算或重量预测。

## 使用语义分割 mask 进行体积估重

当语义分割模型训练完成，并且已经对 33 张固定 ROI 图像完成预测后，可以使用预测 mask 替代 HSV 阈值分割来重新计算面积、长轴、短轴、体积、密度和验证集重量误差。

推荐流程：

```powershell
conda activate PyTorch2.7
cd D:\work\2026.6.11番茄

# 1. 训练分割模型，得到 best_model.pth
python .\scripts\06_train_segmentation_baseline.py --model fcn_resnet50 --epochs 60 --batch-size 4 --image-size 512 --amp --cudnn-benchmark --cache-ram

# 2. 预测 33 张 ROI，得到 idx_XXX_pred_mask.png
python .\scripts\07_predict_roi_with_seg_model.py

# 3. 使用 seg mask 进行体积估重，并与 HSV-red baseline 对比
python .\scripts\08_run_volume_experiment_with_seg_mask.py
```

输入文件：

```text
outputs\roi_tomato_images
outputs\seg_roi_predictions\idx_XXX_pred_mask.png
outputs\weights.csv
outputs\scale_config.json
outputs\summary_red.xlsx
```

注意：

- `idx_XXX_pred_mask.png` 是 `0/1` 标签图，不是 `0/255` 可视化图；普通图片查看器看起来接近纯黑是正常的。
- 脚本计算时会使用 `mask > 0` 得到二值 mask；debug 图中会显示为 `mask * 255`。
- 体积估重时只保留预测 mask 的最大连通域作为目标番茄，避免零散噪声影响面积和轴长。
- HSV-red baseline 不会被覆盖；Seg-mask 与 HSV-red 的最佳方法对比写入 `summary_seg.xlsx`。

主要输出：

```text
outputs\summary_seg.xlsx
outputs\debug_masks_contact_sheet_seg.jpg
outputs\debug_masks_seg\idx_XXX_seg_debug.jpg
outputs\volume_results_seg\volume_features_seg.csv
outputs\volume_results_seg\predictions_seg.csv
outputs\volume_results_seg\experiment_results_seg.csv
outputs\volume_results_seg\cv_results_seg.csv
outputs\volume_results_seg\seg_vs_red_comparison.csv
outputs\volume_results_seg\warnings_seg.csv
```

请优先检查：

```text
outputs\debug_masks_contact_sheet_seg.jpg
outputs\summary_seg.xlsx
outputs\volume_results_seg\seg_vs_red_comparison.csv
```
## GitHub 版本管理说明

本仓库只保存代码和说明文档，用于代码审计、版本回退和记录开发过程。

不会提交以下内容：

- 原始数据集和原始图片；
- `outputs` 中的训练结果、检查图、预测结果和中间产物；
- 模型权重、checkpoint 和导出的模型文件；
- 图片、Excel 表格、压缩包和本地环境文件。

训练结果和实验输出默认保存在本地 `outputs` 目录。大文件、原始数据和训练产物需要继续在本地保存，或使用其他大文件/数据管理方式单独维护。
