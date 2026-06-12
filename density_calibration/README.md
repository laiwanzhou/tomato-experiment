# 番茄密度标定交付包

本文件夹用于根据一组已知真实重量的番茄图片计算估重模型的密度系数 `rho`。它不是用来预测未知重量，而是用来标定后续预测所需的密度参数。

每张图片应只包含：

```text
一个番茄 + 一个 5cm 白色标定立方体
```

脚本会使用 `best_model.pth` 分割番茄，提取最大连通域的长轴 `L` 和短轴 `W`，使用 V3 体积公式：

```text
V = pi / 6 * L * W^2
```

再根据真实重量反推单张图片密度：

```text
rho_i = weight_g / V
```

最后汇总得到对外使用的密度系数 `rho_mean`。

## 目录结构

```text
density_calibration/
├─ calibrate_density.py
├─ README.md
├─ requirements.txt
├─ models/
│  └─ best_model.pth
├─ config/
│  └─ calibration_config.json
├─ images/
├─ weights.csv
└─ outputs/
```

## 准备数据

1. 将参与密度计算的图片放入：

```text
density_calibration\images
```

2. 在 `density_calibration\weights.csv` 中填写真实重量：

```csv
image_name,weight_g
idx_001_roi.jpg,6.66
idx_002_roi.jpg,8.60
idx_003_roi.jpg,10.07
```

`image_name` 必须与 `images/` 中的文件名一致，`weight_g` 单位为 g。

当前交付包已从项目中复制了 3 张示例图片和对应重量，方便演示。

## 标定模式

默认模式：

```text
--calibration-mode group_once
```

含义：

```text
第一张图点击一次白色 5cm 立方体边
-> 计算 cm_per_px
-> 整组图片复用这个尺度
```

这适合同一个人、同一拍摄距离、同一焦距、同一裁剪方式采集的一组图片。

如果图片来自不同批次或拍摄距离不一致，可以使用：

```text
--calibration-mode per_image
```

含义：

```text
每张图片分别点击白色 5cm 立方体边
-> 每张图分别计算 cm_per_px
```

## 运行

第一次运行，默认整组图片点击一次标定：

```powershell
python .\density_calibration\calibrate_density.py
```

每张图片单独标定：

```powershell
python .\density_calibration\calibrate_density.py --calibration-mode per_image
```

复用上次保存的点击点：

```powershell
python .\density_calibration\calibrate_density.py --reuse-calibration-points
```

输出可用于单张预测的配置：

```powershell
python .\density_calibration\calibrate_density.py --reuse-calibration-points --save-updated-predictor-config
```

生成的配置文件：

```text
density_calibration\outputs\predictor_config_updated.json
```

可以复制到：

```text
single_predict\config\predictor_config.json
```

用于后续单张图片预测。

## 点击标定

脚本弹出图片窗口后，请点击白色 5cm 标定立方体正面一条清晰边的两个端点，然后按 `Enter` 或 `Space` 确认。

建议：

- 优先点击正面水平边或竖直边；
- 不要点击侧面透视边；
- 如果点错了，按 `Esc` 取消后重新运行。

## 输出

输出目录：

```text
density_calibration\outputs
```

主要文件：

```text
density_results.csv
density_summary.json
density_summary.csv
calibration_points.csv
debug\
```

如果启用 `--save-updated-predictor-config`，还会生成：

```text
predictor_config_updated.json
```

`density_results.csv` 每张图片一行，包含真实重量、体积、`rho_i`、长轴、短轴、bbox 等。

`density_summary.json` / `density_summary.csv` 汇总整组样本的对外密度结果：

```text
n_samples
rho_mean
```

`debug/` 中每张图片一张检查图，显示原图、预测 mask、最大连通域、overlay、轮廓/椭圆/bbox、标定边信息、体积、真实重量和 `rho_i`。

## 配置

配置文件：

```text
density_calibration\config\calibration_config.json
```

默认内容：

```json
{
  "model": "fcn_resnet50",
  "image_size": 512,
  "volume_formula": "V3_ellipsoid_H_eq_W",
  "density_method": "mean",
  "cube_edge_cm": 5.0
}
```

## 环境

推荐使用已有 `PyTorch2.7` 环境：

```powershell
conda activate PyTorch2.7
```

安装依赖：

```powershell
pip install -r .\density_calibration\requirements.txt
```
