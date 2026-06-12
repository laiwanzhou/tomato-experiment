# 单张番茄图片质量预测交付包

本目录是一个尽量脱离原项目运行的交付包，用于对单张“一个番茄 + 一个 5cm 白色标定立方体”的图片进行番茄质量预测。

当前版本只支持一种尺度标定方式：运行时在图片窗口中点击白色 5cm 标定立方体的一条清晰边的两个端点。这样可以避免误用旧尺度，也更适合交付给普通使用者。

流程：

```text
输入图片
-> 点击白色 5cm 标定立方体一条边的两个端点
-> best_model.pth 语义分割番茄 mask
-> mask resize 回原图尺寸
-> 取最大连通域
-> 提取面积、长轴、短轴
-> 按点击标定结果换算厘米
-> V3 = pi/6 * L * W^2
-> pred_weight_g = rho * V3
```

默认参数来自当前实验的 Seg-mask 最优组合：

```text
model = fcn_resnet50
image_size = 512
volume_formula = V3_ellipsoid_H_eq_W
density_method = mean
rho = 1.0468 g/cm^3
cube_edge_cm = 5.0
```

## 目录结构

```text
single_predict/
├─ predict_single_tomato_weight.py
├─ README.md
├─ models/
│  └─ best_model.pth
├─ config/
│  └─ predictor_config.json
├─ examples/
│  └─ sample.jpg
├─ outputs/
└─ utils/
   ├─ segmentation_utils.py
   ├─ feature_utils.py
   └─ scale_utils.py
```

## 环境

见requirements.txt

## 尺度标定

运行脚本后会弹出图片窗口。请在白色 5cm 标定立方体正面选择一条清晰边，依次点击两个端点，然后按 `Enter` 或 `Space` 确认。

建议：

- 优先点击正面水平边或竖直边；
- 不要点击侧面透视边；
- 如果点错了，可以按 `Esc` 取消后重新运行。

## 方式 1：直接运行脚本

适合在 PyCharm 中直接点击运行脚本。

1. 打开：

```text
single_predict\predict_single_tomato_weight.py
```

2. 修改脚本底部的用户配置区：

```python
DIRECT_RUN_IMAGE = r".\examples\sample.jpg"
DIRECT_RUN_CALIBRATE_CLICK = True
DIRECT_RUN_CUBE_EDGE_CM = 5.0
```

3. 在 PyCharm 中直接运行脚本。

控制台会输出：

```text
==============================
预测番茄重量：... g
==============================
```

## 方式 2：命令行指定图片

```powershell
python .\single_predict\predict_single_tomato_weight.py --image ".\single_predict\examples\sample.jpg" --calibrate-click
```

指定 CPU：

```powershell
python .\single_predict\predict_single_tomato_weight.py --image ".\single_predict\examples\sample.jpg" --calibrate-click --device cpu
```

如果不加 `--calibrate-click`，脚本会报错提示必须通过点击标定物进行尺度标定。

## 输出

对于输入 `sample.jpg`，默认输出到：

```text
single_predict\outputs\sample\
```

生成：

```text
result.json
result.csv
pred_mask_01.png
pred_mask_vis.png
tomato_component_mask_vis.png
overlay.jpg
debug.jpg
```

说明：

- `pred_mask_01.png` 是 0/1 标签图，普通图片查看器看起来接近纯黑是正常的。
- `pred_mask_vis.png` 和 `tomato_component_mask_vis.png` 是 0/255 可视化图。
- `overlay.jpg` 显示番茄连通域叠加结果。
- `debug.jpg` 显示原图、mask、最大连通域、轮廓、椭圆、bbox、`L_cm`、`W_cm`、体积和预测重量。
- `result.json` / `result.csv` 保存完整数值结果。

对于result.csv的说明：
| 字段                      | 含义                   |
| ----------------------- | -------------------- |
| `image_path`            | 输入图片路径               |
| `checkpoint_path`       | 使用的模型权重路径            |
| `model`                 | 分割模型，例如 fcn_resnet50 |
| `image_size`            | 模型输入尺寸，例如 512        |
| `rho`                   | 使用的密度系数，当前是 1.0468   |
| `volume_formula`        | 使用的体积公式，当前是 V3       |
| `density_method`        | 密度方法，当前是 mean        |
| `cm_per_px`             | 点击标定后得到的像素到厘米比例      |
| `cube_edge_cm`          | 标定立方体边长，默认 5cm       |
| `long_axis_px`          | 番茄长轴，单位像素            |
| `short_axis_px`         | 番茄短轴，单位像素            |
| `L_cm`                  | 番茄长轴，单位厘米            |
| `W_cm`                  | 番茄短轴，单位厘米            |
| `volume_cm3`            | 按 V3 公式估算的体积         |
| `pred_weight_g`         | 最终预测重量，单位 g          |
| `area_px`               | 番茄区域像素面积             |
| `bbox_x/y/w/h`          | 番茄外接矩形位置             |
| `area_to_ellipse_ratio` | 实际 mask 面积和拟合椭圆面积的比例 |
