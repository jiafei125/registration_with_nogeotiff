# 光-SAR 异源遥感图像配准算法 (基于改进 RIFT2)

本项目旨在实现光学图像与合成孔径雷达 (SAR) 图像的高精度自动配准。由于光-SAR 图像存在巨大的非线性辐射差异和几何畸变，传统 SIFT/SURF 等方法往往失效。本项目基于 **RIFT2 (Radiation-variant Invariant Feature Transform 2)** 算法核心思想，针对光-SAR 数据特性进行了深度的工程化改造，包括针对性的预处理、相位一致性参数分离、空间网格约束以及带先验几何限制的 FSC 外点剔除。

## 📂 项目目录结构

```text
registration_with_newdata/
│
├── data/                          # 原始数据存放目录 (tif, jpg)
├── output_pre/                    # 预处理中间结果输出目录
│   ├── opt_utm_3b.tif             # 重投影后的光学影像
│   ├── opt_crop.png               # 人工裁剪后的光学灰度图
│   ├── opt_pre.png                # 预处理后的光学图 (输入配准)
│   ├── sar_crop.png               # 人工裁剪后的SAR图
│   └── sar_pre.png                # 预处理后的SAR图 (输入配准)
│
├── output/                        # 配准结果输出目录
│   ├── match_rift2_initial.png    # 初筛匹配连线图
│   ├── match_rift2_final.png      # 最终内点连线图 (精配准结果)
│   ├── match_pairs.json           # 最终匹配点对坐标及误差
│   └── transform_opt_to_sar.txt   # 光学到SAR的几何变换矩阵 (H)
│
├── Pre.py                         # 步骤1：影像重投影、分辨率对齐与裁剪
├── PreProcess_opt.py              # 步骤2a：光学图像预处理
├── PreProcess_sar.py              # 步骤2b：SAR图像预处理
└── matching.py                    # 步骤3：核心配准算法 (RIFT2)

## ⚙️ 环境依赖

* Python 3.8+
* OpenCV (pip install opencv-python)
* NumPy (pip install numpy)
* SciPy (pip install scipy)
* Rasterio (pip install rasterio) - 用于处理 GeoTIFF 和投影转换

## 🚀 运行流程

请按照以下顺序依次执行脚本：

### 步骤 1：数据准备与物理尺度对齐
python Pre.py

* **操作**：程序会自动将光学图投影至 UTM 坐标系，并将 SAR 图降采样至与光学图相同的物理分辨率。随后会弹出窗口，**需要人工依次在光学图和 SAR 图上框选同一块地物区域**。
* **输出**：output_pre/opt_crop.png 和 output_pre/sar_crop.png。

### 步骤 2：图像预处理
python PreProcess_opt.py
python PreProcess_sar.py

* **光学**：仅做 1%-99% 的百分位截断，保留清晰的地物边缘，避免引入虚假纹理。
* **SAR**：采用中值滤波 + 双边滤波，在强力抑制相干斑噪声的同时，保护建筑和道路的几何边缘。
* **输出**：output_pre/opt_pre.png 和 output_pre/sar_pre.png。

### 步骤 3：特征提取与配准
python matching.py

* **算法流程**：
  1. 基于 Log-Gabor 滤波器计算两图的相位一致性（PC），获取 MIM 图与不变矩。
  2. 在 MIM 和最大/最小矩图上利用 FAST 算子提取特征点，并进行 20x20 网格化均匀约束。
  3. 构建基于 MIM 主导索引（DIV）的旋转不变描述子。
  4. KNN 匹配结合一对一约束（Unique Constraint）进行初筛。
  5. 利用带有**尺度 (0.98~1.02) 和旋转 (±2度) 先验约束**的 FSC (Fast Sample Consensus) 算法进行外点剔除。
* **输出**：在 output/ 目录下生成变换矩阵、匹配点对 JSON 文件及可视化连线图。

## 🧠 核心算法亮点

1. **抗辐射差异**：核心描述子完全基于相位一致性 (Phase Congruency) 的 MIM 特征，彻底摒弃了依赖灰度梯度的传统方式。
2. **空间均匀分布**：引入网格化 NMS 限制，确保农田、道路等弱纹理区域也能提取出足够数量的有效特征点。
3. **一对一约束**：针对 SAR 图像中强反射点（如金属、建筑物二次散射）导致的“多对一”误匹配，引入了严格的 Unique Constraint。
4. **先验几何约束 FSC**：由于前置的物理尺度对齐，配准模型被限制在极小的旋转和尺度变化范围内。FSC 在 RANSAC 采样时直接过滤掉不符合先验条件的模型，大幅提升了配准的鲁棒性和内点率。

## 📊 结果评估
最终配准效果可通过查看 match_rift2_final.png 进行直观评估。如果连线呈现高度平行的集中线束，且两端对应同名地物，则说明配准成功。match_pairs.json 中记录了每对点的重投影误差，可用于后续的斜距交会方程等定量解算。

