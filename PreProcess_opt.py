# -*- coding: utf-8 -*-
"""
光学图像预处理
流程：读取 -> 百分位截断 -> 保存
光学图结构清晰，不做 CLAHE 等增强，避免引入 SAR 中不存在的虚假纹理。
"""
import cv2
import numpy as np
import os


def preprocess_optical(img_path, out_path, p_low=1, p_high=99):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到光学图像: {img_path}")

    # 1. 读取灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像，请检查文件格式: {img_path}")

    # 2. 百分位截断：剔除云影、极亮金属屋顶等异常值
    v_low = np.percentile(img, p_low)
    v_high = np.percentile(img, p_high)
    if (v_high - v_low) > 1e-6:
        img_clipped = np.clip(img, v_low, v_high)
        img_norm = ((img_clipped - v_low) / (v_high - v_low) * 255).astype(np.uint8)
    else:
        img_norm = img

    # 3. 输出保存
    cv2.imwrite(out_path, img_norm)
    print(f"[光学预处理完成] 输出: {out_path} | 尺寸: {img_norm.shape}")


if __name__ == "__main__":
    OPT_INPUT = r"D:\Project_Pycharm\Study\registration_with_newdata\output_pre\opt_crop.png"
    OPT_OUTPUT = r"D:\Project_Pycharm\Study\registration_with_newdata\output_pre\opt_pre.png"

    preprocess_optical(OPT_INPUT, OPT_OUTPUT)
