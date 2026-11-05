# -*- coding: utf-8 -*-
"""
SAR 图像预处理
流程：读取 -> 中值滤波(去极值) -> 双边滤波(保边去斑) -> 百分位截断拉伸 -> 保存
"""
import cv2
import numpy as np
import os


def preprocess_sar(img_path, out_path, bilateral_d=9,
                   sigma_color=50, sigma_space=9):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到 SAR 图像: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    img_proc = cv2.medianBlur(img, 3)
    img_proc = cv2.bilateralFilter(
        img_proc, d=bilateral_d,
        sigmaColor=sigma_color, sigmaSpace=sigma_space
    )

    v_low = np.percentile(img_proc, 2)
    v_high = np.percentile(img_proc, 98)
    if (v_high - v_low) > 1e-6:
        img_clipped = np.clip(img_proc, v_low, v_high)
        img_norm = ((img_clipped - v_low) / (v_high - v_low) * 255).astype(np.uint8)
    else:
        img_norm = img_proc.astype(np.uint8)

    cv2.imwrite(out_path, img_norm)
    print(f"[SAR预处理完成] 输出: {out_path} | 尺寸: {img_norm.shape}")


if __name__ == "__main__":
    SAR_INPUT = r"D:\Project_Pycharm\Study\registration_with_nogeotiff\output_pre\sar_crop.png"
    SAR_OUTPUT = r"D:\Project_Pycharm\Study\registration_with_nogeotiff\output_pre\sar_pre.png"

    preprocess_sar(SAR_INPUT, SAR_OUTPUT)
