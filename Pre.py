import rasterio
import numpy as np
import cv2
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ===================== 配置区 =====================
OPT_4326_TIF = r"D:\Project_Pycharm\Study\registration_with_nogeotiff\data\optical_ori.tif"
SAR_RAW_PATH = r"D:\Project_Pycharm\Study\registration_with_nogeotiff\data\BP_EFS_260325_1247.jpg"
OUT_UTM_OPT_TIF = "output_pre/opt_utm_3b.tif"
UTM_EPSG = 32650
SAR_NATIVE_RES = 0.2    # SAR原始分辨率 m/px

OUT_CROP_OPT_TIF = "output_pre/opt_crop.tif"
OUT_CROP_SAR_PNG = "output_pre/sar_crop.png"
OUT_OPT_GRAY_PNG = "output_pre/opt_crop.png"
PREVIEW_MAX_LONG = 2000
# =================================================

def reproject_4326_to_utm(src_tif_path, dst_utm_tif, utm_epsg):
    """光学图：4326转UTM，输出GeoTIFF"""
    with rasterio.open(src_tif_path) as src:
        src_data = src.read([1,2,3])
        src_crs = src.crs
        dst_crs = rasterio.crs.CRS.from_epsg(utm_epsg)
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst_profile = src.profile.copy()
        dst_profile.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "count": 3
        })
        dst_arr = np.zeros((3, height, width), dtype=src_data.dtype)
        for i in range(3):
            reproject(
                source=src_data[i],
                destination=dst_arr[i],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
        with rasterio.open(dst_utm_tif, "w", **dst_profile) as dst:
            dst.write(dst_arr)
        opt_res = transform.a
        print(f"✅【光学UTM】分辨率：{opt_res:.4f} m/px，尺寸 H,W = {height},{width}")
        return dst_arr, dst_profile, transform, opt_res

def optical_rgb2gray(rgb_arr):
    """光学转灰度，用于配准输入"""
    r,g,b = rgb_arr[0], rgb_arr[1], rgb_arr[2]
    gray = 0.299*r + 0.587*g + 0.114*b
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def sar_downsample_to_opt_res(sar_jpg_path, sar_native_res, target_opt_res):
    """SAR降采样到光学分辨率（基准），返回原始降采样数组；额外返回增强预览图"""
    sar_raw = cv2.imread(sar_jpg_path, cv2.IMREAD_GRAYSCALE)
    if sar_raw is None:
        raise FileNotFoundError("SAR图片读取失败，检查路径")
    h,w = sar_raw.shape
    # 地面总宽高
    ground_w = w * sar_native_res
    ground_h = h * sar_native_res
    new_w = int(round(ground_w / target_opt_res))
    new_h = int(round(ground_h / target_opt_res))
    # 正式数据：降采样，INTER_AREA适合下采样
    sar_ds = cv2.resize(sar_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 预览增强：百分比截断拉伸，解决SAR太黑看不清，只给框选用，不参与裁剪
    p2, p98 = np.percentile(sar_ds, (2,98))
    sar_preview = np.clip(sar_ds, p2, p98)
    sar_preview = cv2.normalize(sar_preview, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(f"✅【SAR降采样】原始 {w}×{h} → 目标分辨率 {target_opt_res:.4f}m，新尺寸 {new_h}×{new_w}")
    return sar_ds, sar_preview

def interactive_roi(img_data, preview_img, win_title, max_long=1600):
    """独立框选函数：
    img_data：原始数组（用来裁剪）
    preview_img：增强后的预览图（用来显示给人看）
    """
    h, w = preview_img.shape[:2]
    scale = max_long / max(h, w)
    if scale < 1.0:
        preview_show = cv2.resize(preview_img, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
    else:
        preview_show = preview_img.copy()
        scale = 1.0

    roi_box = None
    drawing = False
    ix, iy = -1, -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_box, drawing, ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                tmp = preview_show.copy()
                cv2.rectangle(tmp, (ix, iy), (x, y), (0,255,0), 2)
                cv2.imshow(win_title, tmp)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1p, y1p = min(ix, x), min(iy, y)
            x2p, y2p = max(ix, x), max(iy, y)
            roi_box = (x1p, y1p, x2p, y2p)
            print(f"🔍【{win_title}】预览框像素：{roi_box}")

    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_title, mouse_callback)
    cv2.imshow(win_title, preview_show)
    print(f"👉【{win_title}】拖动鼠标框选大致区域，选完按 q 关闭窗口")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    if roi_box is None:
        return None
    x1p, y1p, x2p, y2p = roi_box
    # 映射回原图数组坐标
    x1 = int(x1p / scale)
    y1 = int(y1p / scale)
    x2 = int(x2p / scale)
    y2 = int(y2p / scale)
    # 钳位，防止越界
    h_data, w_data = img_data.shape
    x1 = np.clip(x1, 0, w_data)
    x2 = np.clip(x2, 0, w_data)
    y1 = np.clip(y1, 0, h_data)
    y2 = np.clip(y2, 0, h_data)
    print(f"✅【{win_title}】原图裁剪坐标：x1={x1},y1={y1},x2={x2},y2={y2}")
    return (x1,y1,x2,y2)

if __name__ == "__main__":
    # 1. 光学预处理
    opt_rgb, opt_profile, opt_tfm, opt_res = reproject_4326_to_utm(OPT_4326_TIF, OUT_UTM_OPT_TIF, UTM_EPSG)
    opt_gray = optical_rgb2gray(opt_rgb)

    # 2. SAR降采样到光学分辨率基准，返回【原始数据数组 + 增强预览图】
    sar_ds_data, sar_preview_img = sar_downsample_to_opt_res(SAR_RAW_PATH, SAR_NATIVE_RES, opt_res)

    # 3. 先后人工框选：光学图、SAR图分开
    opt_roi = interactive_roi(opt_gray, opt_gray, "光学图（基准）框选", PREVIEW_MAX_LONG)
    sar_roi = interactive_roi(sar_ds_data, sar_preview_img, "SAR图（对照地物框选）", PREVIEW_MAX_LONG)
    if opt_roi is None or sar_roi is None:
        raise RuntimeError("❌ 有图像未完成框选，退出")

    x1o,y1o,x2o,y2o = opt_roi
    x1s,y1s,x2s,y2s = sar_roi

    # 4. 裁剪
    opt_crop_rgb = opt_rgb[:, y1o:y2o, x1o:x2o]
    opt_crop_gray = opt_gray[y1o:y2o, x1o:x2o]
    sar_crop = sar_ds_data[y1s:y2s, x1s:x2s]

    if sar_crop.size == 0 or opt_crop_gray.size == 0:
        raise RuntimeError("❌ 裁剪区域为空！不要紧贴图片边缘框选")

    # 5. 光学输出：保留地理信息的GeoTIFF
    new_tfm = rasterio.transform.Affine(
        opt_tfm.a, opt_tfm.b, opt_tfm.c + x1o * opt_tfm.a,
        opt_tfm.d, opt_tfm.e, opt_tfm.f + y1o * opt_tfm.e
    )
    crop_profile = opt_profile.copy()
    crop_profile.update({
        "width": opt_crop_rgb.shape[2],
        "height": opt_crop_rgb.shape[1],
        "transform": new_tfm
    })
    with rasterio.open(OUT_CROP_OPT_TIF, "w", **crop_profile) as dst:
        dst.write(opt_crop_rgb)

    # 6. SAR输出PNG，光学灰度输出PNG
    cv2.imwrite(OUT_CROP_SAR_PNG, sar_crop)
    cv2.imwrite(OUT_OPT_GRAY_PNG, opt_crop_gray)

    print("\n===== 输出完成 =====")
    print(f"光学裁剪GeoTIFF：{OUT_CROP_OPT_TIF} （带UTM地理信息）")
    print(f"SAR裁剪PNG：{OUT_CROP_SAR_PNG}")
    print(f"光学灰度PNG：{OUT_OPT_GRAY_PNG}")
    print(f"裁剪后光学尺寸：{opt_crop_gray.shape}")
    print(f"裁剪后SAR尺寸：{sar_crop.shape}")
