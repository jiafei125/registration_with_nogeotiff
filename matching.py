# -*- coding: utf-8 -*-
"""
RIFT2 算法实现
包含：相位一致性(PC)、最大索引图(MIM)、FAST特征检测、DIV旋转不变描述子、FSC+RANSAC外点剔除。
光学与SAR使用独立的PC参数，适配不同噪声特性。
"""

import cv2
import numpy as np
import os
import json
from scipy.fft import fft2, ifft2, ifftshift

# =========================================================================
# 参数配置区 —— 光学与SAR分开设置
# =========================================================================
OPT_PC_PARAMS = {
    'nscale': 4,
    'norient': 6,
    'sigma_r': 0.55,
    'noise_threshold': 0.0,
    'min_wavelength': 3.0,
}

SAR_PC_PARAMS = {
    'nscale': 4,
    'norient': 6,
    'sigma_r': 0.55,
    'noise_threshold': 0.0,
    'min_wavelength': 3.0,
}

MATCH_PARAMS = {
    'norient': 6,
    'fast_thresh': 15,
    'max_kps': 5000,
    'max_per_cell': 12,
    'border_margin': 64,
    'patch_size': 128,
    'grid_dim': 8,
    'dip_ratio': 1.0,
    'ratio_thresh': 1.0,
    'fsc_thresh': 10.0,
    'fsc_max_iters': 30000,
    'min_scale': 0.98,
    'max_scale': 1.02,
    'max_rotation_deg': 2.0,
    'max_translation': 150.0,
    'max_initial_displacement': 150.0,
    'min_inliers': 10,
    'min_inlier_span_ratio': 0.25,
    'structure_blur_sigma': 4.0,
    'structure_patch_size': 64,
    'structure_grid_step': 64,
    'structure_search_radius': 20,
    'structure_min_ncc': 0.5,
    'structure_max_error': 15.0,
    'structure_min_scale': 0.95,
    'structure_max_scale': 1.05,
    'structure_max_rotation_deg': 4.0,
    'match_line_thickness': 3,
}


# =========================================================================
# 1. Log-Gabor 滤波器组
# =========================================================================
def build_log_gabor_filters(rows, cols, params):
    nscale = params['nscale']
    norient = params['norient']
    sigma_r = params['sigma_r']
    min_wavelength = params['min_wavelength']

    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    u, v = np.meshgrid(x, y)
    radius = np.maximum(np.sqrt(u ** 2 + v ** 2), 1e-10)
    theta = np.arctan2(v, u)

    filters = []
    for s in range(nscale):
        wavelength = min_wavelength * (2 ** s)
        f0 = 1.0 / wavelength
        log_gabor = np.exp(-(np.log(radius / f0)) ** 2 / (2 * (np.log(sigma_r) ** 2)))
        log_gabor[0, 0] = 0
        for o in range(norient):
            angle = o * np.pi / norient
            dtheta = np.pi / (2 * norient)
            dt = theta - angle
            spread = np.exp(-(np.arctan2(np.sin(dt), np.cos(dt)) ** 2) / (2 * dtheta ** 2))
            filters.append((ifftshift(log_gabor * spread), angle))
    return filters


# =========================================================================
# 2. PC 矩图 + MIM
# =========================================================================
def compute_pc_moments_and_mim(image, params):
    rows, cols = image.shape
    img_fft = fft2(image)
    filters = build_log_gabor_filters(rows, cols, params)
    norient = params['norient']
    noise_th = params['noise_threshold']

    PC_theta = np.zeros((norient, rows, cols), dtype=np.float32)
    response_amplitudes = np.zeros((norient, rows, cols), dtype=np.float32)
    theta_o = np.zeros(norient, dtype=np.float32)

    for o in range(norient):
        sum_even_o = np.zeros_like(image)
        sum_odd_o = np.zeros_like(image)
        sum_amp_o = np.zeros_like(image)
        for s in range(params['nscale']):
            filt, angle = filters[s * norient + o]
            resp = ifft2(img_fft * filt)
            even, odd = np.real(resp), np.imag(resp)
            amp = np.sqrt(even ** 2 + odd ** 2)
            sum_even_o += even
            sum_odd_o += odd
            sum_amp_o += amp

        mean_phase = np.arctan2(sum_odd_o, sum_even_o)
        sum_phase_dev_o = np.zeros_like(image)
        for s in range(params['nscale']):
            filt, angle = filters[s * norient + o]
            resp = ifft2(img_fft * filt)
            even, odd = np.real(resp), np.imag(resp)
            amp = np.sqrt(even ** 2 + odd ** 2)
            phase = np.arctan2(odd, even)
            delta_phi = np.cos(phase - mean_phase) - np.abs(np.sin(phase - mean_phase))
            sum_phase_dev_o += amp * delta_phi

        sum_phase_dev_o = np.maximum(sum_phase_dev_o - noise_th, 0.0)
        PC_theta[o] = np.clip(sum_phase_dev_o / (sum_amp_o + 1e-8), 0, 1)
        response_amplitudes[o] = sum_amp_o
        theta_o[o] = angle

    # 不变矩 Mmax, Mmin
    cos_t = np.cos(theta_o).reshape(-1, 1, 1)
    sin_t = np.sin(theta_o).reshape(-1, 1, 1)
    a = np.sum(PC_theta * cos_t ** 2, axis=0)
    b = 2 * np.sum(PC_theta * cos_t * sin_t, axis=0)
    c = np.sum(PC_theta * sin_t ** 2, axis=0)
    trace = a + c
    det = np.maximum(a * c - (b / 2) ** 2, 0)
    sqrt_term = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0) + 1e-8)

    Mmax = (trace + sqrt_term) / 2
    Mmin = (trace - sqrt_term) / 2
    denom = Mmax.max() - Mmin.min() + 1e-8
    Mmax = (Mmax - Mmax.min()) / denom
    Mmin = (Mmin - Mmin.min()) / denom

    # MIM: 多尺度幅值响应最强的方向索引
    MIM = np.argmax(response_amplitudes, axis=0).astype(np.float32)

    for arr in (Mmax, Mmin, MIM):
        np.nan_to_num(arr, copy=False, nan=0.0)

    return Mmax, Mmin, MIM


# =========================================================================
# 3. FAST 特征检测 + 网格化分布
# =========================================================================
def detect_keypoints_on_moments(Mmax, Mmin, params, grid_size=20):
    def _fast_on_map(moment_map, thresh):
        m_u8 = (np.clip(moment_map, 0, 1) * 255).astype(np.uint8)
        fast = cv2.FastFeatureDetector_create(threshold=thresh, nonmaxSuppression=True)
        return fast.detect(m_u8, None)

    kps_corner = _fast_on_map(Mmin, params['fast_thresh'])
    kps_edge = _fast_on_map(Mmax, params['fast_thresh'])

    # 合并去重
    merged = {}
    for kp in (kps_corner + kps_edge):
        key = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        if key not in merged:
            merged[key] = kp
    kps = list(merged.values())

    # 剔除边界点
    border = params['border_margin']
    h, w = Mmax.shape
    kps = [kp for kp in kps
           if border < kp.pt[0] < w - border and border < kp.pt[1] < h - border]

    # 网格化：保证空间均匀分布，农田区也能出点
    cell_h, cell_w = h / grid_size, w / grid_size
    grid_dict = {}
    for kp in kps:
        r, c = int(kp.pt[1] / cell_h), int(kp.pt[0] / cell_w)
        grid_dict.setdefault((r, c), []).append(kp)

    final_kps = []
    for cell, pts in grid_dict.items():
        pts.sort(key=lambda k: k.response, reverse=True)
        final_kps.extend(pts[:params.get('max_per_cell', 8)])

    final_kps.sort(key=lambda k: k.response, reverse=True)
    final_kps = final_kps[:params['max_kps']]
    print(f"  [FAST检测] 提取 {len(final_kps)} 个点")
    return final_kps


# =========================================================================
# 4. RIFT2 描述子：MIM 直方图 + DIV 相对化 + 双峰策略
# =========================================================================
def compute_rift2_descriptor(MIM, keypoints, params):
    h, w = MIM.shape
    half = params['patch_size'] // 2
    sub = params['patch_size'] // params['grid_dim']
    norient = params['norient']

    des_list, valid_kps = [], []

    for kp in keypoints:
        x0, y0 = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if x0 - half < 0 or x0 + half >= w or y0 - half < 0 or y0 + half >= h:
            continue

        patch = MIM[y0 - half:y0 + half, x0 - half:x0 + half].astype(np.int32)
        hist, _ = np.histogram(patch, bins=norient, range=(0, norient))

        peak = np.max(hist)
        second = np.sort(hist)[-2] if norient >= 2 else 0
        s_main = int(np.argmax(hist))

        candidates = [s_main]
        if second >= params['dip_ratio'] * peak:
            s_second = int(np.argsort(hist)[-2])
            if s_second != s_main:
                candidates.append(s_second)

        for s_idx in candidates:
            rel_mim = (patch - s_idx + norient) % norient
            desc = np.zeros((params['grid_dim'], params['grid_dim'], norient), dtype=np.float32)

            for i in range(params['grid_dim']):
                for j in range(params['grid_dim']):
                    blk = rel_mim[i * sub:(i + 1) * sub, j * sub:(j + 1) * sub]
                    for b in range(norient):
                        desc[i, j, b] = np.sum(blk == b)

            flat = desc.reshape(-1).astype(np.float32)
            flat /= (np.sum(flat) + 1e-8)

            des_list.append(flat)
            valid_kps.append(kp)

    return valid_kps, np.array(des_list, dtype=np.float32)


# =========================================================================
# 5. FSC 外点剔除
# =========================================================================
def _similarity_from_two_points(p1, p2):
    d1 = p1[1] - p1[0]
    d2 = p2[1] - p2[0]
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 < 1e-8:
        return None
    scale = n2 / n1
    theta = np.arctan2(d2[1], d2[0]) - np.arctan2(d1[1], d1[0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]]) * scale
    t = p2[0] - R @ p1[0]
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = t
    return M


def _similarity_is_valid(H, params, structural=False):
    if H is None:
        return False
    scale = np.hypot(H[0, 0], H[1, 0])
    angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
    translation = np.linalg.norm(H[:2, 2])
    if structural:
        min_scale = params['structure_min_scale']
        max_scale = params['structure_max_scale']
        max_rotation = params['structure_max_rotation_deg']
    else:
        min_scale = params['min_scale']
        max_scale = params['max_scale']
        max_rotation = params['max_rotation_deg']
    return (min_scale <= scale <= max_scale and
            abs(angle) <= max_rotation and
            translation <= params['max_translation'])


def _fit_similarity_weighted(pts1, pts2, weights=None):
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts1) < 2:
        return None
    if weights is None:
        weights = np.ones(len(pts1), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total < 1e-12:
        return None

    c1 = np.average(pts1, axis=0, weights=weights)
    c2 = np.average(pts2, axis=0, weights=weights)
    x = pts1 - c1
    y = pts2 - c2
    den = np.sum(weights * np.sum(x * x, axis=1))
    if den < 1e-12:
        return None

    a = np.sum(weights * np.sum(x * y, axis=1)) / den
    b = np.sum(weights * (x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])) / den
    H = np.eye(3, dtype=np.float64)
    H[:2, :2] = [[a, -b], [b, a]]
    H[:2, 2] = c2 - H[:2, :2] @ c1
    return H


def _fit_similarity(pts1, pts2):
    return _fit_similarity_weighted(pts1, pts2)


def fsc_outlier_removal(pts1, pts2, params):
    n = len(pts1)
    if n < 4:
        return np.array([], dtype=int), None

    displacement = np.linalg.norm(pts2 - pts1, axis=1)
    candidates = np.where(displacement < params['max_initial_displacement'])[0]
    if len(candidates) < 4:
        return np.array([], dtype=int), None

    best_inliers = np.array([], dtype=int)
    best_H = None
    rng = np.random.default_rng(19)

    for _ in range(params['fsc_max_iters']):
        idx = rng.choice(candidates, 2, replace=False)
        M = _similarity_from_two_points(pts1[idx], pts2[idx])
        if M is None or not _similarity_is_valid(M, params):
            continue

        proj = pts1 @ M[:2, :2].T + M[:2, 2]
        err = np.linalg.norm(pts2 - proj, axis=1)
        inliers = np.where(err < params['fsc_thresh'])[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = M

    if len(best_inliers) < 4:
        return np.array([], dtype=int), None

    H = best_H
    for _ in range(5):
        candidate_H = _fit_similarity(pts1[best_inliers], pts2[best_inliers])
        if candidate_H is None or not _similarity_is_valid(candidate_H, params):
            break
        H = candidate_H
        proj = pts1 @ H[:2, :2].T + H[:2, 2]
        err = np.linalg.norm(pts2 - proj, axis=1)
        new_inliers = np.where(err < params['fsc_thresh'])[0]
        if len(new_inliers) < 4:
            break
        if np.array_equal(new_inliers, best_inliers):
            break
        best_inliers = new_inliers

    return best_inliers, H


def _registration_is_reliable(H, pairs, image_shape, params):
    if H is None or len(pairs) < params['min_inliers']:
        return False
    if not _similarity_is_valid(H, params):
        return False

    pts = np.float32([pair["pt_optical_px"] for pair in pairs])
    span = np.ptp(pts, axis=0)
    height, width = image_shape
    min_span = params['min_inlier_span_ratio'] * np.array([width, height])
    return bool(np.all(span >= min_span))


def _draw_matches(img1, kp1, img2, kp2, matches, thickness=3):
    border = 10
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + border + img2.shape[1]
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    canvas[:img1.shape[0], :img1.shape[1]] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:img2.shape[0], img1.shape[1] + border:] = cv2.cvtColor(
        img2, cv2.COLOR_GRAY2BGR
    )

    rng = np.random.default_rng(7)
    for match in matches:
        color = tuple(int(c) for c in rng.integers(40, 256, 3))
        p1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        p2 = tuple(np.round(np.array(kp2[match.trainIdx].pt) +
                            np.array([img1.shape[1] + border, 0])).astype(int))
        cv2.line(canvas, p1, p2, color, thickness=thickness,
                 lineType=cv2.LINE_AA)
        point_radius = max(2, thickness)
        cv2.circle(canvas, p1, point_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, point_radius, color, -1, cv2.LINE_AA)

    return canvas


# =========================================================================
# 6. 匹配主流程
# =========================================================================
def match_and_fsc(kp1, des1, kp2, des2, img1, img2, params):
    if len(des1) == 0 or len(des2) == 0:
        return None, None, None, None, []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn_matches:
        if m.distance < params['ratio_thresh'] * n.distance:
            good.append(m)
    print(f"[匹配] Ratio Test ({params['ratio_thresh']}): {len(good)} 对")

    if len(good) < 4:
        return None, None, None, None, []

    good.sort(key=lambda x: x.distance)
    unique = []
    used_q, used_t = set(), set()
    for m in good:
        if m.queryIdx not in used_q and m.trainIdx not in used_t:
            unique.append(m)
            used_q.add(m.queryIdx)
            used_t.add(m.trainIdx)
    print(f"[匹配] 一对一过滤后: {len(unique)} 对")

    if len(unique) < 4:
        return None, None, None, None, []

    img1_u8 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    img2_u8 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
    initial_img = _draw_matches(
        img1_u8, kp1, img2_u8, kp2, unique,
        params['match_line_thickness']
    )

    pts1 = np.float32([kp1[m.queryIdx].pt for m in unique])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in unique])
    inlier_idx, H = fsc_outlier_removal(pts1, pts2, params)

    print(f"[FSC] 最终内点: {len(inlier_idx)} / {len(unique)}")
    if len(inlier_idx) < params['min_inliers']:
        return None, None, None, initial_img, []

    inlier_matches = [unique[i] for i in inlier_idx]
    in_pts1 = pts1[inlier_idx]
    in_pts2 = pts2[inlier_idx]
    proj = in_pts1 @ H[:2, :2].T + H[:2, 2]
    errors = np.linalg.norm(in_pts2 - proj, axis=1)

    scale = np.hypot(H[0, 0], H[1, 0])
    angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))
    print(f"[FSC] 尺度={scale:.6f}, 旋转={angle:.4f} deg")
    print(f"[FSC] 重投影误差 median={np.median(errors):.3f}, "
          f"mean={np.mean(errors):.3f}, max={np.max(errors):.3f} px")

    pairs = []
    for m, error in zip(inlier_matches, errors):
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        pairs.append({
            "pt_optical_px": [float(x1), float(y1)],
            "pt_sar_px": [float(x2), float(y2)],
            "descriptor_distance": float(m.distance),
            "reprojection_error_px": float(error),
            "matching_method": "rift2_fsc"
        })

    final_img = _draw_matches(
        img1_u8, kp1, img2_u8, kp2, inlier_matches,
        params['match_line_thickness']
    )

    return H, inlier_matches, final_img, initial_img, pairs


# =========================================================================
# 7. 弱结构场景：亮目标结构回退
# =========================================================================
def _bright_structure_map(image, percentile, sigma):
    response = np.maximum(image - np.percentile(image, percentile), 0.0)
    return cv2.GaussianBlur(response.astype(np.float32), (0, 0), float(sigma))


def _estimate_coarse_similarity(img1, img2, params):
    pad = int(params['max_translation'])
    border = 48
    sigma = params['structure_blur_sigma']
    a = _bright_structure_map(img1, 65, sigma)
    b = _bright_structure_map(img2, 65, sigma)
    padded = cv2.copyMakeBorder(
        b, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )

    best_score = -np.inf
    best_H = None
    scales = np.arange(params['min_scale'], params['max_scale'] + 0.001, 0.01)
    angles = np.arange(-params['max_rotation_deg'],
                       params['max_rotation_deg'] + 0.001, 0.5)

    for scale in scales:
        for angle_deg in angles:
            angle = np.deg2rad(angle_deg)
            A = np.array([
                [scale * np.cos(angle), -scale * np.sin(angle), 0.0],
                [scale * np.sin(angle), scale * np.cos(angle), 0.0]
            ], dtype=np.float32)
            warped = cv2.warpAffine(a, A, (a.shape[1], a.shape[0]))
            template = warped[border:-border, border:-border]
            result = cv2.matchTemplate(padded, template, cv2.TM_CCOEFF_NORMED)

            yy, xx = np.indices(result.shape)
            offset = A[:, :2] @ np.array([border, border], dtype=np.float32)
            tx = xx - pad - offset[0]
            ty = yy - pad - offset[1]
            result[(np.abs(tx) > pad) | (np.abs(ty) > pad)] = -np.inf

            score = float(result.max())
            if not np.isfinite(score) or score <= best_score:
                continue

            y, x = np.unravel_index(np.argmax(result), result.shape)
            t = np.array([x - pad, y - pad], dtype=np.float64) - offset
            H = np.eye(3, dtype=np.float64)
            H[:2, :2] = A[:, :2]
            H[:2, 2] = t
            best_score = score
            best_H = H

    return best_score, best_H


def _structural_block_matches(img1, img2, H, params, percentile):
    sigma = params['structure_blur_sigma']
    a = _bright_structure_map(img1, percentile, sigma)
    b = _bright_structure_map(img2, percentile, sigma)
    half = params['structure_patch_size'] // 2
    search = params['structure_search_radius']
    step = params['structure_grid_step']
    border = params['border_margin']
    matches = []

    for y in range(border, img1.shape[0] - border, step):
        for x in range(border, img1.shape[1] - border, step):
            template = a[y - half:y + half, x - half:x + half]
            if np.std(template) < 1e-6:
                continue

            pred = np.array([x, y], dtype=np.float64) @ H[:2, :2].T + H[:2, 2]
            cx, cy = np.round(pred).astype(int)
            x0 = cx - half - search
            y0 = cy - half - search
            x1 = cx + half + search
            y1 = cy + half + search
            if x0 < 0 or y0 < 0 or x1 > b.shape[1] or y1 > b.shape[0]:
                continue

            window = b[y0:y1, x0:x1]
            result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            if result.size == 0:
                continue

            _, score, _, loc = cv2.minMaxLoc(result)
            qx = x0 + loc[0] + half
            qy = y0 + loc[1] + half
            matches.append((x, y, qx, qy, float(score)))

    if not matches:
        return None, None, None
    matches = np.asarray(matches, dtype=np.float32)
    return matches[:, :2], matches[:, 2:4], matches[:, 4]


def _refine_structural_similarity(pts1, pts2, scores, H, params):
    current_H = H.copy()
    for _ in range(10):
        proj = pts1 @ current_H[:2, :2].T + current_H[:2, 2]
        errors = np.linalg.norm(pts2 - proj, axis=1)
        idx = ((scores >= params['structure_min_ncc']) &
               (errors < params['structure_max_error']))
        if np.count_nonzero(idx) < params['min_inliers']:
            break

        candidate_H = _fit_similarity_weighted(
            pts1[idx], pts2[idx], scores[idx]
        )
        if candidate_H is None or not _similarity_is_valid(
                candidate_H, params, structural=True):
            break

        change = np.linalg.norm(candidate_H[:2] - current_H[:2])
        current_H = candidate_H
        if change < 1e-5:
            break

    proj = pts1 @ current_H[:2, :2].T + current_H[:2, 2]
    errors = np.linalg.norm(pts2 - proj, axis=1)
    idx = ((scores >= params['structure_min_ncc']) &
           (errors < params['structure_max_error']))
    return current_H, idx, errors


def _transform_discrepancy(H1, H2, image_shape):
    height, width = image_shape
    pts = np.float32([
        [0, 0], [width - 1, 0], [0, height - 1],
        [width - 1, height - 1], [width / 2, height / 2]
    ])
    p1 = pts @ H1[:2, :2].T + H1[:2, 2]
    p2 = pts @ H2[:2, :2].T + H2[:2, 2]
    return float(np.mean(np.linalg.norm(p1 - p2, axis=1)))


def structural_registration(img_opt, img_sar, params):
    print("[回退] 估计亮结构粗变换 ...")
    coarse_score, coarse_H = _estimate_coarse_similarity(
        img_opt, img_sar, params
    )
    if coarse_H is None or coarse_score < 0.25:
        print(f"[回退] 粗变换相关性不足: {coarse_score:.4f}")
        return None

    scale = np.hypot(coarse_H[0, 0], coarse_H[1, 0])
    angle = np.degrees(np.arctan2(coarse_H[1, 0], coarse_H[0, 0]))
    print(f"[回退] 粗变换: NCC={coarse_score:.4f}, "
          f"scale={scale:.6f}, angle={angle:.4f}")

    results = {}
    for percentile in (55, 65, 75):
        pts1, pts2, scores = _structural_block_matches(
            img_opt, img_sar, coarse_H, params, percentile
        )
        if pts1 is None:
            print(f"[回退] 阈值 {percentile} 无有效块")
            return None

        H, idx, errors = _refine_structural_similarity(
            pts1, pts2, scores, coarse_H, params
        )
        count = int(np.count_nonzero(idx))
        span = np.ptp(pts1[idx], axis=0) if count else np.array([0, 0])
        height, width = img_opt.shape
        min_span = params['min_inlier_span_ratio'] * np.array([width, height])
        if count < params['min_inliers'] or not np.all(span >= min_span):
            print(f"[回退] 阈值 {percentile} 分布不足: n={count}, span={span}")
            return None

        results[percentile] = (H, pts1, pts2, scores, idx, errors)
        print(f"[回退] 阈值 {percentile}: n={count}, "
              f"scale={np.hypot(H[0, 0], H[1, 0]):.6f}, "
              f"angle={np.degrees(np.arctan2(H[1, 0], H[0, 0])):.4f}, "
              f"median_error={np.median(errors[idx]):.3f}")

    reference_H = results[65][0]
    discrepancies = {
        percentile: _transform_discrepancy(
            reference_H, results[percentile][0], img_opt.shape
        )
        for percentile in (55, 75)
    }
    if max(discrepancies.values()) > 10.0:
        print(f"[回退] 多阈值变换不一致: {discrepancies}")
        return None
    print(f"[回退] 多阈值一致性: {discrepancies}")

    H, pts1, pts2, scores, idx, errors = results[65]
    kp1 = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts1[idx]]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts2[idx]]
    matches = [
        cv2.DMatch(i, i, 1.0 - float(score))
        for i, score in enumerate(scores[idx])
    ]

    img1_u8 = (np.clip(img_opt, 0, 1) * 255).astype(np.uint8)
    img2_u8 = (np.clip(img_sar, 0, 1) * 255).astype(np.uint8)
    final_img = _draw_matches(
        img1_u8, kp1, img2_u8, kp2, matches,
        params['match_line_thickness']
    )

    pairs = []
    for (x1, y1), (x2, y2), score, error in zip(
            pts1[idx], pts2[idx], scores[idx], errors[idx]):
        pairs.append({
            "pt_optical_px": [float(x1), float(y1)],
            "pt_sar_px": [float(x2), float(y2)],
            "descriptor_distance": None,
            "structure_ncc": float(score),
            "reprojection_error_px": float(error),
            "matching_method": "bright_structure_ncc"
        })

    print(f"[回退] 有效结构匹配: {len(pairs)} 对")
    return H, final_img, pairs


# =========================================================================
# 8. 运行入口
# =========================================================================
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0) if img is not None else None


def run_rift2(opt_path, sar_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_opt, img_sar = load_gray(opt_path), load_gray(sar_path)
    if img_opt is None or img_sar is None:
        raise ValueError("图像读取失败，请检查路径")

    print("[1/4] 计算 PC 与 MIM ...")
    print(f"  光学: {OPT_PC_PARAMS}")
    Mmax_o, Mmin_o, MIM_o = compute_pc_moments_and_mim(img_opt, OPT_PC_PARAMS)
    print(f"  SAR:  {SAR_PC_PARAMS}")
    Mmax_s, Mmin_s, MIM_s = compute_pc_moments_and_mim(img_sar, SAR_PC_PARAMS)
    print(f"  PC非零比例: 光学={np.mean(Mmax_o > 0):.2%}, "
          f"SAR={np.mean(Mmax_s > 0):.2%}")

    print("[2/4] FAST 特征点检测 ...")
    kp_opt = detect_keypoints_on_moments(Mmax_o, Mmin_o, MATCH_PARAMS)
    kp_sar = detect_keypoints_on_moments(Mmax_s, Mmin_s, MATCH_PARAMS)

    print("[3/4] 构建 RIFT2 描述子 ...")
    kp_opt_v, des_opt = compute_rift2_descriptor(MIM_o, kp_opt, MATCH_PARAMS)
    kp_sar_v, des_sar = compute_rift2_descriptor(MIM_s, kp_sar, MATCH_PARAMS)
    print(f"  光学描述子: {len(des_opt)}, SAR描述子: {len(des_sar)}")

    print("[4/4] 匹配与受约束 FSC 剔除 ...")
    H, inliers, final_img, initial_img, pairs = match_and_fsc(
        kp_opt_v, des_opt, kp_sar_v, des_sar, img_opt, img_sar, MATCH_PARAMS
    )

    if not _registration_is_reliable(H, pairs, img_opt.shape, MATCH_PARAMS):
        if H is not None:
            print("[验证] RIFT2 结果未通过全图可靠性检查，启用亮结构回退")
        else:
            print("[验证] RIFT2 未得到可靠结果，启用亮结构回退")
        fallback = structural_registration(img_opt, img_sar, MATCH_PARAMS)
        if fallback is not None:
            H, final_img, pairs = fallback

    if initial_img is not None:
        cv2.imwrite(os.path.join(out_dir, "match_rift2_initial.png"), initial_img)

    if final_img is not None:
        cv2.imwrite(os.path.join(out_dir, "match_rift2_final.png"), final_img)
        np.savetxt(os.path.join(out_dir, "transform_opt_to_sar.txt"), H)
        print("[输出] 最终匹配图: match_rift2_final.png")
        print("[输出] 变换矩阵: transform_opt_to_sar.txt")
    else:
        print("[输出] 无有效匹配")

    if pairs:
        with open(os.path.join(out_dir, "match_pairs.json"), "w",
                  encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"[输出] 匹配点对: match_pairs.json ({len(pairs)} 对)")

    print(f"[完成] 配准{'成功' if H is not None else '失败'}")


if __name__ == "__main__":
    OPT_PRE = r"D:\Project_Pycharm\Study\registration_with_nogeotiff\output_pre\opt_pre.png"
    SAR_PRE = r"D:\Project_Pycharm\Study\registration_with_nogeotiff\output_pre\sar_pre.png"
    OUT_DIR = r"D:\Project_Pycharm\Study\registration_with_nogeotiff\output"
    run_rift2(OPT_PRE, SAR_PRE, OUT_DIR)
