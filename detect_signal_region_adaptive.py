import cv2
import numpy as np

def detect_signal_region_adaptive(data):
    """
    自适应吸收线检测：
    1. Otsu自动找有效列区间，排除黑边；
    2. 每行基于统计特性自适应扩展吸收线区域；
    3. 形态学闭合优化mask连续性。
    """
    H, W = data.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.float32)

    # —— 1. 自动提取有效列区间 —— #
    col_mean = np.mean(data, axis=0)
    norm = (col_mean - col_mean.min()) / (col_mean.max() - col_mean.min() + 1e-8)
    _, mask_valid = cv2.threshold(norm.astype(np.float32), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_valid = mask_valid.astype(bool)

    segments = []
    start = None
    for i, is_valid in enumerate(mask_valid):
        if is_valid and start is None:
            start = i
        if (not is_valid or i == W-1) and start is not None:
            end = i if is_valid else i-1
            segments.append((start, end))
            start = None

    if not segments:
        col_lo, col_hi = 0, W-1
    else:
        col_lo, col_hi = max(segments, key=lambda x: x[1]-x[0])
        col_lo, col_hi = max(0, col_lo), min(W-1, col_hi)

    # —— 2. 每行自适应扩展吸收线区域 —— #
    mask = np.zeros((H, W), dtype=np.float32)
    if col_lo > col_hi:
        return mask

    sub = data[:, col_lo:col_hi+1]

    for i in range(H):
        row_data = sub[i]
        min_val = np.min(row_data)
        mean_val = np.mean(row_data)
        std_val = np.std(row_data)

        intensity = (mean_val - min_val) / (mean_val + 1e-8)
        base_width = 5
        adaptive_width = int(base_width + intensity * 15)

        threshold = min_val + std_val * 0.5
        candidates = np.where(row_data <= threshold)[0]

        if len(candidates) > 0:
            start_idx = max(0, candidates[0] - adaptive_width)
            end_idx = min(len(row_data) - 1, candidates[-1] + adaptive_width)
            start_col = col_lo + start_idx
            end_col = col_lo + end_idx
            mask[i, start_col:end_col+1] = 1.0

    # —— 3. 形态学闭合优化mask —— #
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask