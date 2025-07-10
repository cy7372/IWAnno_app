import numpy as np
from scipy.ndimage import binary_dilation

# === 全局容差像素设定 ===
TOLERANCE = 3

def to_numpy(x):
    """Tensor 转 numpy 的兼容处理"""
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    return x

def miou(pred, target, num_classes=2):
    pred = to_numpy(pred)
    target = to_numpy(target)

    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        target_dilate = binary_dilation(target_cls, iterations=TOLERANCE)
        pred_dilate = binary_dilation(pred_cls, iterations=TOLERANCE)

        intersection = np.logical_and(pred_cls, target_dilate).sum()
        union = np.logical_or(pred_dilate, target_cls).sum()

        iou = np.nan if union == 0 else intersection / union
        iou_list.append(iou)

    return np.nanmean(iou_list)

def f1score(pred, target, num_classes=2):
    pred = to_numpy(pred)
    target = to_numpy(target)

    f1_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        target_dilate = binary_dilation(target_cls, iterations=TOLERANCE)

        tp = np.logical_and(pred_cls, target_dilate).sum()
        fp = np.logical_and(pred_cls, np.logical_not(target_dilate)).sum()
        fn = np.logical_and(np.logical_not(pred_cls), target_cls).sum()

        denom = 2 * tp + fp + fn
        f1 = np.nan if denom == 0 else (2 * tp) / denom
        f1_list.append(f1)

    return np.nanmean(f1_list)

def precision(pred, target, num_classes=2):
    pred = to_numpy(pred)
    target = to_numpy(target)

    precision_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        target_dilate = binary_dilation(target_cls, iterations=TOLERANCE)

        tp = np.logical_and(pred_cls, target_dilate).sum()
        fp = np.logical_and(pred_cls, np.logical_not(target_dilate)).sum()

        denom = tp + fp
        precision = np.nan if denom == 0 else tp / denom
        precision_list.append(precision)

    return np.nanmean(precision_list)

def recall(pred, target, num_classes=2):
    pred = to_numpy(pred)
    target = to_numpy(target)

    recall_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        # 注意：recall 对 target 的 dilation 不扩展（与 TP 的一致性）
        tp = np.logical_and(pred_cls, target_cls).sum()
        fn = np.logical_and(np.logical_not(pred_cls), target_cls).sum()

        denom = tp + fn
        recall = np.nan if denom == 0 else tp / denom
        recall_list.append(recall)

    return np.nanmean(recall_list)
