# demo_evaluation.py
# ============================================================
# 对比预测掩码 vs. 真值掩码，输出平均 mIoU / F1 / Precision / Recall（百分比）
# ============================================================

import os
import glob
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.evaluation import calculate_metrics


def mean_dict(list_of_dicts: list[Dict[str, float]]) -> Dict[str, float]:
    """计算若干 dict 中每个键的平均值（保留两位小数）。"""
    keys = list_of_dicts[0].keys()
    return {
        k: round(float(np.mean([d[k] for d in list_of_dicts])), 2) if list_of_dicts else 0.0
        for k in keys
    }


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 0. 路径配置 —— 修改为你的预测结果与真值存放目录
    # -------------------------------------------------------------------------
    preds_folder = "datasets/eval/S1/model_anno"   # 预测掩码
    masks_folder = "datasets/eval/S1/ground_truth"   # 真值掩码

    # 支持的掩码扩展名
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")

    # -------------------------------------------------------------------------
    # 1. 枚举预测掩码文件
    # -------------------------------------------------------------------------
    pred_paths = sorted(p for e in exts for p in glob.glob(os.path.join(preds_folder, e)))
    if not pred_paths:
        raise ValueError(f"在 {preds_folder} 下未找到任何掩码文件。")

    # -------------------------------------------------------------------------
    # 2. 遍历并计算指标
    # -------------------------------------------------------------------------
    metrics_each = []  # 保存每张图的四指标 dict

    for pred_path in tqdm(pred_paths, desc="Evaluating"):
        base = Path(pred_path).stem

        # 查找对应的真值掩码
        gt_path = next(
            (
                os.path.join(masks_folder, base + ext)
                for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                if os.path.exists(os.path.join(masks_folder, base + ext))
            ),
            None,
        )
        if gt_path is None:
            print(f"[WARNING] 跳过 {pred_path}：未找到真值掩码")
            continue

        try:
            pred_np = np.array(Image.open(pred_path).convert("L"))
            gt_np = np.array(Image.open(gt_path).convert("L"))
            metrics = calculate_metrics(pred_np, gt_np)  # {'miou': .., 'f1': .., ...}
            metrics_each.append(metrics)
        except Exception as e:
            print(f"[ERROR] 评估 {pred_path} 失败：{e}")

    # -------------------------------------------------------------------------
    # 3. 计算并输出平均结果
    # -------------------------------------------------------------------------
    if not metrics_each:
        print("未成功评估任何样本。")
        exit()

    avg_metrics = mean_dict(metrics_each)

    print("========================================")
    print(f"预测掩码：{preds_folder}")
    print(f"真值掩码：{masks_folder}")
    print("----------------------------------------")
    print(f"Average mIoU     : {avg_metrics['miou']:.2f}%")
    print(f"Average F1-score : {avg_metrics['f1']:.2f}%")
    print(f"Average Precision: {avg_metrics['precision']:.2f}%")
    print(f"Average Recall   : {avg_metrics['recall']:.2f}%")
    print("========================================")
