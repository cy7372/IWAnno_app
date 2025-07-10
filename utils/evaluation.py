# utils/evaluation.py
# ============================================================
# 统一评估脚本：支持单图 / 批量，输出 mIoU、F1、Precision、Recall（百分比）
# ============================================================

import os
import glob
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.metrics import miou, f1score, precision, recall
from predictor import ModelPredictor
from utils.image_loader import load_image_auto


# ---------- 指标计算 ----------
def _to_bin(arr: np.ndarray) -> np.ndarray:
    """把 0/255 灰度掩码转为 0/1 二值。"""
    return (arr > 128).astype(np.uint8)


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    调用 4 个指标函数，统一返回百分比（保留两位小数）。
    """
    pred_bin, gt_bin = _to_bin(pred), _to_bin(gt)

    results = {
        "miou": miou(pred_bin, gt_bin) * 100,
        "f1": f1score(pred_bin, gt_bin) * 100,
        "precision": precision(pred_bin, gt_bin) * 100,
        "recall": recall(pred_bin, gt_bin) * 100,
    }
    return {k: round(v, 2) for k, v in results.items()}


# ---------- 评估器 ----------
class Evaluator:
    """
    Evaluator：统一评估器，根据输入（文件或文件夹）自动调用 predictor，
    输出 mIoU / F1 / Precision / Recall（百分比）。

    示例
    ----
    >>> predictor = ModelPredictor("FY4A", remove_small_noises=True)
    >>> evaluator = Evaluator(predictor)
    >>> metrics = evaluator.evaluate("image.png", "mask.png")
    >>> print(metrics)   # {'miou': 87.11, 'f1': 91.35, 'precision': 89.77, 'recall': 93.02}
    """

    def __init__(self, predictor: ModelPredictor):
        """
        Parameters
        ----------
        predictor : ModelPredictor
            已初始化好的 ModelPredictor，需提供 predict(pil_img)→NumPy(0/255) 接口。
        """
        self.predictor = predictor

    # ----- 单图 -----
    def evaluate_file(self, img_path: str, gt_path: str) -> Dict[str, float]:
        """
        单张评估，返回 4 指标字典（百分比）。
        """
        pil_img, *_ = load_image_auto(img_path)
        pred_mask = self.predictor.predict(pil_img)

        gt_np = np.array(Image.open(gt_path).convert("L"))
        return calculate_metrics(pred_mask, gt_np)

    # ----- 批量 -----
    def evaluate_folder(self, images_folder: str, masks_folder: str) -> Dict[str, float]:
        """
        批量评估：对文件夹内全部图像进行计算，返回平均指标（百分比）。
        """
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        img_paths = sorted(
            p for e in exts for p in glob.glob(os.path.join(images_folder, e))
        )
        if not img_paths:
            raise ValueError(f"在 {images_folder} 下未找到任何支持格式的图像。")

        metrics_accum = {"miou": [], "f1": [], "precision": [], "recall": []}

        for img_path in tqdm(img_paths, desc="Evaluating"):
            base = Path(img_path).stem
            # 查找同名 mask
            gt_path = next(
                (
                    os.path.join(masks_folder, base + ext)
                    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                    if os.path.exists(os.path.join(masks_folder, base + ext))
                ),
                None,
            )
            if gt_path is None:
                print(f"[WARNING] 跳过 {img_path}：未找到真值 mask")
                continue

            try:
                metrics = self.evaluate_file(img_path, gt_path)
            except Exception as e:
                print(f"[ERROR] 评估 {img_path} 失败：{e}")
                continue

            for k in metrics_accum:
                metrics_accum[k].append(metrics[k])

        return {
            k: round(float(np.mean(v)), 2) if v else 0.0
            for k, v in metrics_accum.items()
        }

    # ----- 通用入口 -----
    def evaluate(self, input_path: str, gt_path: str) -> Dict[str, float]:
        """
        若输入、真值均为文件夹→批量；均为文件→单图；否则抛错。
        """
        if os.path.isdir(input_path) and os.path.isdir(gt_path):
            return self.evaluate_folder(input_path, gt_path)
        if os.path.isfile(input_path) and os.path.isfile(gt_path):
            return self.evaluate_file(input_path, gt_path)
        raise ValueError("input_path 与 gt_path 必须同时为文件或同时为文件夹")
