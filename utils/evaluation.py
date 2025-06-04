# utils/evaluation.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from dancher_tools.metrics import miou, f1score
from predictor import ModelPredictor
from utils.image_loader import load_image_auto

def calculate_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    计算二值预测与真值之间的 mIoU 和 F1-score，返回百分比（保留两位小数）。

    参数:
      pred: NumPy 数组，预测结果（像素值 0 或 255）。
      gt:   NumPy 数组，真值 mask（像素值 0 或 255）。

    返回:
      miou: Intersection over Union，百分比格式（例如 85.23 表示 85.23%）。
      f1:   F1-score，百分比格式。
    """
    pred_bin = (pred > 128).astype(np.uint8)
    gt_bin   = (gt   > 128).astype(np.uint8)

    # 交并
    intersection = np.sum(np.logical_and(pred_bin, gt_bin))
    union        = np.sum(np.logical_or (pred_bin, gt_bin))
    miou_val = (intersection / union) if (union != 0) else 0.0

    # F1
    pred_sum = np.sum(pred_bin)
    gt_sum   = np.sum(gt_bin)
    f1_val = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) != 0 else 0.0

    # 转换为百分比并保留两位小数
    return round(miou_val * 100, 2), round(f1_val * 100, 2)


class Evaluator:
    """
    Evaluator：统一评估器，根据输入（文件或文件夹）自动调用 predictor，输出 mIoU 和 F1-score。

    使用示例:
      from utils.evaluation import Evaluator
      predictor = ModelPredictor("FY4A", remove_small_noises=True)
      evaluator = Evaluator(predictor)

      # 单图评估：
      miou, f1 = evaluator.evaluate("path/to/image.png", "path/to/gt_mask.png")

      # 批量评估：
      avg_miou, avg_f1 = evaluator.evaluate("path/to/images_folder", "path/to/masks_folder")
    """

    def __init__(self, predictor: ModelPredictor):
        """
        :param predictor: 已初始化好的 ModelPredictor 实例，
                          其 predict(pil_img) 方法接受 PIL.Image 并返回 NumPy 二值掩码 (0/255)。
        """
        self.predictor = predictor

    def evaluate_file(self, img_path: str, gt_path: str):
        """
        对单张输入图像和单张真值 mask 做评估，返回 (mIoU%, F1%)。

        支持 .png/.jpg/.jpeg 以及 .tif/.tiff 输入。
        """
        # 1. 加载输入图像（自动区分 TIFF 和普通格式）
        pil_img, lon_arr, lat_arr = load_image_auto(img_path)

        # 2. predictor.predict 返回 NumPy 二值掩码 (0/255)
        pred_mask = self.predictor.predict(pil_img)

        # 3. 读取真值 mask（统一当作灰度图），转 NumPy
        gt_img = Image.open(gt_path).convert("L")
        gt_np = np.array(gt_img)

        # 4. 计算指标
        miou_val, f1_val = calculate_metrics(pred_mask, gt_np)
        return miou_val, f1_val

    def evaluate_folder(self, images_folder: str, masks_folder: str):
        """
        对图像文件夹与真值 mask 文件夹进行批量评估，返回平均 mIoU% 与平均 F1%。

        支持文件夹内混合 .png/.jpg/.jpeg/.tif/.tiff 格式，
        要求 images_folder 与 masks_folder 中同名文件一一对应（仅扩展名不同）。
        """
        # 1. 枚举 images_folder 下的所有支持格式
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        img_paths = []
        for e in exts:
            img_paths.extend(glob.glob(os.path.join(images_folder, e)))
        img_paths = sorted(img_paths)

        if not img_paths:
            raise ValueError(f"在 {images_folder} 下未找到任何支持格式的图像。")

        miou_list = []
        f1_list   = []

        for img_path in tqdm(img_paths, desc="Evaluating"):
            base_name = Path(img_path).stem  # 文件名（不含扩展名）

            # 2. 找对应的 ground-truth mask：尝试常见扩展名
            found_gt = None
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                cand = os.path.join(masks_folder, base_name + ext)
                if os.path.exists(cand):
                    found_gt = cand
                    break
            if found_gt is None:
                print(f"[WARNING] 跳过 {img_path}：找不到对应的真值 mask。")
                continue

            # 3. 走 evaluate_file 流程
            try:
                miou_val, f1_val = self.evaluate_file(img_path, found_gt)
            except Exception as e:
                print(f"[ERROR] 评估 {img_path} 时发生异常: {e}")
                continue

            miou_list.append(miou_val)
            f1_list.append(f1_val)

        # 4. 计算平均值（忽略空列表）
        if miou_list:
            avg_miou = round(np.mean(miou_list), 2)
        else:
            avg_miou = 0.0
        if f1_list:
            avg_f1 = round(np.mean(f1_list), 2)
        else:
            avg_f1 = 0.0

        return avg_miou, avg_f1

    def evaluate(self, input_path: str, gt_path: str):
        """
        通用评估接口：如果两个路径都是文件夹，就调用 evaluate_folder；
        如果两者都是文件，就调用 evaluate_file；否则报错。

        :param input_path: 图像文件路径或图像文件夹路径
        :param gt_path:    真值 mask 文件路径或文件夹路径
        :return: 单图 (mIoU%, F1%) 或 全集平均 (avg_mIoU%, avgF1%)
        """
        if os.path.isdir(input_path) and os.path.isdir(gt_path):
            return self.evaluate_folder(input_path, gt_path)
        elif os.path.isfile(input_path) and os.path.isfile(gt_path):
            return self.evaluate_file(input_path, gt_path)
        else:
            raise ValueError("输入路径与真值路径必须同时为文件或同时为文件夹。")

