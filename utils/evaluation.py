import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_metrics(pred, gt):
    """
    计算二值预测与真值之间的 mIoU 和 F1-score 指标，
    并将结果转换为百分比（保留两位小数）。

    参数:
      pred: 预测结果，NumPy 数组（像素值为 0 或 255）。
      gt: 真值 mask，NumPy 数组（像素值为 0 或 255）。

    返回:
      miou: Intersection over Union 指标，百分比格式（例如 85.23 表示 85.23%）。
      f1: F1-score 指标，百分比格式。
    """
    # 将像素值转换为二值（0 和 1）
    pred_bin = (pred > 128).astype(np.uint8)
    gt_bin = (gt > 128).astype(np.uint8)
    
    intersection = np.sum(np.logical_and(pred_bin, gt_bin))
    union = np.sum(np.logical_or(pred_bin, gt_bin))
    miou = intersection / union if union != 0 else 0

    pred_sum = np.sum(pred_bin)
    gt_sum = np.sum(gt_bin)
    f1 = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) != 0 else 0

    # 转换为百分比并保留两位小数
    return round(miou * 100, 2), round(f1 * 100, 2)

class Evaluator:
    """
    Evaluator：统一评估器，根据输入路径判断是单个文件评估还是文件夹批量评估，
    自动生成预测结果，并计算 mIoU 和 F1-score 指标（以百分比输出，保留两位小数）。

    使用方法:
      evaluator = Evaluator(predictor)
      # 对单个文件进行评估：
      miou, f1 = evaluator.evaluate("path/to/prediction.png", "path/to/ground_truth.png")
      # 对文件夹进行评估（要求输入图像和 mask 文件夹中文件名一一对应）：
      avg_miou, avg_f1 = evaluator.evaluate("path/to/images_folder", "path/to/masks_folder")
    """
    def __init__(self, predictor):
        """
        初始化 Evaluator 对象。

        参数:
          predictor: ModelPredictor 对象，用于生成预测结果（返回 NumPy 数组）。
        """
        self.predictor = predictor

    def evaluate_file(self, pred_path, gt_path):
        """
        对单个预测文件与对应真值 mask 文件进行评估，并返回 mIoU 和 F1-score（百分比）。
        """
        pred_img = Image.open(pred_path).convert("L")
        gt_img = Image.open(gt_path).convert("L")
        pred_np = np.array(pred_img)
        gt_np = np.array(gt_img)
        return calculate_metrics(pred_np, gt_np)

    def evaluate_folder(self, images_folder, masks_folder):
        """
        对图像文件夹与真值 mask 文件夹进行批量评估，要求两个文件夹中文件名一一对应。
        使用 tqdm 显示处理进度。

        参数:
          images_folder: 输入图像文件夹路径。
          masks_folder: 真值 mask 文件夹路径。

        返回:
          avg_miou, avg_f1: 所有图像的平均 mIoU 和 F1-score（百分比）。
        """
        image_files = sorted([f for f in os.listdir(images_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        mask_files = sorted([f for f in os.listdir(masks_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if len(image_files) != len(mask_files):
            raise ValueError("输入图像和真值 mask 文件夹中的文件数量不匹配。")
        
        miou_list = []
        f1_list = []
        for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Evaluating"):
            img_path = os.path.join(images_folder, img_file)
            mask_path = os.path.join(masks_folder, mask_file)
            # 使用 predictor 对每个输入图像生成预测结果（返回 NumPy 数组）
            pred = self.predictor.predict(input_image_path=img_path)
            gt = np.array(Image.open(mask_path).convert("L"))
            miou, f1 = calculate_metrics(pred, gt)
            miou_list.append(miou)
            f1_list.append(f1)
        avg_miou = round(np.mean(miou_list), 2) if miou_list else 0
        avg_f1 = round(np.mean(f1_list), 2) if f1_list else 0
        return avg_miou, avg_f1

    def evaluate(self, input_path, gt_path):
        """
        统一评估接口，根据输入判断是单个文件评估还是文件夹批量评估。

        参数:
          input_path: 预测结果的文件路径或输入图像文件夹路径。
          gt_path: 真值 mask 的文件路径或文件夹路径。

        返回:
          如果输入为文件，则返回单个图像的 mIoU 和 F1-score（百分比）；
          如果输入为文件夹，则返回所有图像的平均 mIoU 和 F1-score（百分比）。
        """
        if os.path.isdir(input_path) and os.path.isdir(gt_path):
            return self.evaluate_folder(input_path, gt_path)
        elif os.path.isfile(input_path) and os.path.isfile(gt_path):
            return self.evaluate_file(input_path, gt_path)
        else:
            raise ValueError("输入路径和真值路径应同时为文件或同时为文件夹。")
