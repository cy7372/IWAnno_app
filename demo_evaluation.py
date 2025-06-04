# demo_evaluation.py

import os
from pathlib import Path

from predictor import ModelPredictor
from utils.evaluation import Evaluator

if __name__ == '__main__':
    # ----------------------------------------------------------------------------
    # 0. 配置：修改为你自己的数据集路径（可以是普通图像文件夹或 TIFF 文件夹）
    # ----------------------------------------------------------------------------
    images_folder = "../datasets/GOS/FY4A/test/images"
    masks_folder  = "../datasets/GOS/FY4A/test/masks"

    # ----------------------------------------------------------------------------
    # 1. 初始化 ModelPredictor
    # ----------------------------------------------------------------------------
    # 这里传入 "FY4A" 表示会加载 models/FY4A.onnx（由 ONNXModelLoader 底层完成），
    # remove_small_noises=True 表示在 PatchInferenceEngine 里会去掉小连通域噪点。
    predictor = ModelPredictor("FY4A", remove_small_noises=True)

    # ----------------------------------------------------------------------------
    # 2. 创建 Evaluator，并调用 evaluate() 方法
    # ----------------------------------------------------------------------------
    evaluator = Evaluator(predictor)
    avg_miou, avg_f1 = evaluator.evaluate(images_folder, masks_folder)

    # ----------------------------------------------------------------------------
    # 3. 打印最终的平均 mIoU 和 F1-score（百分比）
    # ----------------------------------------------------------------------------
    print("========================================")
    print(f"数据集路径：{images_folder}  vs  {masks_folder}")
    print(f"Average mIoU  : {avg_miou:.2f}%")
    print(f"Average F1-score: {avg_f1:.2f}%")
    print("========================================")
