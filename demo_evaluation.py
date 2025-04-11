import os
from predictor import ModelPredictor
from utils.evaluation import Evaluator

if __name__ == '__main__':
    # 定义图像和 mask 文件夹的路径
    images_folder = "../datasets/GOS/FY4A/test/images"
    masks_folder = "../datasets/GOS/FY4A/test/masks"
    
    # 创建 ModelPredictor 对象，加载指定类型模型（例如 "FY4A"），并开启小噪点去除功能
    predictor = ModelPredictor("FY4A", remove_small_noises=True)
    
    # 创建 Evaluator 对象，传入 predictor 用于生成预测结果
    evaluator = Evaluator(predictor)
    
    # 对文件夹中的图像和 mask 进行评估，返回所有图像的平均 mIoU 和 F1-score
    avg_miou, avg_f1 = evaluator.evaluate(images_folder, masks_folder)
    
    # 输出评估结果
    print("Average mIoU:", avg_miou)
    print("Average F1-score:", avg_f1)
