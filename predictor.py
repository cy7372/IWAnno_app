# predictor.py

from utils.onnx_model_loader   import ONNXModelLoader
from utils.patch_inference     import PatchInferenceEngine
from PIL import Image
import numpy as np

class ModelPredictor:
    """
    统一接口：ModelPredictor("FY4A") 在 models/FY4A.onnx 中加载模型，
    并用 PatchInferenceEngine 完成切片→ONNX 推理→加权融合，返回二值掩码。
    """

    def __init__(self, model_type: str, remove_small_noises: bool = True):
        """
        :param model_type:         ONNX 模型名称（比如 "FY4A"、"GK2A"、"GENERAL"，
                                   底层会自动在 models/ 里补上 ".onnx"）
        :param remove_small_noises: 是否在最终输出中去除小连通域噪点
        """
        # 1) 先读取 ONNX 文件并创建 InferenceSession
        loader = ONNXModelLoader(model_type)
        onnx_session = loader.load_model()  # 打印 "ONNX model loaded: <model_type>.onnx"

        # 2) 把 onnx_session 和 remove_small_noises 一起传给 PatchInferenceEngine
        self.inference_engine = PatchInferenceEngine(onnx_session, remove_small_noises)

    def predict(self, pil_img: Image.Image) -> np.ndarray:
        """
        :param pil_img: 单张 PIL.Image（任意分辨率）
        :return:       uint8 二值掩码（像素值为 0 或 255）
        """
        return self.inference_engine.predict_image(pil_img)
