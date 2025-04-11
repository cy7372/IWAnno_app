import os
from PIL import Image
from utils.onnx_model_loader import ONNXModelLoader
from utils.patch_inference import PatchInferenceEngine

class ModelPredictor:
    """
    统一接口，完成模型加载和图片预测。
    
    用法示例:
        predictor = ModelPredictor("FY4A", remove_small_noises=True)
        result = predictor.predict("example_input.png")
    """
    def __init__(self, model_type, remove_small_noises=True):
        # 加载模型（调用 ONNXModelLoader）
        self.session = ONNXModelLoader(model_type).load_model()
        # 创建预测引擎（调用 PatchInferenceEngine，crop_size 已固定为 224）
        self.inference_engine = PatchInferenceEngine(remove_small_noises=remove_small_noises)

    def predict(self, input_image_path):
        """
        对输入图片进行预测，返回预测结果（PIL Image 对象）
        """
        image = Image.open(input_image_path).convert('RGB')
        return self.inference_engine.predict_image(image, self.session)
