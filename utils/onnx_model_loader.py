import os
import onnxruntime as ort

class ONNXModelLoader:
    """
    Model Initialization: 负责加载 ONNX 模型。
    模型文件位于项目根目录下的 models 文件夹中，通过当前文件的路径来定位模型文件，
    根据输入的 model_type 自动补全文件名：如果输入不包含 ".onnx"，则自动加上后缀 ".onnx"，
    否则直接使用输入的文件名。

    支持的模型类型示例包括：
        - FY4A
        - FY4B
        - GK2A
        - MODIS
        - S1
    """
    def __init__(self, model_type):
        # 保存模型类型或模型文件名
        self.model_type = model_type
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 假设 models 文件夹位于项目根目录下，即 utils 文件夹的上一级目录中的 models 文件夹
        models_dir = os.path.join(current_dir, "..", "models")
        
        # 如果输入的 model_type 不包含 ".onnx" 后缀，则自动补全，否则直接使用输入值
        if not model_type.lower().endswith(".onnx"):
            filename = f"{model_type}.onnx"
        else:
            filename = model_type
        
        # 构造完整的模型文件路径
        self.model_path = os.path.join(models_dir, filename)
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件 {self.model_path} 不存在，请检查 model_type 是否正确。")
        
        self.session = None

    def load_model(self):
        # 指定使用 CUDAExecutionProvider 优先，如果不可用则回退到 CPUExecutionProvider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        print("ONNX model loaded:", self.model_type)
        return self.session
