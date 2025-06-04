# utils/onnx_model_loader.py

import os
import onnxruntime as ort

class ONNXModelLoader:
    """
    负责加载 ONNX 模型。

    使用方式示例：
        loader = ONNXModelLoader("FY4A")      # 会在 models/FY4A.onnx 中查找
        session = loader.load_model()
    """

    def __init__(self, model_type: str, onnx_dir: str = None):
        """
        :param model_type: 数据集关键字或 ONNX 文件名（可带或不带 .onnx 后缀）。
        :param onnx_dir:   可选地指定 ONNX 模型所在的目录；
                           如果为 None，则默认使用项目根目录下的 models/ 子文件夹。
        """
        # 如果 model_type 中没有 .onnx，就自动补全后缀
        if not model_type.lower().endswith(".onnx"):
            model_filename = model_type + ".onnx"
        else:
            model_filename = model_type

        # 如果用户没传入 onnx_dir，就默认指向项目根目录下的 models/
        if onnx_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            onnx_dir = os.path.join(project_root, "models")

        # 构造完整的 ONNX 文件路径
        self.model_path = os.path.join(onnx_dir, model_filename)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"ONNX 模型文件不存在：{self.model_path}。\n"
                f"• 请确认目录 {onnx_dir} 下确实有 '{model_filename}'，"
                f"或检查传入的 model_type 是否拼写正确。"
            )

        self.session = None
        self.model_type = model_type

    def load_model(self) -> ort.InferenceSession:
        """
        加载 ONNX 模型，优先使用 CUDA，如果不可用则回退到 CPU。
        返回：onnxruntime.InferenceSession 实例
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            raise RuntimeError(f"加载 ONNX 模型时出错：{e}")

        print(f"ONNX model loaded: {os.path.basename(self.model_path)}")
        return self.session
