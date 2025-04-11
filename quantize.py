import os
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat

def quantize_onnx_model(fp32_model_path, quantized_model_path, per_channel=True):
    """
    对指定的 FP32 ONNX 模型进行动态量化，并保存量化后的模型。

    参数:
      fp32_model_path: 原始 FP32 模型文件路径，例如 "models/FY4A.onnx"
      quantized_model_path: 保存量化模型的路径，例如 "models/FY4A_quantized.onnx"
      per_channel: 是否对权重采用 per-channel 量化，默认为 True

    量化后的模型使用 INT8 数据类型（QuantType.QInt8），
    并采用 QDQ 格式进行表示，这有助于提高 GPU 环境下的兼容性和推理性能。
    """
    print("开始量化模型：", fp32_model_path)
    quantize_dynamic(
        model_input=fp32_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        quant_format=QuantFormat.QDQ  # 使用 QDQ 格式
    )
    print("量化完成，量化模型保存在：", quantized_model_path)

if __name__ == '__main__':
    # 定义原始模型和量化后模型的路径
    fp32_model = os.path.join("models", "FY4A.onnx")
    quant_model = os.path.join("models", "FY4A_quantized.onnx")
    
    # 执行动态量化
    quantize_onnx_model(fp32_model, quant_model, per_channel=True)
