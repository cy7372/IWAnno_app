from PIL import Image
from predictor import ModelPredictor
from utils.visualizer import PredictionVisualizer

if __name__ == '__main__':
    # ----------------------------------------------------------------------------
    # 模型加载与预测
    # ----------------------------------------------------------------------------
    # 创建 ModelPredictor 对象，并指定加载的模型类型（可选项： "FY4A", "FY4B", "GK2A"）
    # 以及是否对预测结果进行小噪点去除处理（remove_small_noises）
    # ModelPredictor 内部会自动加载位于 models 文件夹中的 FY4A.onnx 模型
    # 并初始化图像预测所需的切片预测引擎。
    model = ModelPredictor("FY4A", remove_small_noises=True)

    # 利用 predictor 对指定输入图像进行预测
    # 输入图像路径为 "example_input.png"，返回的预测结果为 NumPy 数组
    # 数组中像素值通常为 0 或 255，便于后续的数值处理与定量分析
    result = model.predict(input_image_path="example_input.png")

    # ----------------------------------------------------------------------------
    # 结果展示与导出（使用 PredictionVisualizer 统一接口）
    # ----------------------------------------------------------------------------
    # 创建 PredictionVisualizer 对象，用于展示或保存预测结果图像
    # 需要指定输出文件夹（此处为 "outputs"）和默认的输出文件名（例如与输入文件同名）
    visualizer = PredictionVisualizer(output_folder="outputs", default_filename="example_input.png")

    # 1. 保存仅包含预测结果的图像
    # 使用 mode="prediction" 表示只输出预测结果，不显示原始输入图像
    # 该方法会将预测结果转换为图像并保存到指定输出文件夹中
    visualizer.save(result, mode="prediction")

    # 2. 显示输入图像与预测结果的并排对比图
    # 先读取原始输入图像（转换为 RGB 模式），再调用 show 方法
    # 使用 mode="comparison" 生成对比图，左侧为原始图像，右侧为预测结果图像
    # 并直接调用系统图像查看器显示该对比图。
    input_image = Image.open("example_input.png").convert("RGB")
    visualizer.show(result, mode="comparison", input_image=input_image)

    # 3. 保存覆盖图，将预测结果以红色作为掩码覆盖在原始输入图像上
    # 使用 mode="overlay"，并设置 overlay_alpha=0.5 表示覆盖图的透明度为 50%
    # 最终生成的图像会将红色覆盖区域与原始图像混合后保存到输出文件夹中
    visualizer.save(result, mode="overlay", input_image=input_image, overlay_alpha=0.5)
