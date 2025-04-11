import os
from PIL import Image
import numpy as np

class PredictionVisualizer:
    """
    PredictionVisualizer：提供预测结果图像的展示和导出功能，接口风格类似于 matplotlib.pyplot。
    
    功能：
      - 将预测结果（NumPy 数组格式）转换为图像，并支持多种展示模式：
          "prediction"：仅显示预测结果图像；
          "comparison"：生成输入图像与预测结果的并排对比图；
          "overlay"：将预测结果（掩码）覆盖在输入图像上，生成融合图。
      - 提供 show() 方法直接显示图像，以及 save() 方法保存图像到指定文件夹。
    
    参数：
      output_folder: 图像保存的目标文件夹；
      default_filename: 默认输出文件名（例如 "example_input.png"），如果未在保存时指定，则使用此名称。
    """
    def __init__(self, output_folder, default_filename="output.png"):
        self.output_folder = output_folder
        self.default_filename = default_filename
        os.makedirs(self.output_folder, exist_ok=True)
    
    def _get_output_path(self, filename):
        """内部方法，根据文件名生成完整的保存路径。"""
        return os.path.join(self.output_folder, filename if filename else self.default_filename)
    
    def _prepare_image(self, prediction, mode="prediction", input_image=None, overlay_alpha=0.5):
        """
        根据指定的模式对预测结果和（可选）输入图像进行处理，生成最终展示的图像。
        
        参数：
          prediction: 预测结果，格式为 NumPy 数组（像素值为 0 或 255）。
          mode: 展示模式，支持以下几种：
                - "prediction": 直接显示预测结果图像；
                - "comparison": 并排显示输入图像与预测结果图像；
                - "overlay": 将预测结果作为掩码覆盖在输入图像上。
          input_image: 原始输入图像（PIL Image 对象），在 "comparison" 和 "overlay" 模式下必需。
          overlay_alpha: 当 mode 为 "overlay" 时，控制覆盖图的透明度，范围 0～1。
        
        返回：
          output_img: 最终处理后的图像，格式为 PIL Image 对象。
        """
        # 如果预测结果是 NumPy 数组，则转换为 PIL Image 对象
        if isinstance(prediction, np.ndarray):
            pred_img = Image.fromarray(prediction)
        else:
            pred_img = prediction

        if mode == "prediction":
            # 仅输出预测结果图像
            output_img = pred_img

        elif mode == "comparison":
            # 对比图模式：并排显示输入图像和预测结果
            if input_image is None:
                raise ValueError("在 'comparison' 模式下，必须提供原始输入图像。")
            # 调整输入图像大小，使之与预测图像尺寸一致
            input_resized = input_image.resize(pred_img.size)
            total_width = input_resized.width + pred_img.width
            max_height = max(input_resized.height, pred_img.height)
            output_img = Image.new("RGB", (total_width, max_height))
            output_img.paste(input_resized, (0, 0))
            output_img.paste(pred_img, (input_resized.width, 0))

        elif mode == "overlay":
            # 覆盖模式：将预测结果作为掩码覆盖在输入图像上
            if input_image is None:
                raise ValueError("在 'overlay' 模式下，必须提供原始输入图像。")
            # 将预测图像转换为灰度图作为掩码
            mask = pred_img.convert("L")
            # 将原始图像转换为 RGBA 以便混合
            base_img = input_image.convert("RGBA")
            # 创建一个全红色的覆盖图，透明度初始为0
            overlay_img = Image.new("RGBA", base_img.size, (255, 0, 0, 0))
            # 根据预测掩码设置 alpha 通道：非零部分设为 overlay_alpha 的透明度
            alpha_mask = mask.point(lambda p: int(overlay_alpha * 255) if p > 0 else 0)
            overlay_img.putalpha(alpha_mask)
            # 将覆盖图与原图混合
            output_img = Image.alpha_composite(base_img, overlay_img).convert("RGB")
        else:
            raise ValueError("无效的 mode 参数。支持 'prediction', 'comparison', 'overlay'。")
        return output_img
    
    def show(self, prediction, mode="prediction", input_image=None, overlay_alpha=0.5):
        """
        显示处理后的图像，类似于 plt.show()。
        
        参数：
          prediction: 预测结果，格式为 NumPy 数组。
          mode: 展示模式，详见 _prepare_image()。
          input_image: 在 'comparison' 或 'overlay' 模式下需要提供原始输入图像（PIL Image）。
          overlay_alpha: 在 'overlay' 模式下，覆盖的透明度。
        """
        output_img = self._prepare_image(prediction, mode=mode, input_image=input_image, overlay_alpha=overlay_alpha)
        output_img.show()
    
    def save(self, prediction, mode="prediction", input_image=None, overlay_alpha=0.5, filename=None):
        """
        保存处理后的图像到输出文件夹，类似于 plt.savefig()。
        
        参数：
          prediction: 预测结果，格式为 NumPy 数组。
          mode: 展示模式，详见 _prepare_image()。
          input_image: 在 'comparison' 或 'overlay' 模式下需要提供原始输入图像（PIL Image）。
          overlay_alpha: 在 'overlay' 模式下，覆盖的透明度。
          filename: 保存时使用的文件名。如果未指定，则根据模式自动生成默认文件名：
                    "example_input.png"（prediction），
                    "comparison_example_input.png"（comparison），
                    "overlay_example_input.png"（overlay）。
        
        返回：
          output_path: 完整的保存路径。
        """
        if filename is None:
            if mode == "prediction":
                filename = self.default_filename
            elif mode == "comparison":
                filename = "comparison_" + self.default_filename
            elif mode == "overlay":
                filename = "overlay_" + self.default_filename
            else:
                filename = self.default_filename
        output_img = self._prepare_image(prediction, mode=mode, input_image=input_image, overlay_alpha=overlay_alpha)
        output_path = self._get_output_path(filename)
        output_img.save(output_path)
        print(f"Image saved to: {output_path}")
        return output_path
