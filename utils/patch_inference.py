import numpy as np
import torchvision.transforms as T
from PIL import Image
import scipy.ndimage as ndimage

class PatchInferenceEngine:
    """
    Patch Inference Engine：对输入图像进行切片预测，再将各个预测结果拼接成完整的预测结果。
    本模块中，crop_size 固定为 224，且最终返回的预测结果为 NumPy 数组，
    方便后续进行数值分析和内波定量处理。
    """
    def __init__(self, remove_small_noises=True):
        self.crop_size = 224  # 固定切片尺寸为 224
        self.remove_small_noises = remove_small_noises
        # 定义与训练时一致的预处理流程，将图像调整为 (224, 224)，归一化后转换为 Tensor
        self.transform = T.Compose([
            T.Resize((self.crop_size, self.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def cut_image(self, image):
        """
        将输入图像裁剪为多个大小为 crop_size × crop_size 的块，
        步长设置为 crop_size//2（即50%重叠）。
        
        参数：
            image: PIL Image 对象
        
        返回：
            blocks: 包含所有切片的列表，每个切片为 PIL Image 对象
            positions: 每个切片在原图中的位置信息列表，格式为 (x, y, w, h)
        """
        width, height = image.size
        blocks = []
        positions = []
        step_size = self.crop_size // 2  # 使用50%重叠
        for y in range(0, height - self.crop_size + 1, step_size):
            for x in range(0, width - self.crop_size + 1, step_size):
                x_end = min(x + self.crop_size, width)
                y_end = min(y + self.crop_size, height)
                block = image.crop((x, y, x_end, y_end)).resize(
                    (self.crop_size, self.crop_size), Image.LANCZOS)
                blocks.append(block)
                positions.append((x, y, x_end - x, y_end - y))
        return blocks, positions

    def remove_small_objects(self, mask, min_size=150):
        """
        去除预测结果中连通区域像素数小于 min_size 的小噪点。
        
        参数：
            mask: NumPy 数组格式的二值化预测结果
            min_size: 最小连通区域的像素数
        
        返回：
            mask_cleaned: 去除小噪点后的 NumPy 数组预测结果
        """
        binary_mask = mask > 0
        label_im, num_labels = ndimage.label(binary_mask)
        sizes = ndimage.sum(binary_mask, label_im, range(num_labels + 1))
        mask_cleaned = np.zeros_like(mask)
        for i in range(1, num_labels + 1):
            if sizes[i] >= min_size:
                mask_cleaned[label_im == i] = 255
        return mask_cleaned

    def assemble_image(self, blocks, positions, original_size):
        """
        将各个切片预测结果（均为 NumPy 数组）拼接成一张完整图像，
        对于重叠区域采用逐像素取最大值的方法合并。
        如果 remove_small_noises 为 True，则对拼接结果进行噪点去除处理。
        
        参数：
            blocks: 每个切片预测结果组成的列表（每项为 NumPy 数组）
            positions: 对应每个切片在原图中的位置信息，格式为 (x, y, w, h)
            original_size: 原图的尺寸，格式为 (height, width)
        
        返回：
            full_mask: 拼接后的完整预测结果，为 NumPy 数组
        """
        full_mask = np.zeros(original_size, dtype=np.uint8)
        for block, (x, y, w, h) in zip(blocks, positions):
            # 调整每个切片大小以匹配原图中对应区域的尺寸
            block = np.array(Image.fromarray(block).resize((w, h), Image.NEAREST))
            full_mask[y:y + h, x:x + w] = np.maximum(
                full_mask[y:y + h, x:x + w], block)
        if self.remove_small_noises:
            full_mask = self.remove_small_objects(full_mask)
        return full_mask

    def predict_block(self, block, session):
        """
        对单个图像切片使用 ONNX 模型进行预测。
        
        参数：
            block: 单个图像切片，PIL Image 对象
            session: ONNX 模型的 InferenceSession 对象
        
        返回：
            pred_mask_img: 单个切片的预测结果，格式为 NumPy 数组（取值 0 或 255）
        """
        # 对切片进行预处理
        image_tensor = self.transform(block).unsqueeze(0)  # shape: (1, 3, 224, 224)
        input_array = image_tensor.numpy()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_array})
        logits = outputs[0]  # 假设输出 shape 为 (1, num_classes, H, W)
        pred_mask = np.argmax(logits, axis=1).squeeze()  # 得到预测类别，shape: (H, W)
        # 将类别（0/1）映射为 0/255，返回 NumPy 数组
        pred_mask_img = (pred_mask * 255).astype(np.uint8)
        return pred_mask_img

    def predict_image(self, image, session):
        """
        对整张图像进行切片、逐块预测后拼接，返回完整的预测结果。
        
        参数：
            image: 输入图像，PIL Image 对象
            session: ONNX 模型的 InferenceSession 对象
        
        返回：
            full_pred_mask: 完整的预测结果，为 NumPy 数组格式
        """
        # 分块并记录各块位置信息
        blocks, positions = self.cut_image(image)
        # 对每个切片进行预测，得到的结果均为 NumPy 数组
        pred_blocks = [self.predict_block(block, session) for block in blocks]
        # 获取原图尺寸（转换为 (height, width) 格式）
        original_size = image.size[::-1]
        # 拼接各个切片的预测结果，得到完整预测结果（NumPy 数组）
        full_pred_mask = self.assemble_image(pred_blocks, positions, original_size)
        return full_pred_mask
