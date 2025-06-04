# utils/patch_inference.py

import numpy as np
import torchvision.transforms as T
from PIL import Image
import scipy.ndimage as ndimage

class PatchInferenceEngine:
    """
    Patch Inference Engine：切片 → ONNX 推理 → 加权融合 → 输出二值掩码。
    """

    def __init__(self, onnx_session, remove_small_noises: bool = True):
        """
        :param onnx_session:        onnxruntime.InferenceSession
        :param remove_small_noises: 是否在最终输出时去除小连通域噪点
        """
        # 保留传进来的 ONNX Session
        self.session = onnx_session

        self.crop_size = 224
        self.stride = self.crop_size // 2
        self.remove_small_noises = remove_small_noises

        # 与训练时一致的预处理：Resize→ToTensor→Normalize
        self.transform = T.Compose([
            T.Resize((self.crop_size, self.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 生成固定 224×224 的 2D Gaussian 窗口（中心权重更高）
        y = np.linspace(-1, 1, self.crop_size)
        x = np.linspace(-1, 1, self.crop_size)
        xv, yv = np.meshgrid(x, y)
        self.weight_mask = np.exp(-(xv**2 + yv**2) * 4).astype(np.float32)  # shape=(224,224)

    # ------------------------------------------------------------
    # 切片（接口不变）
    # ------------------------------------------------------------
    def cut_image(self, image: Image.Image):
        """
        将 PIL Image 切成多个大小为 crop_size×crop_size 的块，步长为 stride（crop_size//2）。
        返回：
          - blocks: List[PIL.Image]，每个大小都是 (crop_size × crop_size)
          - positions: List[(x, y, w, h)]，表示每个 patch 在原图中的位置以及实际宽高
        """
        width, height = image.size
        blocks, positions = [], []
        for y in range(0, height - self.crop_size + 1, self.stride):
            for x in range(0, width - self.crop_size + 1, self.stride):
                x_end = min(x + self.crop_size, width)
                y_end = min(y + self.crop_size, height)
                patch = image.crop((x, y, x_end, y_end)).resize(
                    (self.crop_size, self.crop_size), Image.LANCZOS
                )
                blocks.append(patch)
                positions.append((x, y, x_end - x, y_end - y))
        return blocks, positions

    # ------------------------------------------------------------
    # 小噪点去除（接口不变）
    # ------------------------------------------------------------
    def remove_small_objects(self, mask: np.ndarray, min_size: int = 150) -> np.ndarray:
        """
        将二值掩码中连通域小于 min_size 的部分去掉。
        输入 mask: uint8 或 bool 数组，非零即视为前景。
        返回 uint8(mask)（0/255）。
        """
        binary = mask > 0
        lbl, num = ndimage.label(binary)
        sizes = ndimage.sum(binary, lbl, range(num + 1))
        out = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num + 1):
            if sizes[i] >= min_size:
                out[lbl == i] = 255
        return out

    # ------------------------------------------------------------
    # 返回 softmax 概率而不是直接 argmax 掩码
    # ------------------------------------------------------------
    def predict_block(self, block: Image.Image) -> np.ndarray:
        """
        对单张 patch 用 ONNX 推理，返回 softmax 概率张量，shape=(C, crop_size, crop_size)。
        """
        # 1) 预处理 → numpy(float32)
        x = self.transform(block).unsqueeze(0).cpu().numpy()  # (1,3,224,224)

        # 2) ONNX 推理
        logits = self.session.run(
            None, {self.session.get_inputs()[0].name: x}
        )[0]  # 得到 (1, C, 224, 224)

        probs = self._softmax_np(logits[0])  # (C, 224, 224)
        return probs

    @staticmethod
    def _softmax_np(arr: np.ndarray) -> np.ndarray:
        """
        Numpy 实现的 softmax，按第一个维度（类别维度）归一化：
        输入 arr.shape = (C, H, W)，返回同 shape 的概率张量。
        """
        # 为了数值稳定性，先减去每个像素位置上的最大值
        max_per_pixel = np.max(arr, axis=0, keepdims=True)  # (1, H, W)
        e = np.exp(arr - max_per_pixel)
        return e / np.sum(e, axis=0, keepdims=True)

    # ------------------------------------------------------------
    # 加权融合 logits → argmax
    # ------------------------------------------------------------
    def predict_image(self, image: Image.Image) -> np.ndarray:
        """
        主入口：对整张 PIL.Image 做切片推理 + 加权融合，返回 uint8 mask (0/255)。
        """
        blocks, positions = self.cut_image(image)
        if not blocks:
            # 如果整张图都比 224×224 小，就先全图 resize → 推理 → 再 resize 回去
            small_block = image.resize((self.crop_size, self.crop_size), Image.LANCZOS)
            probs_small = self.predict_block(small_block)  # (C,224,224)
            H, W = image.size[1], image.size[0]

            # 把 probs_small 按比例 resize 回 (H,W) 并加权
            full_logits = np.zeros((probs_small.shape[0], H, W), dtype=np.float32)
            full_weight = np.zeros((H, W), dtype=np.float32)
            wm_big = self.weight_mask

            for c in range(probs_small.shape[0]):
                resized_prob = np.array(
                    Image.fromarray((probs_small[c] * 255).astype(np.uint8))
                         .resize((W, H), Image.NEAREST), dtype=np.float32
                ) / 255.0
                resized_wm = np.array(
                    Image.fromarray((wm_big * 255).astype(np.uint8))
                         .resize((W, H), Image.NEAREST), dtype=np.float32
                ) / 255.0
                full_logits[c] += resized_prob * resized_wm
                full_weight += resized_wm

            full_logits /= np.maximum(full_weight, 1e-6)
            pred_small = (np.argmax(full_logits, axis=0).astype(np.uint8)) * 255
            if self.remove_small_noises:
                pred_small = self.remove_small_objects(pred_small)
            return pred_small

        # 第一块先跑一次，确定类别数
        first_probs = self.predict_block(blocks[0])  # (C,224,224)
        num_classes = first_probs.shape[0]
        H, W = image.size[1], image.size[0]

        full_logits = np.zeros((num_classes, H, W), dtype=np.float32)
        full_weight = np.zeros((H, W), dtype=np.float32)

        all_probs = [first_probs] + [self.predict_block(b) for b in blocks[1:]]
        for probs_c, (x, y, w, h) in zip(all_probs, positions):
            wm = self.weight_mask[:h, :w]
            for c in range(num_classes):
                full_logits[c, y:y+h, x:x+w] += probs_c[c, :h, :w] * wm
            full_weight[y:y+h, x:x+w] += wm

        full_logits /= np.maximum(full_weight, 1e-6)
        pred_mask = (np.argmax(full_logits, axis=0).astype(np.uint8)) * 255

        if self.remove_small_noises:
            pred_mask = self.remove_small_objects(pred_mask)

        return pred_mask

    def assemble_image(self, blocks, positions, original_size):
        """
        兼容旧接口：如果外部想自己做拼接，可以调用此方法。
        """
        full_mask = np.zeros(original_size, dtype=np.uint8)
        for block, (x, y, w, h) in zip(blocks, positions):
            arr = np.array(block.resize((w, h), Image.NEAREST))
            full_mask[y:y+h, x:x+w] = np.maximum(full_mask[y:y+h, x:x+w], arr)
        if self.remove_small_noises:
            full_mask = self.remove_small_objects(full_mask)
        return full_mask
