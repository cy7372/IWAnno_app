import numpy as np
import torchvision.transforms as T
from PIL import Image
import scipy.ndimage as ndimage


class PatchInferenceEngine:
    """
    Patch Inference Engine：切片-预测-融合（crop_size=224）。
    改进点：
        • 采用中心加权 Gaussian window 对 logits 进行 soft-voting 融合，
          缓解 patch 接缝断裂。
        • predict_block 返回 softmax 概率，最终一次 argmax。
    """
    def __init__(self, remove_small_noises: bool = True):
        self.crop_size = 224
        self.stride    = self.crop_size // 2
        self.remove_small_noises = remove_small_noises

        # 与训练一致的预处理
        self.transform = T.Compose([
            T.Resize((self.crop_size, self.crop_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 生成中心加权 2D Gaussian mask（固定 224×224）
        y = np.linspace(-1, 1, self.crop_size)
        x = np.linspace(-1, 1, self.crop_size)
        xv, yv = np.meshgrid(x, y)
        self.weight_mask = np.exp(-(xv ** 2 + yv ** 2) * 4).astype(np.float32)  # shape (H,W)

    # ------------------------------------------------------------
    # 切片（接口不变）
    # ------------------------------------------------------------
    def cut_image(self, image):
        width, height = image.size
        blocks, positions = [], []
        for y in range(0, height - self.crop_size + 1, self.stride):
            for x in range(0, width - self.crop_size + 1, self.stride):
                x_end = min(x + self.crop_size, width)
                y_end = min(y + self.crop_size, height)
                patch = image.crop((x, y, x_end, y_end)).resize(
                    (self.crop_size, self.crop_size), Image.LANCZOS)
                blocks.append(patch)
                positions.append((x, y, x_end - x, y_end - y))
        return blocks, positions

    # ------------------------------------------------------------
    # 小噪点去除（接口不变）
    # ------------------------------------------------------------
    def remove_small_objects(self, mask, min_size=150):
        binary = mask > 0
        lbl, num = ndimage.label(binary)
        sizes = ndimage.sum(binary, lbl, range(num + 1))
        out = np.zeros_like(mask)
        for i in range(1, num + 1):
            if sizes[i] >= min_size:
                out[lbl == i] = 255
        return out

    # ------------------------------------------------------------
    # 🔄 改进：返回 softmax 概率而不是 argmax mask
    # ------------------------------------------------------------
    def predict_block(self, block: Image.Image, session):
        """
        返回 softmax 概率张量 (C,H,W) —— 便于后续 soft-voting。
        """
        x = self.transform(block).unsqueeze(0).numpy()  # (1,3,224,224)
        logits = session.run(None, {session.get_inputs()[0].name: x})[0]  # (1,C,H,W)
        probs = self._softmax_np(logits[0])  # (C,H,W)
        return probs

    @staticmethod
    def _softmax_np(arr):
        e = np.exp(arr - np.max(arr, axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)

    # ------------------------------------------------------------
    # 🔄 改进：加权融合 logits → argmax
    # ------------------------------------------------------------
    def predict_image(self, image: Image.Image, session):
        """
        主入口：切片预测 + 加权融合，输出 uint8 mask (0/255)。
        """
        blocks, positions = self.cut_image(image)
        # 首块先推理，获取类别数
        first_probs = self.predict_block(blocks[0], session)
        num_classes = first_probs.shape[0]

        H, W = image.size[1], image.size[0]
        full_logits = np.zeros((num_classes, H, W), dtype=np.float32)
        full_weight = np.zeros((H, W), dtype=np.float32)

        # 逐块累积
        for block_probs, (x, y, w, h) in zip(
            [first_probs] + [self.predict_block(b, session) for b in blocks[1:]],
            positions
        ):
            # 高斯权窗口（若 w/h!=224 时裁切）
            wm = self.weight_mask[:h, :w]
            for c in range(num_classes):
                full_logits[c, y:y+h, x:x+w] += block_probs[c, :h, :w] * wm
            full_weight[y:y+h, x:x+w] += wm

        # 归一化并取 argmax
        full_logits /= np.maximum(full_weight, 1e-6)
        pred_mask = np.argmax(full_logits, axis=0).astype(np.uint8) * 255

        if self.remove_small_noises:
            pred_mask = self.remove_small_objects(pred_mask)

        return pred_mask

    # ------------------------------------------------------------
    # 兼容旧 assemble_image（若外部仍需直接调用）
    # ------------------------------------------------------------
    def assemble_image(self, blocks, positions, original_size):
        # 若仍想使用旧接口可调用，但推荐直接用 predict_image
        return super().assemble_image(blocks, positions, original_size)
