# utils/image_loader.py

import os
from pathlib import Path
from PIL import Image
import numpy as np

from .tiff_handler import load_and_enhance_tiff

def load_image(path: str):
    """
    根据文件后缀自动判断：
      - 如果是 .tif / .tiff，则调用 load_and_enhance_tiff(path)，
        返回 (pil_img, lon_array, lat_array)。
      - 如果是其他常见格式 (.png/.jpg/.jpeg)，则直接用 PIL.Image.open 读取，
        返回 (pil_img, None, None)。

    :param path: 图像文件路径（可以是 .tif/.tiff，也可以是常见的 .png/.jpg/.jpeg）
    :return: (pil_img, lon, lat)
             • pil_img: PIL.Image，三通道 uint8
             • lon:   numpy.ndarray 或 None，shape=(H,W)，每个像素经度
             • lat:   numpy.ndarray 或 None，shape=(H,W)，每个像素纬度
    """
    ext = Path(path).suffix.lower()
    if ext in ('.tif', '.tiff'):
        # 对于 GeoTIFF，调用专门的加载与增强函数
        pil_img, lon_arr, lat_arr = load_and_enhance_tiff(path)
        return pil_img, lon_arr, lat_arr
    else:
        # 对于普通图像，直接用 PIL 打开
        pil_img = Image.open(path).convert("RGB")
        return pil_img, None, None
