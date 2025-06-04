# utils/tiff_handler.py

import numpy as np
import rioxarray as rxr
from skimage import exposure
from PIL import Image

def load_and_enhance_tiff(tiff_path: str) -> (Image.Image, np.ndarray, np.ndarray):
    """
    读取单波段 GeoTIFF 并做内波增强（log → 1/99 percentile clip → CLAHE），
    返回：
      - PIL.Image（三通道 uint8 RGB，用于后续 ONNX 推理）
      - lon 经度网格 (H, W)
      - lat 纬度网格 (H, W)

    :param tiff_path: GeoTIFF 文件路径
    :return: (pil_img, lon, lat)
             • pil_img: PIL.Image，shape=(H, W, 3)，dtype=uint8
             • lon: numpy.ndarray，shape=(H, W)，对应每个像素的经度
             • lat: numpy.ndarray，shape=(H, W)，对应每个像素的纬度
    """
    # 用 rioxarray 打开 GeoTIFF
    ds = rxr.open_rasterio(tiff_path, masked=True)
    # 取第一波段的数据（假设单波段）
    dn = np.nan_to_num(ds[0].data, nan=0.0)

    # 1) 对数压缩
    dn = np.log1p(dn)
    # 2) 去掉极端值（1st percentile 和 99th percentile 之外的值剪裁掉）
    vmin, vmax = np.percentile(dn, [1, 99])
    dn = np.clip(dn, vmin, vmax)
    # 3) 归一化到 [0,1]
    dn = (dn - vmin) / (vmax - vmin + 1e-12)
    # 4) 局部自适应直方图均衡（CLAHE）以增强对比度
    dn = exposure.equalize_adapthist(dn, clip_limit=0.02)
    # 5) 转成 uint8
    dn_uint8 = (dn * 255).astype(np.uint8)

    # 构造三通道伪 RGB 图像
    rgb = np.stack([dn_uint8]*3, axis=-1)  # shape = (H, W, 3)
    pil_img = Image.fromarray(rgb)

    # 构造经纬度网格：rioxarray 的坐标 'x' 表示经度，'y' 表示纬度
    x = ds['x'].values  # shape = (W,)
    y = ds['y'].values  # shape = (H,)
    # meshgrid 注意 y 在第二维，x 在第一维
    lon, lat = np.meshgrid(x, y)

    return pil_img, lon, lat
