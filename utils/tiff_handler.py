import numpy as np
import rioxarray as rxr
from skimage import exposure
from PIL import Image

def load_and_enhance_tiff(tiff_path: str) -> (Image.Image, np.ndarray, np.ndarray):
    """
    读取单波段 GeoTIFF 并做内波增强（log→1/99 percentile clip→CLAHE），
    返回：
      - PIL Image（uint8 RGB 三通道）
      - lon 经度网格 (H, W)
      - lat 纬度网格 (H, W)
    """
    ds = rxr.open_rasterio(tiff_path, masked=True)
    dn = np.nan_to_num(ds[0].data, nan=0.0)

    # 1) 对数压缩
    dn = np.log1p(dn)
    # 2) 去掉极端点
    vmin, vmax = np.percentile(dn, [1, 99])
    dn = np.clip(dn, vmin, vmax)
    # 3) 归一化到 [0,1]
    dn = (dn - vmin) / (vmax - vmin)
    # 4) 局部直方图均衡（CLAHE）
    dn = exposure.equalize_adapthist(dn, clip_limit=0.02)
    # 5) 转 uint8
    dn_uint8 = (dn * 255).astype(np.uint8)

    # 生成三通道伪 RGB
    rgb = np.stack([dn_uint8]*3, axis=-1)
    pil_img = Image.fromarray(rgb)

    # 构造经纬度网格
    x = ds['x'].values
    y = ds['y'].values
    lon, lat = np.meshgrid(x, y)

    return pil_img, lon, lat
