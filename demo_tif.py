import os
from utils.onnx_model_loader import ONNXModelLoader
from utils.patch_inference import PatchInferenceEngine
from utils.visualizer import PredictionVisualizer
from utils.tiff_handler import load_and_enhance_tiff

# 只需改这两个常量
TIF_FILE      = "S1_example.tif"
OUTPUT_FOLDER = "outputs/example"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1. 读取并增强 TIFF（仅传入文件名）
pil_img, lon, lat = load_and_enhance_tiff(TIF_FILE)

# 2. 加载 ONNX 模型（自动补全 .onnx）
loader  = ONNXModelLoader("GENERAL")
session = loader.load_model()

# 3. 切片推理得到二值掩码
engine    = PatchInferenceEngine(remove_small_noises=True)
pred_mask = engine.predict_image(pil_img, session)

# 4. 多模式保存
viz = PredictionVisualizer(OUTPUT_FOLDER, default_filename=os.path.basename(TIF_FILE).replace(".tif", ".png"))
viz.save(pred_mask, mode="prediction")
viz.save(pred_mask, mode="comparison", input_image=pil_img)
viz.save(pred_mask, mode="overlay",    input_image=pil_img, overlay_alpha=0.5)

# 5. 导出经纬度 TXT
txt_file = os.path.join(OUTPUT_FOLDER, "iw_locations.txt")
with open(txt_file, "w") as f:
    rows, cols = (pred_mask > 128).nonzero()
    for r, c in zip(rows, cols):
        f.write(f"{lon[r,c]:.6f},{lat[r,c]:.6f}\n")
print(f"[INFO] Locations TXT saved → {txt_file}")
