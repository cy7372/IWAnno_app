# demo.py

import os
from pathlib import Path
from PIL import Image

from utils.onnx_model_loader import ONNXModelLoader
from utils.patch_inference     import PatchInferenceEngine
from utils.visualizer          import PredictionVisualizer
from utils.image_loader        import load_image  # 新增这一行

if __name__ == '__main__':
    # -------------------- 配置区 --------------------
    INPUT_PATH    = "example_input.png"   # <--- 可以是 .tif/.tiff，也可以是 .png/.jpg
    MODEL_KEY     = "GENERAL"                # 对应 models/GENERAL.onnx
    OUTPUT_FOLDER = "outputs"                # 结果输出目录

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # -------------------- 1. 读取并处理输入 --------------------
    pil_img, lon_arr, lat_arr = load_image(INPUT_PATH)
    # 如果是 TIFF，lon_arr、lat_arr 就是 numpy.ndarray；否则都是 None

    # -------------------- 2. 加载 ONNX 模型 --------------------
    loader  = ONNXModelLoader(MODEL_KEY)
    session = loader.load_model()  # 会打印 "ONNX model loaded: GENERAL.onnx"

    # -------------------- 3. 切片推理得到二值掩码 --------------------
    engine    = PatchInferenceEngine(session, remove_small_noises=True)
    pred_mask = engine.predict_image(pil_img)
    # pred_mask: numpy.ndarray (H, W)，dtype=uint8，值为 0 或 255

    # -------------------- 4. 多模式保存 --------------------
    base_name = Path(INPUT_PATH).stem  # 去掉后缀的文件名
    viz = PredictionVisualizer(
        output_folder=OUTPUT_FOLDER,
        default_filename=f"{base_name}.png"
    )

    # 4.1 保存纯预测结果（二值掩码）
    viz.save(pred_mask, mode="prediction")
    print(f"[INFO] 已保存二值掩码 → {OUTPUT_FOLDER}/{base_name}.png")

    # 4.2 并排对比：左原图，右掩码
    viz.save(pred_mask, mode="comparison", input_image=pil_img)
    print(f"[INFO] 已保存并排对比 → {OUTPUT_FOLDER}/{base_name}_comparison.png")

    # 4.3 半透明覆盖：掩码覆盖在原图上
    viz.save(
        pred_mask,
        mode="overlay",
        input_image=pil_img,
        overlay_alpha=0.5
    )
    print(f"[INFO] 已保存叠加覆盖 → {OUTPUT_FOLDER}/{base_name}_overlay.png")

    # -------------------- 5. TIFF 时导出经纬度 TXT --------------------
    if lon_arr is not None and lat_arr is not None:
        txt_path = Path(OUTPUT_FOLDER) / f"{base_name}_locations.txt"
        with open(txt_path, "w") as f:
            rows, cols = (pred_mask > 128).nonzero()
            for r, c in zip(rows, cols):
                f.write(f"{lon_arr[r, c]:.6f},{lat_arr[r, c]:.6f}\n")
        print(f"[INFO] 已保存经纬度 → {txt_path}")

    print("[DONE] 全部处理完成。")
