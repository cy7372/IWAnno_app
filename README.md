# IWAnno_app

IWAnno_app 是 IWAnno 模型的应用板块，旨在利用 ONNX 模型对卫星图像中的内波进行预测。项目实现了模型加载、图像切片预测与拼接，以及多种结果展示方式（如预测图、输入对比图和覆盖图）。预测结果以 NumPy 数组格式返回，便于后续数值分析与定量处理。

## 使用方法

1. **环境安装**  
   请运行以下命令安装项目所需依赖：
   
   ```bash
   pip install -r requirements.txt
   ```
2. **安装模型**  
   将.onnx模型文件放入models文件夹下

3. **运行示例**  
   项目根目录下的 $demo.py$ 文件展示了如何加载模型、对输入图像进行预测，以及展示和保存预测结果。请确保示例输入图像（例如 example_input.png）存在于项目根目录，或提供绝对路径。  
   运行示例：
   
   ```python
   python demo.py
   ```
