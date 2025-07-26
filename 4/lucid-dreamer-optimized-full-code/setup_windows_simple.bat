@echo off
chcp 65001 >nul
echo ==========================================================
echo LucidDreamer 简化安装脚本 (适用于已有基础环境)
echo ==========================================================

:: 激活现有环境或创建新环境
if exist "lucid_optimized_env" (
    echo [INFO] 激活现有虚拟环境...
    call lucid_optimized_env\Scripts\activate.bat
) else (
    echo [INFO] 创建新虚拟环境...
    python -m venv lucid_optimized_env
    call lucid_optimized_env\Scripts\activate.bat
)

:: 更新 pip
echo [INFO] 更新 pip...
python -m pip install --upgrade pip

:: 一键安装所有 Python 依赖
echo [INFO] 安装所有依赖包...
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.23.5 opencv-python==4.8.0.74 Pillow==9.5.0 scipy==1.10.1 tqdm==4.65.0 PyYAML==6.0 plyfile==0.8.1
pip install basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0
pip install huggingface-hub==0.14.1 timm==0.9.2 torchmetrics==0.11.4
pip install peft diffusers imageio[ffmpeg] open3d gradio omegaconf

:: 编译 CUDA 扩展
echo [INFO] 编译 CUDA 扩展...

:: GLM 库准备
cd submodules\depth-diff-gaussian-rasterization-min\third_party
if not exist "glm" (
    git clone https://github.com/g-truc/glm.git
)

:: 编译模块
cd ..\
python setup.py install

cd ..\simple-knn
python setup.py install

cd ..\..

:: 创建目录
mkdir inputs outputs logs pretrained 2>nul

echo [INFO] 安装完成！使用 start_lucid_dreamer.bat 启动程序
pause 