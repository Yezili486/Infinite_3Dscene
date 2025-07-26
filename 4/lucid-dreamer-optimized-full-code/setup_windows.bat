@echo off
chcp 65001 >nul
echo ==========================================================
echo LucidDreamer 优化版 Windows 环境配置脚本
echo ==========================================================

:: 设置错误处理
setlocal enabledelayedexpansion

:: 检查管理员权限
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] 管理员权限确认
) else (
    echo [ERROR] 请以管理员身份运行此脚本
    pause
    exit /b 1
)

:: 检查系统要求
echo [INFO] 检查系统要求...

:: 检查 CUDA
echo [INFO] 检查 CUDA 安装...
nvcc --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] 未检测到 CUDA 11.8，请先安装 CUDA 11.8
    echo [INFO] 下载地址: https://developer.nvidia.com/cuda-11-8-0-download-archive
    pause
    exit /b 1
) else (
    echo [OK] CUDA 已安装
)

:: 检查 Visual Studio Build Tools
echo [INFO] 检查 Visual Studio Build Tools...
where cl >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] 未检测到 Visual Studio Build Tools C++ 编译器
    echo [INFO] 请安装 Visual Studio 2019/2022 Build Tools 或完整版
    echo [INFO] 下载地址: https://visualstudio.microsoft.com/downloads/
    echo [INFO] 确保安装 "C++ build tools" 和 "MSVC v143 - VS 2022 C++ x64/x86 build tools"
    pause
    exit /b 1
) else (
    echo [OK] Visual Studio Build Tools 已安装
)

:: 检查 Python 3.9
echo [INFO] 检查 Python 安装...
python --version 2>nul | findstr "Python 3.9" >nul
if %errorLevel% neq 0 (
    echo [ERROR] 需要 Python 3.9，当前版本不符合要求
    echo [INFO] 请安装 Python 3.9.x
    echo [INFO] 下载地址: https://www.python.org/downloads/release/python-3916/
    pause
    exit /b 1
) else (
    echo [OK] Python 3.9 已安装
)

:: 检查 pip
echo [INFO] 检查 pip...
python -m pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] pip 未安装
    python -m ensurepip --upgrade
) else (
    echo [OK] pip 已安装
)

:: 创建虚拟环境
echo [INFO] 创建虚拟环境...
if exist "lucid_optimized_env" (
    echo [INFO] 删除现有虚拟环境...
    rmdir /s /q lucid_optimized_env
)

python -m venv lucid_optimized_env
if %errorLevel% neq 0 (
    echo [ERROR] 创建虚拟环境失败
    pause
    exit /b 1
)

:: 激活虚拟环境
echo [INFO] 激活虚拟环境...
call lucid_optimized_env\Scripts\activate.bat

:: 更新 pip
echo [INFO] 更新 pip...
python -m pip install --upgrade pip

:: 安装 PyTorch (CUDA 11.8)
echo [INFO] 安装 PyTorch CUDA 11.8...
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
if %errorLevel% neq 0 (
    echo [ERROR] PyTorch 安装失败
    pause
    exit /b 1
)

:: 安装基础依赖
echo [INFO] 安装基础依赖包...
pip install numpy==1.23.5
pip install opencv-python==4.8.0.74
pip install Pillow==9.5.0
pip install scipy==1.10.1
pip install tqdm==4.65.0
pip install PyYAML==6.0
pip install plyfile==0.8.1

:: 安装纹理增强相关依赖
echo [INFO] 安装纹理增强相关依赖...
pip install basicsr==1.4.2
pip install facexlib==0.3.0
pip install realesrgan==0.3.0

:: 安装深度估计相关依赖
echo [INFO] 安装深度估计相关依赖...
pip install huggingface-hub==0.14.1
pip install timm==0.9.2

:: 安装评估指标相关依赖
echo [INFO] 安装评估指标相关依赖...
pip install torchmetrics==0.11.4

:: 安装其他必要依赖
echo [INFO] 安装其他依赖...
pip install peft diffusers imageio[ffmpeg] open3d gradio omegaconf

:: 设置编译环境变量
echo [INFO] 设置编译环境变量...
set CUDA_HOME=%CUDA_PATH%
set TORCH_CUDA_ARCH_LIST=7.0;7.5;8.6
set NVCC_FLAGS=-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86

:: 准备 GLM 库 (depth-diff-gaussian-rasterization-min 需要)
echo [INFO] 准备 GLM 数学库...
cd submodules\depth-diff-gaussian-rasterization-min\third_party
if not exist "glm" (
    echo [INFO] 下载 GLM 库...
    git clone https://github.com/g-truc/glm.git
    if %errorLevel% neq 0 (
        echo [ERROR] GLM 库下载失败，请检查网络连接
        echo [INFO] 如果无法下载，请手动从 https://github.com/g-truc/glm 下载并解压到此目录
        pause
        exit /b 1
    )
) else (
    echo [OK] GLM 库已存在
)

:: 编译 depth-diff-gaussian-rasterization-min
echo [INFO] 编译 depth-diff-gaussian-rasterization-min...
cd ..
python setup.py build_ext --inplace
if %errorLevel% neq 0 (
    echo [WARNING] build_ext 失败，尝试 install...
)
python setup.py install
if %errorLevel% neq 0 (
    echo [ERROR] depth-diff-gaussian-rasterization-min 编译失败
    echo [INFO] 请检查 CUDA 和 Visual Studio Build Tools 是否正确安装
    pause
    exit /b 1
) else (
    echo [OK] depth-diff-gaussian-rasterization-min 编译成功
)

:: 编译 simple-knn
echo [INFO] 编译 simple-knn...
cd ..\simple-knn
python setup.py build_ext --inplace
if %errorLevel% neq 0 (
    echo [WARNING] build_ext 失败，尝试 install...
)
python setup.py install
if %errorLevel% neq 0 (
    echo [ERROR] simple-knn 编译失败
    pause
    exit /b 1
) else (
    echo [OK] simple-knn 编译成功
)

:: 返回项目根目录
cd ..\..

:: 创建必要的目录结构
echo [INFO] 创建项目目录结构...
mkdir inputs 2>nul
mkdir outputs 2>nul
mkdir logs 2>nul
mkdir pretrained 2>nul

:: 测试安装
echo [INFO] 测试安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}');"
if %errorLevel% neq 0 (
    echo [ERROR] PyTorch 测试失败
    pause
    exit /b 1
)

python -c "import depth_diff_gaussian_rasterization_min"
if %errorLevel% neq 0 (
    echo [ERROR] depth_diff_gaussian_rasterization_min 模块导入失败
    pause
    exit /b 1
)

python -c "import simple_knn"
if %errorLevel% neq 0 (
    echo [ERROR] simple_knn 模块导入失败
    pause
    exit /b 1
)

:: 下载预训练模型
echo [INFO] 准备下载预训练模型...
echo.
echo ==========================================================
echo 预训练模型下载指南
echo ==========================================================
echo.
echo 请手动下载以下预训练模型到 pretrained 目录：
echo.
echo 1. ZoeDepth 模型:
echo    - 自动下载（首次运行时会自动从 HuggingFace 下载）
echo.
echo 2. 稳定扩散模型（可选）:
echo    - 位置: pretrained/stable-diffusion/
echo    - 来源: https://huggingface.co/runwayml/stable-diffusion-v1-5
echo.
echo 3. RealESRGAN 模型（纹理增强）:
echo    - 位置: pretrained/RealESRGAN/
echo    - 下载: https://github.com/xinntao/Real-ESRGAN/releases
echo    - 文件: RealESRGAN_x4plus.pth
echo.

:: 创建启动脚本
echo [INFO] 创建启动脚本...
echo @echo off > start_lucid_dreamer.bat
echo call lucid_optimized_env\Scripts\activate.bat >> start_lucid_dreamer.bat
echo python run_optimized.py %%* >> start_lucid_dreamer.bat

echo @echo off > start_app.bat
echo call lucid_optimized_env\Scripts\activate.bat >> start_app.bat
echo python app.py >> start_app.bat

echo @echo off > start_mini_app.bat
echo call lucid_optimized_env\Scripts\activate.bat >> start_mini_app.bat
echo python app_mini.py >> start_mini_app.bat

echo.
echo ==========================================================
echo 安装完成！
echo ==========================================================
echo.
echo 使用说明：
echo 1. 运行主程序: start_lucid_dreamer.bat
echo 2. 运行 Web 应用: start_app.bat
echo 3. 运行简化版应用: start_mini_app.bat
echo.
echo 或者手动激活环境：
echo   lucid_optimized_env\Scripts\activate.bat
echo   python run_optimized.py
echo.
echo 注意事项：
echo 1. 首次运行可能需要下载额外的模型文件
echo 2. 确保有足够的 GPU 内存 (建议 8GB+)
echo 3. 如果遇到 CUDA 内存不足，可以降低渲染分辨率
echo.

pause 