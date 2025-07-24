@echo off
echo =====================================
echo LucidDreamer Windows 环境配置脚本
echo =====================================
echo.

REM 检查是否有Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python！请先安装Python 3.9并确保添加到PATH
    echo 下载地址: https://www.python.org/downloads/windows/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo [信息] 检测到Python版本:
python --version

REM 检查CUDA
echo [信息] 检查CUDA环境...
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 未检测到CUDA，GPU加速可能不可用
) else (
    echo [信息] 检测到CUDA:
    nvcc --version | findstr "release"
)

echo.
echo =====================================
echo 步骤 1: 创建虚拟环境
echo =====================================

REM 删除旧的虚拟环境（如果存在）
if exist "lucid_dreamer_env" (
    echo [信息] 删除旧的虚拟环境...
    rmdir /s /q lucid_dreamer_env
)

echo [信息] 创建新的虚拟环境...
python -m venv lucid_dreamer_env
if %errorlevel% neq 0 (
    echo [错误] 虚拟环境创建失败！
    pause
    exit /b 1
)

echo [信息] 激活虚拟环境...
call lucid_dreamer_env\Scripts\activate.bat

echo.
echo =====================================
echo 步骤 2: 更新pip并安装PyTorch
echo =====================================

echo [信息] 更新pip...
python -m pip install --upgrade pip

echo [信息] 安装PyTorch (CUDA 11.8版本)...
REM 根据README推荐安装CUDA 11.8版本的PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo.
echo =====================================
echo 步骤 3: 安装项目依赖
echo =====================================

echo [信息] 安装Python依赖包...
pip install peft diffusers scipy numpy imageio[ffmpeg] opencv-python Pillow open3d gradio omegaconf
pip install timm==0.6.7
pip install plyfile==0.8.1

echo.
echo =====================================
echo 步骤 4: 编译CUDA扩展模块
echo =====================================

echo [信息] 编译depth-diff-gaussian-rasterization-min...
cd submodules\depth-diff-gaussian-rasterization-min

REM 检查是否存在third_party/glm目录
if not exist "third_party\glm" (
    echo [信息] 下载GLM库...
    cd third_party
    git clone https://github.com/g-truc/glm.git
    if %errorlevel% neq 0 (
        echo [警告] GLM下载失败，尝试继续...
    )
    cd ..
)

echo [信息] 编译rasterization模块...
python setup.py install
if %errorlevel% neq 0 (
    echo [错误] rasterization模块编译失败！
    echo 请确保已安装Visual Studio Build Tools
    cd ..\..
    pause
    exit /b 1
)

echo [信息] 编译simple-knn模块...
cd ..\simple-knn
python setup.py install
if %errorlevel% neq 0 (
    echo [错误] simple-knn模块编译失败！
    cd ..\..
    pause
    exit /b 1
)

cd ..\..

echo.
echo =====================================
echo 步骤 5: 验证安装
echo =====================================

echo [信息] 验证PyTorch CUDA支持...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA设备数: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'CUDA不可用')"

echo [信息] 验证关键模块...
python -c "try: import depth_diff_gaussian_rasterization_min; print('✓ depth_diff_gaussian_rasterization_min 导入成功'); except: print('✗ depth_diff_gaussian_rasterization_min 导入失败')"
python -c "try: import simple_knn; print('✓ simple_knn 导入成功'); except: print('✗ simple_knn 导入失败')"

echo.
echo =====================================
echo 🎉 环境配置完成！
echo =====================================
echo.

echo =====================================
echo 🚀 运行演示实例
echo =====================================
echo.

echo [信息] 运行第一个演示: Christmas场景生成
echo 使用图片: examples/christmas.png
echo 文本提示: "Cozy livingroom in christmas"
echo.

REM 创建输出目录
if not exist "outputs" mkdir outputs

echo [信息] 开始生成3D场景... (这可能需要几分钟时间)
echo 命令: python run.py --image examples/christmas.png --text examples/christmas.txt --save_dir outputs/christmas_demo

python run.py --image examples/christmas.png --text examples/christmas.txt --save_dir outputs/christmas_demo --seed 42 --diff_steps 20

if %errorlevel% neq 0 (
    echo [错误] 演示运行失败！
    echo 请检查CUDA环境和依赖安装是否正确
) else (
    echo.
    echo ✅ 演示完成！输出文件保存在: outputs/christmas_demo/
    echo.
    echo 📁 生成的文件:
    if exist "outputs\christmas_demo" (
        dir outputs\christmas_demo /b
    )
    echo.
    echo 🎯 你可以使用以下方式查看结果:
    echo 1. .ply文件: 使用Super-Splat查看器 (https://playcanvas.com/super-splat)
    echo 2. .mp4文件: 直接用视频播放器观看生成的3D视频
    echo.
    echo 🔧 运行其他示例:
    echo    python run.py --image examples/cabin.png --text examples/cabin.txt --save_dir outputs/cabin_demo
    echo    python run.py --image examples/doge.png --text examples/doge.txt --save_dir outputs/doge_demo
    echo.
    echo 🌐 启动Web界面:
    echo    python app_mini.py  (轻量版)
    echo    python app.py       (完整版，需要更多内存)
)

echo.
echo =====================================
echo 💡 使用提示
echo =====================================
echo 1. 每次使用前请激活虚拟环境: lucid_dreamer_env\Scripts\activate
echo 2. 如需GPU加速，请确保安装了CUDA 11.8+
echo 3. 第一次运行可能需要下载预训练模型
echo 4. 生成过程需要大量显存 (建议8GB+)
echo.

pause 