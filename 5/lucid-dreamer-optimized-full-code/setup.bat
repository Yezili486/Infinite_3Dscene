@echo off
REM LucidDreamer优化版环境配置脚本 - Windows版本

echo 正在检查Python安装...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python未安装，请先安装Python 3.9+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo 正在检查pip安装...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip未安装，请重新安装Python并确保勾选"Add Python to PATH"选项
    pause
    exit /b 1
)

echo 创建虚拟环境...
python -m venv lucid_optimized_env

echo 激活虚拟环境...
call lucid_optimized_env\Scripts\activate.bat

echo 更新pip...
python -m pip install --upgrade pip

echo 安装PyTorch (CUDA 11.8)...
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

echo 安装项目依赖...
pip install -r requirements.txt

echo 创建项目目录结构...
if not exist "inputs" mkdir inputs
if not exist "outputs" mkdir outputs
if not exist "logs" mkdir logs
if not exist "pretrained" mkdir pretrained

echo.
echo 请手动下载以下预训练模型并放入pretrained目录：
echo 1. ESRGAN模型: https://github.com/xinntao/ESRGAN/releases
echo 2. ZoeDepth模型: https://github.com/isl-org/ZoeDepth
echo 3. 3DGS基础模型: 参考LucidDreamer官方仓库
echo.
echo 环境配置完成！
echo.
echo 使用以下命令激活环境：
echo   lucid_optimized_env\Scripts\activate.bat
echo.
echo 运行程序：
echo   python run_optimized.py
echo.
pause 