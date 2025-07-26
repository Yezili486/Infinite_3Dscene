@echo off
echo =====================================
echo 手动重建兼容环境脚本
echo =====================================
echo.

echo [信息] 这个脚本将创建一个全新的兼容环境
echo [警告] 这将删除现有的 lucid_dreamer_env 环境
echo.

set /p confirm="确定要继续吗? (y/n): "
if /i not "%confirm%"=="y" (
    echo 操作已取消
    pause
    exit /b 0
)

echo.
echo [信息] 1. 删除旧环境...
if exist "lucid_dreamer_env" (
    rmdir /s /q lucid_dreamer_env
    echo [完成] 旧环境已删除
)

echo.
echo [信息] 2. 创建新的虚拟环境...
python -m venv lucid_dreamer_env
if %errorlevel% neq 0 (
    echo [错误] 虚拟环境创建失败！
    pause
    exit /b 1
)

echo [信息] 3. 激活虚拟环境...
call lucid_dreamer_env\Scripts\activate.bat

echo.
echo [信息] 4. 更新pip...
python -m pip install --upgrade pip

echo.
echo [信息] 5. 按照兼容顺序安装包...

echo [步骤1] 安装NumPy 1.26.4 (兼容版本)
pip install "numpy==1.26.4"

echo [步骤2] 安装PyTorch及相关包
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo [步骤3] 安装scipy (兼容NumPy 1.26.4)
pip install "scipy==1.11.4"

echo [步骤4] 安装opencv-python (兼容版本)
pip install "opencv-python==4.8.1.78"

echo [步骤5] 安装diffusers (兼容PyTorch 2.0.1)
pip install "diffusers==0.21.4"

echo [步骤6] 安装transformers
pip install "transformers==4.30.2"

echo [步骤7] 安装其他必需包
pip install "accelerate==0.21.0"
pip install "peft==0.6.2"
pip install "timm==0.6.7"
pip install "plyfile==0.8.1"
pip install "Pillow<11.0"
pip install "imageio[ffmpeg]"
pip install "open3d"
pip install "gradio"
pip install "omegaconf"

echo.
echo [信息] 6. 重新编译CUDA扩展...

echo [编译1] depth-diff-gaussian-rasterization-min
cd submodules\depth-diff-gaussian-rasterization-min
python setup.py install
if %errorlevel% neq 0 (
    echo [错误] rasterization模块编译失败！
    cd ..\..
    pause
    exit /b 1
)

echo [编译2] simple-knn
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
echo [信息] 7. 验证安装...
echo 检查关键包版本:
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"

echo.
echo [信息] 8. 测试模块导入...
python -c "from luciddreamer import LucidDreamer; print('✓ LucidDreamer导入成功')" 2>nul
if %errorlevel% equ 0 (
    echo [成功] ✅ 所有模块导入成功！
    echo.
    echo [信息] 9. 运行测试...
    python run.py --image examples/christmas.png --text examples/christmas.txt --save_dir outputs/manual_fix_test --diff_steps 5
    
    if %errorlevel% equ 0 (
        echo.
        echo =====================================
        echo 🎉 环境重建完成！
        echo =====================================
        echo.
        echo ✅ 所有依赖包版本兼容
        echo ✅ CUDA扩展编译成功
        echo ✅ 测试运行通过
        echo.
        echo 现在可以正常使用 quick_start_offline.bat 了
    else (
        echo [警告] 测试运行失败，可能还有其他问题
    )
) else (
    echo [错误] 模块导入失败，请检查安装
)

echo.
pause 