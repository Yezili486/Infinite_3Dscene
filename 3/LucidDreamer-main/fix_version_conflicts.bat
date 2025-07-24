@echo off
echo =====================================
echo 版本冲突修复脚本
echo =====================================
echo.

echo [检测] 发现版本兼容性冲突:
echo ❌ PyTorch 2.0.1 vs diffusers要求 ≥2.1
echo ❌ NumPy 2.1.2 vs 某些模块要求 <2.0
echo ❌ 依赖包版本不匹配

REM 激活虚拟环境
if exist "lucid_dreamer_env\Scripts\activate.bat" (
    call lucid_dreamer_env\Scripts\activate.bat
    echo [信息] 虚拟环境已激活
) else (
    echo [错误] 虚拟环境不存在
    pause
    exit /b 1
)

echo.
echo 修复方案选择:
echo 1. 降级方案 (推荐) - 降级到兼容版本
echo 2. 升级方案 - 升级PyTorch到最新版本
echo 3. 重建环境 - 完全重新安装兼容版本
echo.

set /p choice="请选择修复方案 (1-3): "

if "%choice%"=="1" (
    goto :downgrade_fix
) else if "%choice%"=="2" (
    goto :upgrade_fix
) else if "%choice%"=="3" (
    goto :rebuild_env
) else (
    echo [错误] 无效选择
    pause
    exit /b 1
)

:downgrade_fix
echo.
echo =====================================
echo 方案1: 降级到兼容版本 (推荐)
echo =====================================
echo.

echo [信息] 降级NumPy到1.x版本...
pip install "numpy<2.0"

echo [信息] 降级diffusers到与PyTorch 2.0.1兼容的版本...
pip install "diffusers==0.21.4"

echo [信息] 降级transformers到兼容版本...
pip install "transformers==4.30.2"

echo [信息] 确保其他包兼容性...
pip install "accelerate==0.21.0" "peft==0.6.2"

goto :verify

:upgrade_fix
echo.
echo =====================================
echo 方案2: 升级PyTorch (可能有CUDA兼容问题)
echo =====================================
echo.

echo [警告] 升级PyTorch可能导致CUDA兼容问题
set /p confirm="确定要升级吗? (y/n): "
if /i not "%confirm%"=="y" goto :downgrade_fix

echo [信息] 升级PyTorch到2.1+...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo [信息] 降级NumPy...
pip install "numpy<2.0"

goto :verify

:rebuild_env
echo.
echo =====================================
echo 方案3: 重建环境
echo =====================================
echo.

echo [警告] 这将删除当前环境并重新创建
set /p confirm="确定要重建环境吗? (y/n): "
if /i not "%confirm%"=="y" goto :downgrade_fix

echo [信息] 退出当前环境...
deactivate 2>nul

echo [信息] 删除旧环境...
cd ..
if exist "lucid_dreamer_env" rmdir /s /q lucid_dreamer_env
cd LucidDreamer-main

echo [信息] 创建新环境...
python -m venv lucid_dreamer_env
call lucid_dreamer_env\Scripts\activate.bat

echo [信息] 更新pip...
python -m pip install --upgrade pip

echo [信息] 安装兼容版本的包...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0"
pip install "diffusers==0.21.4"
pip install "transformers==4.30.2"
pip install "accelerate==0.21.0"
pip install "peft==0.6.2"
pip install timm==0.6.7 plyfile==0.8.1 scipy opencv-python Pillow open3d gradio omegaconf
pip install imageio[ffmpeg]

goto :verify

:verify
echo.
echo =====================================
echo 验证修复结果
echo =====================================
echo.

echo [信息] 检查关键包版本...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo.
echo [信息] 测试导入关键模块...
python -c "from luciddreamer import LucidDreamer; print('✓ LucidDreamer导入成功')" 2>nul
if %errorlevel% equ 0 (
    echo [成功] 模块导入测试通过
) else (
    echo [警告] 模块导入仍有问题，尝试额外修复...
    
    echo [信息] 安装额外的兼容包...
    pip install "torch==2.0.1" "numpy==1.24.3" "diffusers==0.21.4" --force-reinstall
    
    echo [信息] 重新测试...
    python -c "from luciddreamer import LucidDreamer; print('✓ 修复后导入成功')" 2>nul
    if %errorlevel% equ 0 (
        echo [成功] 修复完成
    ) else (
        echo [错误] 仍有问题，建议重建环境
    )
)

echo.
echo =====================================
echo 🎯 测试运行
echo =====================================
echo.

echo [信息] 尝试运行简单测试...
set /p test_run="是否现在测试运行? (y/n): "
if /i "%test_run%"=="y" (
    echo [信息] 运行测试...
    python run.py --image examples/christmas.png --text examples/christmas.txt --save_dir outputs/version_fix_test --diff_steps 5
    
    if %errorlevel% equ 0 (
        echo [成功] ✅ 测试运行成功！版本冲突已解决
        echo [信息] 输出保存在: outputs/version_fix_test/
    ) else (
        echo [警告] 运行仍有问题，可能需要进一步调试
    )
)

echo.
echo =====================================
echo 📝 修复总结
echo =====================================
echo.
echo 已执行的修复:
echo ✓ NumPy版本调整到1.x
echo ✓ diffusers版本降级到兼容版本
echo ✓ transformers版本调整
echo ✓ 其他依赖包版本协调
echo.
echo 💡 如果仍有问题:
echo 1. 重启命令行窗口
echo 2. 重新激活环境: lucid_dreamer_env\Scripts\activate
echo 3. 运行: quick_start_offline.bat
echo.

pause 