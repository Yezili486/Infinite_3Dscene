@echo off
echo =====================================
echo LucidDreamer 导入问题诊断
echo =====================================
echo.

REM 激活虚拟环境
if exist "lucid_dreamer_env\Scripts\activate.bat" (
    call lucid_dreamer_env\Scripts\activate.bat
    echo [信息] 虚拟环境已激活
) else (
    echo [错误] 虚拟环境不存在
    pause
    exit /b 1
)

echo [步骤1] 检查关键包版本:
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import huggingface_hub; print(f'HuggingFace Hub: {huggingface_hub.__version__}')"

echo.
echo [步骤2] 测试CUDA扩展模块:
python -c "try: import depth_diff_gaussian_rasterization_min; print('✓ depth_diff_gaussian_rasterization_min 导入成功'); except Exception as e: print(f'✗ depth_diff_gaussian_rasterization_min 错误: {e}')"
python -c "try: import simple_knn; print('✓ simple_knn 导入成功'); except Exception as e: print(f'✗ simple_knn 错误: {e}')"

echo.
echo [步骤3] 逐步测试导入链:
python -c "try: from diffusers import StableDiffusionInpaintPipeline; print('✓ StableDiffusionInpaintPipeline 导入成功'); except Exception as e: print(f'✗ StableDiffusionInpaintPipeline 错误: {e}')"
python -c "try: from transformers import CLIPTextModel; print('✓ CLIPTextModel 导入成功'); except Exception as e: print(f'✗ CLIPTextModel 错误: {e}')"

echo.
echo [步骤4] 详细测试LucidDreamer导入:
python -c "
try:
    print('正在导入 luciddreamer...')
    from luciddreamer import LucidDreamer
    print('✅ LucidDreamer 导入成功！')
except Exception as e:
    print(f'❌ LucidDreamer 导入失败: {e}')
    import traceback
    traceback.print_exc()
"

echo.
echo [步骤5] 如果仍有问题，尝试手动修复:
set /p manual_fix="是否尝试手动修复其他可能的版本冲突? (y/n): "
if /i "%manual_fix%"=="y" (
    echo.
    echo [修复1] 降级可能冲突的包...
    pip install "accelerate==0.20.3" "peft==0.4.0"
    
    echo [修复2] 确保torch版本正确...
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
    
    echo [修复3] 重新测试...
    python -c "
try:
    from luciddreamer import LucidDreamer
    print('🎉 修复后导入成功！')
except Exception as e:
    print(f'仍然失败: {e}')
    print('建议检查错误详情并尝试重建环境')
"
)

echo.
echo [步骤6] 最终建议:
echo 如果LucidDreamer仍无法导入，可能的解决方案：
echo 1. 重新运行: manual_fix_environment.bat（完全重建环境）
echo 2. 检查CUDA环境: fix_cuda_environment.bat
echo 3. 使用在线版本: https://huggingface.co/spaces/ironjr/LucidDreamer-mini
echo.

pause 