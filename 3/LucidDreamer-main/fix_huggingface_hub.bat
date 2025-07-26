@echo off
echo =====================================
echo HuggingFace Hub 版本修复脚本
echo =====================================
echo.

echo [问题] huggingface_hub版本过新，与diffusers 0.21.4不兼容
echo [错误] cannot import name 'cached_download' from 'huggingface_hub'
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

echo [信息] 当前包版本:
python -c "import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')" 2>nul
python -c "import diffusers; print(f'diffusers: {diffusers.__version__}')" 2>nul

echo.
echo [修复] 降级huggingface_hub到兼容版本...
pip install "huggingface_hub==0.16.4"

if %errorlevel% neq 0 (
    echo [错误] huggingface_hub降级失败
    pause
    exit /b 1
)

echo.
echo [验证] 检查修复结果...
python -c "import huggingface_hub; print(f'✓ huggingface_hub: {huggingface_hub.__version__}')"
python -c "from huggingface_hub import cached_download; print('✓ cached_download导入成功')"

echo.
echo [测试] 尝试导入LucidDreamer...
python -c "from luciddreamer import LucidDreamer; print('✅ LucidDreamer导入成功！')" 2>nul

if %errorlevel% equ 0 (
    echo.
    echo =====================================
    echo 🎉 修复成功！
    echo =====================================
    echo.
    echo ✅ huggingface_hub版本已降级
    echo ✅ cached_download函数可用
    echo ✅ LucidDreamer模块导入正常
    echo.
    echo 现在可以重新运行: quick_start_offline.bat
) else (
    echo.
    echo [警告] LucidDreamer导入仍有问题，检查其他依赖...
    
    echo [额外修复] 安装其他可能需要的兼容版本...
    pip install "transformers==4.30.2" "tokenizers==0.13.3"
    
    echo [重新测试]
    python -c "from luciddreamer import LucidDreamer; print('✅ 额外修复后导入成功！')" 2>nul
    
    if %errorlevel% equ 0 (
        echo 🎉 完全修复成功！
    else (
        echo ❌ 仍有问题，可能需要更多调试
    )
)

echo.
pause 