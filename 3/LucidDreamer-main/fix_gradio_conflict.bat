@echo off
echo =====================================
echo Gradio 版本冲突修复脚本
echo =====================================
echo.

echo [问题分析]
echo - gradio 5.38.2 需要 huggingface_hub >= 0.28.1
echo - diffusers 0.21.4 需要 huggingface_hub <= 0.16.4
echo - 这是一个无法调和的依赖冲突
echo.

call lucid_dreamer_env\Scripts\activate.bat

echo [解决方案1] 降级gradio到兼容版本...
echo 尝试安装gradio的旧版本，与huggingface_hub 0.16.4兼容

pip install "gradio==3.50.2"

if %errorlevel% neq 0 (
    echo [方案1失败] 尝试更激进的降级...
    pip install "gradio==3.40.1"
)

echo.
echo [验证修复] 测试导入...
python -c "import gradio; print('✓ gradio版本:', gradio.__version__)"
python -c "from luciddreamer import LucidDreamer; print('✅ LucidDreamer导入成功！')"

if %errorlevel% equ 0 (
    echo.
    echo 🎉 修复成功！现在可以运行 quick_start_offline.bat
) else (
    echo.
    echo [解决方案2] 完全移除gradio依赖...
    echo 某些功能可能受限，但核心功能应该可用
    
    pip uninstall gradio -y
    
    echo 修改luciddreamer.py，注释掉gradio导入...
    python -c "
import re
with open('luciddreamer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 注释掉gradio导入行
content = re.sub(r'^import gradio as gr', '# import gradio as gr', content, flags=re.MULTILINE)

with open('luciddreamer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✓ 已注释掉gradio导入')
"
    
    echo 测试修改后的导入...
    python -c "from luciddreamer import LucidDreamer; print('✅ 无gradio版本导入成功！')"
    
    if %errorlevel% equ 0 (
        echo.
        echo 🎉 核心功能修复成功！
        echo ⚠️ Web界面功能可能不可用，但3D生成应该正常
    else (
        echo ❌ 仍有其他问题
    )
)

echo.
pause 