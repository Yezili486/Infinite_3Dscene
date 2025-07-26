@echo off
echo =====================================
echo 最终修复脚本
echo =====================================
echo.

call lucid_dreamer_env\Scripts\activate.bat

echo [步骤1] 修复NumPy版本冲突...
pip install "numpy==1.26.4" --force-reinstall

echo [步骤2] 确保其他包兼容...
pip install "scipy==1.11.4" --force-reinstall

echo [步骤3] 重新编译CUDA扩展...
echo 编译 depth-diff-gaussian-rasterization-min...
cd submodules\depth-diff-gaussian-rasterization-min
if exist "build" rmdir /s /q build
if exist "*.egg-info" rmdir /s /q *.egg-info
python setup.py install

echo 编译 simple-knn...
cd ..\simple-knn
if exist "build" rmdir /s /q build  
if exist "*.egg-info" rmdir /s /q *.egg-info
python setup.py install

cd ..\..

echo [步骤4] 最终测试...
python -c "from luciddreamer import LucidDreamer; print('✅ 修复成功！')"

if %errorlevel% equ 0 (
    echo.
    echo 🎉 环境修复完成！现在可以运行 quick_start_offline.bat
) else (
    echo ❌ 仍有问题，建议完全重建环境
)

pause 