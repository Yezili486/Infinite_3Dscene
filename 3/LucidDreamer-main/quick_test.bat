@echo off
echo =====================================
echo 快速环境测试
echo =====================================
echo.

REM 激活虚拟环境
call lucid_dreamer_env\Scripts\activate.bat

echo [测试1] 检查关键包版本:
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"

echo.
echo [测试2] 测试CUDA扩展:
python -c "import depth_diff_gaussian_rasterization_min; print('CUDA扩展1: OK')"
python -c "import simple_knn; print('CUDA扩展2: OK')"

echo.
echo [测试3] 测试LucidDreamer导入:
python -c "from luciddreamer import LucidDreamer; print('LucidDreamer: OK')"

echo.
echo 如果上面都显示OK，环境就正常了！
pause 