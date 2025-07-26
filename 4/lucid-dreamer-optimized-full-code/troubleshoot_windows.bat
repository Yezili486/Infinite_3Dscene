@echo off
chcp 65001 >nul
echo ==========================================================
echo LucidDreamer Windows 故障排除脚本
echo ==========================================================

echo [INFO] 检查系统环境...

:: 检查 Python 版本
echo.
echo --- Python 检查 ---
python --version
python -c "import sys; print(f'Python 路径: {sys.executable}')"

:: 检查 CUDA
echo.
echo --- CUDA 检查 ---
nvcc --version 2>nul
if %errorLevel% neq 0 (
    echo [ERROR] CUDA 未正确安装或不在 PATH 中
) else (
    echo [OK] CUDA 可用
)

:: 检查 Visual Studio
echo.
echo --- Visual Studio Build Tools 检查 ---
where cl 2>nul
if %errorLevel% neq 0 (
    echo [ERROR] Visual Studio Build Tools 未找到
) else (
    echo [OK] Visual Studio Build Tools 可用
)

:: 检查 GPU
echo.
echo --- GPU 检查 ---
if exist "lucid_optimized_env\Scripts\activate.bat" (
    call lucid_optimized_env\Scripts\activate.bat
    python -c "import torch; print(f'PyTorch CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU 数量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
) else (
    echo [WARNING] 虚拟环境不存在，无法检查 PyTorch
)

:: 检查关键模块
echo.
echo --- 关键模块检查 ---
if exist "lucid_optimized_env\Scripts\activate.bat" (
    call lucid_optimized_env\Scripts\activate.bat
    
    echo 检查 depth_diff_gaussian_rasterization_min...
    python -c "import depth_diff_gaussian_rasterization_min; print('✓ depth_diff_gaussian_rasterization_min 正常')" 2>nul
    if %errorLevel% neq 0 echo [ERROR] depth_diff_gaussian_rasterization_min 模块问题
    
    echo 检查 simple_knn...
    python -c "import simple_knn; print('✓ simple_knn 正常')" 2>nul
    if %errorLevel% neq 0 echo [ERROR] simple_knn 模块问题
    
    echo 检查其他核心模块...
    python -c "import torch, numpy, cv2, PIL; print('✓ 核心模块正常')" 2>nul
    if %errorLevel% neq 0 echo [ERROR] 核心模块问题
)

:: 检查目录结构
echo.
echo --- 目录结构检查 ---
if exist "inputs" (echo ✓ inputs 目录存在) else (echo ✗ inputs 目录缺失)
if exist "outputs" (echo ✓ outputs 目录存在) else (echo ✗ outputs 目录缺失)
if exist "pretrained" (echo ✓ pretrained 目录存在) else (echo ✗ pretrained 目录缺失)
if exist "submodules\depth-diff-gaussian-rasterization-min\third_party\glm" (echo ✓ GLM 库存在) else (echo ✗ GLM 库缺失)

echo.
echo ==========================================================
echo 常见问题解决方案
echo ==========================================================
echo.
echo 1. CUDA 编译错误:
echo    - 确保 CUDA 11.8 已安装
echo    - 确保 Visual Studio 2019/2022 Build Tools 已安装
echo    - 重新运行 setup_windows.bat
echo.
echo 2. 内存不足:
echo    - 关闭其他占用 GPU 的程序
echo    - 降低渲染分辨率
echo    - 使用较小的输入图像
echo.
echo 3. 模块导入错误:
echo    - 重新编译 CUDA 扩展: cd submodules\xxx ^&^& python setup.py install
echo    - 检查虚拟环境是否正确激活
echo.
echo 4. 下载错误:
echo    - 检查网络连接
echo    - 使用代理或镜像源
echo    - 手动下载模型文件
echo.

echo 如需重新安装，请运行: setup_windows.bat
echo.
pause 