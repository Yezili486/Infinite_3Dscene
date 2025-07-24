@echo off
echo =====================================
echo LucidDreamer Windows 环境配置脚本 (修复版)
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

REM 检查并设置CUDA环境变量
echo [信息] 检查CUDA环境...
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 未检测到CUDA，尝试查找CUDA安装路径...
    
    REM 尝试常见的CUDA安装路径
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" (
        set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
        echo [信息] 找到CUDA 11.8，设置环境变量
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" (
        set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
        echo [信息] 找到CUDA 12.0，设置环境变量
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7" (
        set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
        echo [信息] 找到CUDA 11.7，设置环境变量
    ) else (
        echo [警告] 未找到CUDA安装，将跳过GPU编译
        set "FORCE_CUDA=0"
        set "CUDA_HOME="
    )
) else (
    echo [信息] 检测到CUDA:
    nvcc --version | findstr "release"
    
    REM 尝试从nvcc路径推断CUDA_HOME
    for /f "tokens=*" %%i in ('where nvcc 2^>nul') do (
        set "nvcc_path=%%i"
        goto :found_nvcc
    )
    :found_nvcc
    if defined nvcc_path (
        for %%a in ("%nvcc_path%") do set "cuda_bin=%%~dpa"
        for %%a in ("%cuda_bin%..") do set "CUDA_HOME=%%~fa"
        echo [信息] 设置CUDA_HOME为: %CUDA_HOME%
    )
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
echo 步骤 4: 准备GLM库 (离线方式)
echo =====================================

echo [信息] 准备GLM库...
cd submodules\depth-diff-gaussian-rasterization-min

REM 检查是否存在third_party/glm目录
if not exist "third_party\glm" (
    echo [信息] 创建GLM目录结构...
    if not exist "third_party" mkdir third_party
    cd third_party
    
    echo [信息] 尝试下载GLM库...
    git clone https://github.com/g-truc/glm.git
    if %errorlevel% neq 0 (
        echo [警告] Git下载失败，尝试使用预编译头文件...
        
        REM 创建最小GLM结构
        if not exist "glm" mkdir glm
        cd glm
        if not exist "glm" mkdir glm
        
        REM 创建基本的GLM头文件 (最小版本)
        echo #ifndef GLM_GLM_HPP > glm\glm.hpp
        echo #define GLM_GLM_HPP >> glm\glm.hpp
        echo #include ^<cmath^> >> glm\glm.hpp
        echo namespace glm { >> glm\glm.hpp
        echo   typedef float vec3[3]; >> glm\glm.hpp
        echo   typedef float vec4[4]; >> glm\glm.hpp
        echo   typedef float mat3[9]; >> glm\glm.hpp
        echo   typedef float mat4[16]; >> glm\glm.hpp
        echo } >> glm\glm.hpp
        echo #endif >> glm\glm.hpp
        
        echo [信息] 创建了最小GLM头文件
        cd ..
    )
    cd ..
) else (
    echo [信息] GLM库已存在
)

echo.
echo =====================================
echo 步骤 5: 编译CUDA扩展模块
echo =====================================

echo [信息] 编译depth-diff-gaussian-rasterization-min...

REM 设置编译环境变量
if defined CUDA_HOME (
    set "PATH=%CUDA_HOME%\bin;%PATH%"
    echo [信息] 添加CUDA路径到PATH: %CUDA_HOME%\bin
)

echo [信息] 尝试编译rasterization模块...
python setup.py install
if %errorlevel% neq 0 (
    echo [警告] CUDA版本编译失败，尝试CPU版本...
    
    REM 尝试强制CPU编译
    set FORCE_CUDA=0
    set CUDA_VISIBLE_DEVICES=""
    python setup.py install
    
    if %errorlevel% neq 0 (
        echo [错误] 编译完全失败！
        echo.
        echo 可能的解决方案:
        echo 1. 安装Visual Studio Build Tools 2022
        echo 2. 确保CUDA正确安装并设置环境变量
        echo 3. 重启计算机后重试
        echo.
        echo 继续安装其他组件...
    ) else (
        echo [警告] 使用CPU版本编译成功（性能可能较慢）
    )
) else (
    echo [成功] CUDA版本编译成功
)

echo [信息] 编译simple-knn模块...
cd ..\simple-knn
python setup.py install
if %errorlevel% neq 0 (
    echo [警告] simple-knn编译失败，但可能不影响基本功能
)

cd ..\..

echo.
echo =====================================
echo 步骤 6: 验证安装
echo =====================================

echo [信息] 验证PyTorch CUDA支持...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA设备数: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'CUDA不可用，将使用CPU')"

echo [信息] 验证关键模块...
python -c "try: import depth_diff_gaussian_rasterization_min; print('✓ depth_diff_gaussian_rasterization_min 导入成功'); except Exception as e: print('✗ depth_diff_gaussian_rasterization_min 导入失败:', str(e))"
python -c "try: import simple_knn; print('✓ simple_knn 导入成功'); except Exception as e: print('✗ simple_knn 导入失败:', str(e))"

echo.
echo =====================================
echo 🎉 环境配置完成！
echo =====================================
echo.

echo =====================================
echo 🚀 运行演示实例 (仅CPU模式)
echo =====================================
echo.

echo [信息] 运行轻量级演示...
echo 注意: 如果CUDA模块编译失败，将使用CPU模式（速度较慢）

REM 创建输出目录
if not exist "outputs" mkdir outputs

echo [信息] 尝试运行Web界面演示（推荐）...
echo 启动app_mini.py，请在浏览器中访问 http://localhost:7860

start /b python app_mini.py

echo [信息] Web界面正在后台启动...
echo 如果无法访问，请检查防火墙设置
echo.

echo ✅ 配置基本完成！
echo.
echo 📝 重要提示:
echo 1. 如果CUDA编译失败，仍可使用CPU模式（较慢）
echo 2. Web界面: http://localhost:7860
echo 3. 命令行使用: python run.py --image examples/christmas.png --text examples/christmas.txt
echo 4. 每次使用前激活环境: lucid_dreamer_env\Scripts\activate
echo.

pause 