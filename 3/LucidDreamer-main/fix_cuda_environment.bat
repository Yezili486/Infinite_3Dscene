@echo off
echo =====================================
echo CUDA 环境修复脚本
echo =====================================
echo.

echo [信息] 正在检查CUDA安装...

REM 检查常见CUDA安装路径
set "CUDA_FOUND=0"

if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" (
    set "CUDA_VERSION=11.8"
    set "CUDA_PATH_FOUND=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    set "CUDA_FOUND=1"
    echo [找到] CUDA 11.8: %CUDA_PATH_FOUND%
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" (
    set "CUDA_VERSION=12.0"
    set "CUDA_PATH_FOUND=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
    set "CUDA_FOUND=1"
    echo [找到] CUDA 12.0: %CUDA_PATH_FOUND%
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7" (
    set "CUDA_VERSION=11.7"
    set "CUDA_PATH_FOUND=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
    set "CUDA_FOUND=1"
    echo [找到] CUDA 11.7: %CUDA_PATH_FOUND%
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" (
    set "CUDA_VERSION=12.1"
    set "CUDA_PATH_FOUND=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    set "CUDA_FOUND=1"
    echo [找到] CUDA 12.1: %CUDA_PATH_FOUND%
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6" (
    set "CUDA_VERSION=11.6"
    set "CUDA_PATH_FOUND=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"
    set "CUDA_FOUND=1"
    echo [找到] CUDA 11.6: %CUDA_PATH_FOUND%
)

if "%CUDA_FOUND%"=="0" (
    echo [错误] 未找到CUDA安装！
    echo.
    echo 请按照以下步骤安装CUDA:
    echo 1. 访问: https://developer.nvidia.com/cuda-11-8-0-download-archive
    echo 2. 选择: Windows ^> x86_64 ^> exe (local)
    echo 3. 下载并安装 CUDA 11.8
    echo 4. 重启计算机
    echo 5. 重新运行此脚本
    echo.
    pause
    exit /b 1
)

echo.
echo [信息] 设置当前会话的CUDA环境变量...
set "CUDA_HOME=%CUDA_PATH_FOUND%"
set "CUDA_PATH=%CUDA_PATH_FOUND%"
set "PATH=%CUDA_PATH_FOUND%\bin;%PATH%"

echo CUDA_HOME=%CUDA_HOME%
echo CUDA_PATH=%CUDA_PATH%

echo.
echo [信息] 验证CUDA命令...
nvcc --version
if %errorlevel% neq 0 (
    echo [错误] nvcc命令仍然无法使用
    echo 请检查CUDA安装是否完整
) else (
    echo [成功] nvcc命令可用
)

echo.
echo [信息] 创建永久环境变量设置批处理文件...

REM 创建set_cuda_env.bat文件
echo @echo off > set_cuda_env.bat
echo REM 设置CUDA环境变量 >> set_cuda_env.bat
echo set "CUDA_HOME=%CUDA_PATH_FOUND%" >> set_cuda_env.bat
echo set "CUDA_PATH=%CUDA_PATH_FOUND%" >> set_cuda_env.bat
echo set "PATH=%CUDA_PATH_FOUND%\bin;%%PATH%%" >> set_cuda_env.bat
echo echo [信息] CUDA环境变量已设置 >> set_cuda_env.bat

echo [信息] 已创建 set_cuda_env.bat 文件
echo 在每次使用前运行此文件来设置CUDA环境变量

echo.
echo [信息] 设置系统环境变量（需要管理员权限）...
echo 注意: 以下操作可能需要管理员权限

REM 尝试设置系统环境变量
setx CUDA_HOME "%CUDA_PATH_FOUND%" /M >nul 2>&1
if %errorlevel% equ 0 (
    echo [成功] 系统环境变量 CUDA_HOME 已设置
) else (
    echo [警告] 无法设置系统环境变量，可能需要管理员权限
)

setx CUDA_PATH "%CUDA_PATH_FOUND%" /M >nul 2>&1
if %errorlevel% equ 0 (
    echo [成功] 系统环境变量 CUDA_PATH 已设置
) else (
    echo [警告] 无法设置系统环境变量，可能需要管理员权限
)

echo.
echo =====================================
echo ✅ CUDA环境修复完成
echo =====================================
echo.
echo 📝 使用说明:
echo 1. 当前会话的CUDA环境已设置
echo 2. 如需永久设置，请以管理员身份重新运行此脚本
echo 3. 或者在每次使用前运行: set_cuda_env.bat
echo 4. 重启计算机后环境变量将生效
echo.

echo [信息] 验证当前PyTorch CUDA支持...
if exist "lucid_dreamer_env\Scripts\activate.bat" (
    call lucid_dreamer_env\Scripts\activate.bat
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA设备数: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'CUDA不可用')" 2>nul
    if %errorlevel% neq 0 (
        echo [信息] PyTorch环境未就绪，请先运行配置脚本
    )
) else (
    echo [信息] 虚拟环境未创建，请先运行配置脚本
)

echo.
pause 