@echo off
echo =====================================
echo GLM库依赖修复脚本
echo =====================================
echo.

echo [信息] 检测到编译错误：缺少glm/glm.hpp
echo [信息] 正在下载并设置GLM库...
echo.

cd submodules\depth-diff-gaussian-rasterization-min\third_party

echo [步骤1] 下载GLM库...
if exist "glm" (
    echo [信息] GLM目录已存在，删除旧版本...
    rmdir /s /q glm
)

echo [信息] 使用PowerShell下载GLM库...
powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/g-truc/glm/archive/refs/tags/0.9.9.8.zip' -OutFile 'glm.zip'}"

if not exist "glm.zip" (
    echo [错误] GLM下载失败，尝试备用方法...
    echo [信息] 尝试使用curl下载...
    curl -L -o glm.zip https://github.com/g-truc/glm/archive/refs/tags/0.9.9.8.zip
    
    if not exist "glm.zip" (
        echo [错误] 所有下载方法都失败了
        echo [信息] 请手动下载GLM库：
        echo 1. 访问: https://github.com/g-truc/glm/releases
        echo 2. 下载最新版本的zip文件
        echo 3. 解压到: submodules\depth-diff-gaussian-rasterization-min\third_party\glm\
        echo 4. 确保存在文件: submodules\depth-diff-gaussian-rasterization-min\third_party\glm\glm\glm.hpp
        pause
        exit /b 1
    )
)

echo [步骤2] 解压GLM库...
powershell -Command "& {Expand-Archive -Path 'glm.zip' -DestinationPath '.' -Force}"

if exist "glm-0.9.9.8" (
    echo [信息] 重命名GLM目录...
    move glm-0.9.9.8 glm
) else (
    echo [错误] GLM解压失败
    if exist "glm.zip" del glm.zip
    pause
    exit /b 1
)

echo [步骤3] 清理下载文件...
if exist "glm.zip" del glm.zip

echo [步骤4] 验证GLM安装...
if exist "glm\glm\glm.hpp" (
    echo [成功] ✅ GLM库安装成功！
    echo [信息] GLM头文件位置: third_party\glm\glm\glm.hpp
) else (
    echo [错误] GLM头文件不存在
    echo [调试] 检查目录结构:
    dir glm /s /b | findstr "glm.hpp"
    pause
    exit /b 1
)

cd ..\..\..

echo.
echo =====================================
echo 重新编译CUDA扩展
echo =====================================
echo.

echo [信息] 激活虚拟环境...
call lucid_dreamer_env\Scripts\activate.bat

echo [信息] 重新编译 depth-diff-gaussian-rasterization-min...
cd submodules\depth-diff-gaussian-rasterization-min

REM 清理之前的编译文件
if exist "build" rmdir /s /q build
if exist "depth_diff_gaussian_rasterization_min.egg-info" rmdir /s /q depth_diff_gaussian_rasterization_min.egg-info

python setup.py install
if %errorlevel% neq 0 (
    echo [错误] rasterization模块编译仍然失败！
    echo [调试信息] 请检查：
    echo 1. CUDA是否正确安装 (nvcc --version)
    echo 2. Visual Studio Build Tools是否安装
    echo 3. GLM库路径是否正确
    cd ..\..
    pause
    exit /b 1
)

echo [信息] 编译 simple-knn...
cd ..\simple-knn

REM 清理之前的编译文件
if exist "build" rmdir /s /q build
if exist "simple_knn.egg-info" rmdir /s /q simple_knn.egg-info

python setup.py install
if %errorlevel% neq 0 (
    echo [错误] simple-knn模块编译失败！
    cd ..\..
    pause
    exit /b 1
)

cd ..\..

echo.
echo =====================================
echo 🎉 GLM依赖修复完成！
echo =====================================
echo.

echo [信息] 测试模块导入...
python -c "from luciddreamer import LucidDreamer; print('✓ LucidDreamer导入成功')" 2>nul
if %errorlevel% equ 0 (
    echo [成功] ✅ 所有模块导入成功！
    echo.
    echo [信息] 运行快速测试...
    python run.py --image examples/christmas.png --text examples/christmas.txt --save_dir outputs/glm_fix_test --diff_steps 3
    
    if %errorlevel% equ 0 (
        echo.
        echo ✅ 测试运行成功！现在可以正常使用 quick_start_offline.bat 了
    else (
        echo [警告] 测试运行有问题，但模块导入成功
    )
) else (
    echo [错误] 模块导入失败，可能还有其他问题
)

echo.
pause 