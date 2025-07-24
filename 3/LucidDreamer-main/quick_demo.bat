@echo off
echo =====================================
echo LucidDreamer 快速演示脚本
echo =====================================
echo.

REM 检查虚拟环境是否存在
if not exist "lucid_dreamer_env\Scripts\activate.bat" (
    echo [错误] 虚拟环境不存在！
    echo 请先运行 setup_luciddreamer_windows.bat 来配置环境
    pause
    exit /b 1
)

echo [信息] 激活虚拟环境...
call lucid_dreamer_env\Scripts\activate.bat

echo [信息] 可用的演示示例:
echo 1. Christmas 圣诞客厅场景 (christmas.png)
echo 2. Cabin 魔法师小屋场景 (cabin.png)  
echo 3. Doge 温馨客厅场景 (doge.png)
echo 4. Web界面演示 (app_mini.py)
echo.

set /p choice="请选择要运行的演示 (1-4): "

if not exist "outputs" mkdir outputs

if "%choice%"=="1" (
    echo [信息] 运行Christmas场景演示...
    python run.py --image examples/christmas.png --text examples/christmas.txt --save_dir outputs/christmas_demo --seed 42 --diff_steps 20
) else if "%choice%"=="2" (
    echo [信息] 运行Cabin场景演示...
    python run.py --image examples/cabin.png --text examples/cabin.txt --save_dir outputs/cabin_demo --seed 42 --diff_steps 20
) else if "%choice%"=="3" (
    echo [信息] 运行Doge场景演示...
    python run.py --image examples/doge.png --text examples/doge.txt --save_dir outputs/doge_demo --seed 42 --diff_steps 20
) else if "%choice%"=="4" (
    echo [信息] 启动Web界面演示...
    echo 启动后请在浏览器中访问: http://localhost:7860
    python app_mini.py
) else (
    echo [错误] 无效选择！
    pause
    exit /b 1
)

if %errorlevel% neq 0 (
    echo [错误] 演示运行失败！
) else (
    echo [成功] 演示完成！
    if "%choice%" neq "4" (
        echo 输出文件保存在: outputs/ 目录下
        echo 使用Super-Splat查看器查看.ply文件: https://playcanvas.com/super-splat
    )
)

echo.
pause 