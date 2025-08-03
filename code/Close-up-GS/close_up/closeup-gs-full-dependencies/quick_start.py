#!/usr/bin/env python3
"""
Close-up GS 一键启动脚本
自动完成环境检查、数据处理和结果可视化
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"正在执行: {description}")
    print(f"命令: {cmd}")
    print('='*50)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        print(f"✓ 完成! 耗时: {elapsed_time:.2f}秒")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def check_environment():
    """检查环境"""
    print("🔍 检查环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 11):
        print("⚠️  警告: 建议使用Python 3.11+")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA可用")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 Close-up GS 一键启动")
    print("="*60)
    
    # 1. 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请先安装依赖")
        return
    
    # 2. 测试环境
    if not run_command("python simple_test.py", "环境测试"):
        print("❌ 环境测试失败")
        return
    
    # 3. 运行项目
    if not run_command("python run_simplified.py --create_sample", "运行Close-up GS"):
        print("❌ 项目运行失败")
        return
    
    # 4. 可视化结果
    if not run_command("python visualize_results.py", "结果可视化"):
        print("❌ 可视化失败")
        return
    
    # 5. 显示结果
    print("\n🎉 项目运行完成!")
    print("="*60)
    print("生成的文件:")
    
    results_dir = "./results"
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            file_path = os.path.join(results_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  📄 {file} ({size:,} bytes)")
    
    print("\n📊 统计信息:")
    try:
        import numpy as np
        pc_path = os.path.join(results_dir, "enhanced_point_cloud.npy")
        if os.path.exists(pc_path):
            point_cloud = np.load(pc_path)
            print(f"  • 点云点数: {len(point_cloud):,}")
            print(f"  • 点云范围: X[{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}]")
            print(f"  • 点云范围: Y[{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]")
            print(f"  • 点云范围: Z[{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}]")
    except Exception as e:
        print(f"  • 无法读取点云统计: {e}")
    
    print("\n📁 结果目录:")
    print(f"  {os.path.abspath(results_dir)}")
    
    print("\n🔗 查看结果:")
    print("  • 可视化图像: results/visualization.png")
    print("  • 对比报告: results/comparison_report.html")
    print("  • 增强图像: results/enhanced_image.jpg")
    print("  • 深度图: results/depth_map.jpg")
    
    print("\n✨ 项目运行成功!")

if __name__ == "__main__":
    main() 