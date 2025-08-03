#!/usr/bin/env python3
"""
可视化Close-up GS处理结果
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

def visualize_results(results_dir="./results"):
    """可视化处理结果"""
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Close-up GS 处理结果', fontsize=16)
    
    # 1. 原始输入图像
    if os.path.exists("sample_input.jpg"):
        img_original = Image.open("sample_input.jpg")
        axes[0, 0].imshow(img_original)
        axes[0, 0].set_title('原始输入图像')
        axes[0, 0].axis('off')
    
    # 2. 增强后的图像
    enhanced_path = os.path.join(results_dir, "enhanced_image.jpg")
    if os.path.exists(enhanced_path):
        img_enhanced = Image.open(enhanced_path)
        axes[0, 1].imshow(img_enhanced)
        axes[0, 1].set_title('增强后的图像')
        axes[0, 1].axis('off')
    
    # 3. 深度图
    depth_path = os.path.join(results_dir, "depth_map.jpg")
    if os.path.exists(depth_path):
        img_depth = Image.open(depth_path)
        axes[1, 0].imshow(img_depth, cmap='gray')
        axes[1, 0].set_title('深度图')
        axes[1, 0].axis('off')
    
    # 4. 点云可视化（2D投影）
    pc_path = os.path.join(results_dir, "enhanced_point_cloud.npy")
    if os.path.exists(pc_path):
        point_cloud = np.load(pc_path)
        if len(point_cloud) > 0:
            # 取前1000个点进行可视化
            sample_points = point_cloud[:1000] if len(point_cloud) > 1000 else point_cloud
            # 确保颜色值在0-1范围内
            colors = np.clip(sample_points[:, 3:6], 0, 1)
            axes[1, 1].scatter(sample_points[:, 0], sample_points[:, 1], 
                              c=colors, s=1, alpha=0.6)
            axes[1, 1].set_title('增强点云 (2D投影)')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')
            axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "visualization.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*50)
    print("处理结果统计")
    print("="*50)
    
    if os.path.exists(pc_path):
        point_cloud = np.load(pc_path)
        print(f"点云点数: {len(point_cloud)}")
        print(f"点云范围:")
        print(f"  X: [{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}]")
        print(f"  Y: [{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]")
        print(f"  Z: [{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}]")
    
    # 显示相机参数
    camera_path = os.path.join(results_dir, "cameras.json")
    if os.path.exists(camera_path):
        with open(camera_path, 'r') as f:
            cameras = json.load(f)
        print(f"相机数量: {len(cameras['cameras'])}")
        for i, cam in enumerate(cameras['cameras']):
            print(f"  相机 {i}: {cam['width']}x{cam['height']}")
    
    print("="*50)

def create_comparison_report(results_dir="./results"):
    """创建对比报告"""
    report_path = os.path.join(results_dir, "comparison_report.html")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Close-up GS 处理结果对比</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .image-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
            .image-item { text-align: center; }
            .image-item img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            .stats { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Close-up GS 处理结果对比</h1>
            
            <div class="image-grid">
                <div class="image-item">
                    <h3>原始输入</h3>
                    <img src="../sample_input.jpg" alt="原始输入">
                </div>
                <div class="image-item">
                    <h3>增强图像</h3>
                    <img src="enhanced_image.jpg" alt="增强图像">
                </div>
                <div class="image-item">
                    <h3>深度图</h3>
                    <img src="depth_map.jpg" alt="深度图">
                </div>
                <div class="image-item">
                    <h3>可视化结果</h3>
                    <img src="visualization.png" alt="可视化结果">
                </div>
            </div>
            
            <div class="stats">
                <h3>处理统计</h3>
                <p>✓ 图像增强完成</p>
                <p>✓ 深度估计完成</p>
                <p>✓ 点云生成完成</p>
                <p>✓ Closeup GS增强完成</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"对比报告已保存: {report_path}")

if __name__ == "__main__":
    # 可视化结果
    visualize_results()
    
    # 创建对比报告
    create_comparison_report()
    
    print("\n可视化完成！") 