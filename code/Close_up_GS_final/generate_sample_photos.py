#!/usr/bin/env python3
"""
生成示例照片用于测试Close-up-GS模型
创建8张不同角度的示例照片
"""

import numpy as np
import cv2
from pathlib import Path
import math

def create_sample_photo(angle, size=(512, 512)):
    """创建一张示例照片"""
    
    # 创建空白图像
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # 计算中心点
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # 绘制一个3D立方体的2D投影（模拟物体）
    cube_size = 100
    cube_center = [center_x, center_y]
    
    # 根据角度计算立方体的旋转
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # 立方体的8个顶点（简化版本）
    vertices = [
        [-cube_size//2, -cube_size//2, -cube_size//2],
        [cube_size//2, -cube_size//2, -cube_size//2],
        [cube_size//2, cube_size//2, -cube_size//2],
        [-cube_size//2, cube_size//2, -cube_size//2],
        [-cube_size//2, -cube_size//2, cube_size//2],
        [cube_size//2, -cube_size//2, cube_size//2],
        [cube_size//2, cube_size//2, cube_size//2],
        [-cube_size//2, cube_size//2, cube_size//2]
    ]
    
    # 应用旋转和透视投影
    projected_vertices = []
    for vertex in vertices:
        # 应用Y轴旋转
        x, y, z = vertex
        new_x = x * cos_a - z * sin_a
        new_y = y  # Y坐标保持不变
        new_z = x * sin_a + z * cos_a
        
        # 简单的透视投影
        distance = 3.0
        scale = distance / (distance + new_z)
        proj_x = int(new_x * scale + cube_center[0])
        proj_y = int(new_y * scale + cube_center[1])
        
        projected_vertices.append([proj_x, proj_y])
    
    # 绘制立方体的边
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接边
    ]
    
    # 绘制边线
    for edge in edges:
        pt1 = tuple(projected_vertices[edge[0]])
        pt2 = tuple(projected_vertices[edge[1]])
        cv2.line(img, pt1, pt2, (0, 0, 255), 3)
    
    # 绘制顶点
    for vertex in projected_vertices:
        cv2.circle(img, tuple(vertex), 5, (255, 0, 0), -1)
    
    # 添加一些纹理和细节
    # 绘制一些圆形作为装饰
    for i in range(3):
        circle_x = center_x + int(50 * math.cos(angle + i * 2.1))
        circle_y = center_y + int(30 * math.sin(angle + i * 1.7))
        radius = 20 + i * 10
        color = (0, 100 + i * 50, 100 + i * 50)
        cv2.circle(img, (circle_x, circle_y), radius, color, -1)
    
    # 添加文字标识
    text = f"View {angle:.1f}"
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 添加一些噪声和纹理
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def generate_sample_photos():
    """生成8张示例照片"""
    
    # 确保real_data目录存在
    real_data_dir = Path("real_data")
    images_dir = real_data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("正在生成示例照片...")
    
    # 生成8张不同角度的照片
    num_photos = 8
    for i in range(num_photos):
        # 计算角度（0到2π）
        angle = i * 2 * math.pi / num_photos
        
        # 创建照片
        photo = create_sample_photo(angle)
        
        # 保存照片
        filename = f"image_{i:03d}.jpg"
        filepath = images_dir / filename
        cv2.imwrite(str(filepath), photo)
        
        print(f"生成照片: {filename} (角度: {angle:.2f})")
    
    # 更新数据集信息
    dataset_info_path = real_data_dir / "dataset_info.json"
    if dataset_info_path.exists():
        import json
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        dataset_info["num_images"] = num_photos
        
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n示例照片生成完成！")
    print(f"照片保存在: {images_dir}")
    print(f"共生成 {num_photos} 张照片")
    print("\n现在可以运行训练了！")

if __name__ == "__main__":
    generate_sample_photos()
