#!/usr/bin/env python3
"""
创建真实照片数据集结构
用于测试Close-up-GS模型
"""

import os
import json
import numpy as np
from pathlib import Path

def create_real_dataset_structure():
    """创建真实照片数据集结构"""
    
    # 创建数据集目录
    dataset_dir = Path("real_data")
    dataset_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    images_dir = dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    poses_dir = dataset_dir / "poses"
    poses_dir.mkdir(exist_ok=True)
    
    # 创建数据集信息文件
    dataset_info = {
        "dataset_type": "real_photos",
        "num_images": 0,
        "image_width": 512,
        "image_height": 512,
        "focal_length": 1000.0,
        "camera_center": [0.0, 0.0, 0.0],
        "object_center": [0.0, 0.0, 2.0],  # 物体在z=2位置
        "description": "真实照片数据集，用于测试Close-up-GS模型"
    }
    
    with open(dataset_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    # 创建相机参数文件
    camera_params = {
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 256.0,
        "cy": 256.0,
        "width": 512,
        "height": 512
    }
    
    with open(dataset_dir / "camera_params.json", "w", encoding="utf-8") as f:
        json.dump(camera_params, f, indent=2)
    
    # 创建示例相机姿态（围绕物体旋转）
    num_views = 8
    poses = []
    
    for i in range(num_views):
        angle = i * 2 * np.pi / num_views
        radius = 3.0
        
        # 相机位置（围绕物体旋转）
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.0  # 相机高度
        
        # 相机朝向物体中心
        look_at = np.array([0.0, 0.0, 2.0])  # 物体中心
        camera_pos = np.array([x, y, z])
        
        # 计算相机坐标系
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # 构建旋转矩阵
        rotation = np.column_stack([right, up, -forward])
        
        # 构建变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = camera_pos
        
        poses.append({
            "image_id": f"image_{i:03d}",
            "transform_matrix": transform.tolist(),
            "camera_position": camera_pos.tolist(),
            "look_at": look_at.tolist()
        })
    
    # 保存相机姿态
    with open(poses_dir / "camera_poses.json", "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
    
    # 创建README文件
    readme_content = """# 真实照片数据集

## 数据集结构
```
real_data/
├── images/              # 放置真实照片
├── poses/              # 相机姿态文件
├── dataset_info.json   # 数据集信息
├── camera_params.json  # 相机参数
└── README.md          # 说明文件
```

## 使用方法

1. 将真实照片放入 `images/` 目录
2. 照片命名格式：`image_000.jpg`, `image_001.jpg`, ...
3. 照片数量应与 `camera_poses.json` 中的姿态数量一致
4. 运行训练：
   ```bash
   python train_closeup_gs.py --data_path ./real_data --dataset_type real_photos --target_resolution 512 512
   ```

## 相机参数
- 焦距：1000像素
- 图像尺寸：512x512
- 相机围绕物体旋转，距离3米
- 物体中心位置：[0, 0, 2]

## 注意事项
- 照片应该是同一物体的不同角度拍摄
- 建议使用8-12张照片
- 照片质量越高，训练效果越好
- 确保照片光照条件相对一致
"""
    
    with open(dataset_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("真实照片数据集结构创建完成！")
    print(f"数据集目录: {dataset_dir}")
    print("\n下一步：")
    print("1. 将真实照片放入 images/ 目录")
    print("2. 照片命名：image_000.jpg, image_001.jpg, ...")
    print("3. 运行训练脚本")

if __name__ == "__main__":
    create_real_dataset_structure()
