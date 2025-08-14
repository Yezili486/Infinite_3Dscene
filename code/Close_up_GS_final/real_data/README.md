# 真实照片数据集

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
