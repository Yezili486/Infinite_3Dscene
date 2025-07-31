import torch
import numpy as np
from gaussian_splatting.scene import Scene
from gaussian_splatting.gaussian_renderer import render as render_gs

def render(point_cloud, cameras, model_path, output_dir, iterations=3000):
    """
    使用 3DGS 渲染 3D 场景
    point_cloud: 增强后的高斯点云
    cameras: 相机参数 JSON 路径
    model_path: 3DGS 基础模型权重
    output_dir: 输出目录
    """
    # 初始化场景
    scene = Scene(cameras, point_cloud, model_path)
    scene.train(iterations)  # 优化高斯参数
    
    # 渲染多角度视图
    os.makedirs(output_dir, exist_ok=True)
    render_poses = scene.get_camera_poses()  # 获取多角度相机位姿
    
    for i, pose in enumerate(render_poses[:10]):  # 渲染前10个视角
        rendering = render_gs(scene.gaussians, pose, scene.render_params)
        # 保存渲染结果
        img = Image.fromarray((rendering * 255).astype(np.uint8))
        img.save(f"{output_dir}/render_{i:03d}.jpg")
    
    return output_dir
    