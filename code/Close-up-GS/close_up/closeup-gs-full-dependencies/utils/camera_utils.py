import json
import numpy as np

def load_cameras(cameras_path):
    """加载相机参数 JSON 文件并转换为 3DGS 所需格式"""
    with open(cameras_path, 'r') as f:
        cameras = json.load(f)
    
    # 转换为 3DGS 兼容的相机参数格式（内参、外参矩阵）
    processed = []
    for cam in cameras:
        # 内参矩阵 (3x3)
        K = np.array([
            [cam['fx'], 0, cam['cx']],
            [0, cam['fy'], cam['cy']],
            [0, 0, 1]
        ])
        # 外参矩阵 (4x4，旋转+平移)
        R = np.array(cam['rotation'])
        T = np.array(cam['translation']).reshape(3, 1)
        RT = np.hstack([R, T])
        RT = np.vstack([RT, [0, 0, 0, 1]])
        
        processed.append({
            'K': K,
            'RT': RT,
            'width': cam['width'],
            'height': cam['height']
        })
    return processed
    