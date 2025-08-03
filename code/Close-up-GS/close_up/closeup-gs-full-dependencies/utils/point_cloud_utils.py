import numpy as np
import plyfile
from PIL import Image

def generate_point_cloud(image, depth_map, fx=500, fy=500, cx=None, cy=None):
    """
    从 RGB 图像和深度图生成初始点云
    image: PIL 图像 (RGB)
    depth_map: 深度图 (2D 数组)
    fx, fy: 相机内参焦距
    cx, cy: 主点坐标（默认图像中心）
    """
    img = np.array(image)
    h, w = depth_map.shape
    
    # 确保图像和深度图尺寸一致
    if img.shape[:2] != (h, w):
        img = np.array(Image.fromarray(img).resize((w, h)))
    
    cx = w // 2 if cx is None else cx
    cy = h // 2 if cy is None else cy
    
    # 生成像素坐标网格
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    
    # 转换为相机坐标系 3D 点
    z = depth_map / 1000.0  # 假设深度图单位为毫米，转换为米
    x3d = (xx - cx) * z / fx
    y3d = (yy - cy) * z / fy
    
    # 获取对应像素颜色
    colors = img[yy, xx] / 255.0  # 归一化到 [0,1]
    
    # 确保所有数组都是相同的形状
    x3d_flat = x3d.ravel()
    y3d_flat = y3d.ravel()
    z_flat = z.ravel()
    r_flat = colors[:,:,0].ravel()
    g_flat = colors[:,:,1].ravel()
    b_flat = colors[:,:,2].ravel()
    
    # 合并为点云 (x,y,z,r,g,b)
    point_cloud = np.column_stack([x3d_flat, y3d_flat, z_flat, r_flat, g_flat, b_flat])
    
    # 过滤无效点（深度为0的点）
    valid_mask = point_cloud[:, 2] > 0.1
    return point_cloud[valid_mask]

def save_ply(point_cloud, path):
    """保存点云为 PLY 文件"""
    vertices = np.array([(x, y, z, r, g, b) for x, y, z, r, g, b in point_cloud],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertices, 'vertex')])
    ply.write(path)
    