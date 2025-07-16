#!/usr/bin/env python3
"""
缩放栈数据结构演示脚本
演示 'Generative Powers of Ten' 论文中的缩放栈实现
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# 简化版的缩放栈类（不依赖 diffusers）
class ZoomStackDemo:
    """缩放栈演示版本"""
    
    def __init__(self, zoom_factors, H=256, W=256):
        self.zoom_factors = zoom_factors
        self.N = len(zoom_factors)
        self.H = H
        self.W = W
        
        print(f"创建缩放栈: {self.N} 层, {H}x{W} 分辨率")
        print(f"缩放因子: {zoom_factors}")
        
        # 验证缩放因子
        assert zoom_factors[0] == 1, "第一个缩放因子必须是1"
        for i in range(1, len(zoom_factors)):
            assert zoom_factors[i] == 2 * zoom_factors[i-1], "必须是2的幂序列"
        
        # 初始化层
        self.layers = self._create_demo_layers()
    
    def _create_demo_layers(self):
        """创建演示层（带有可视化模式）"""
        layers = []
        for i, p in enumerate(self.zoom_factors):
            # 创建不同的模式来演示不同的缩放级别
            layer = self._create_pattern_layer(i, p)
            layers.append(layer)
        return layers
    
    def _create_pattern_layer(self, layer_idx, zoom_factor):
        """为每层创建不同的可视化模式"""
        layer = torch.zeros(self.H, self.W, 3)
        
        # 为不同层创建不同的模式
        if layer_idx == 0:  # 最远距离 - 大尺度结构
            # 创建星系螺旋模式
            y, x = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing='ij')
            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(y, x)
            spiral = torch.sin(3 * theta + 5 * r)
            layer[:, :, 0] = 0.3 + 0.2 * spiral * torch.exp(-2 * r)  # 红色通道
            layer[:, :, 2] = 0.2 + 0.3 * spiral * torch.exp(-2 * r)  # 蓝色通道
            
        elif layer_idx == 1:  # 中等距离 - 恒星系统
            # 创建恒星和行星模式
            y, x = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing='ij')
            r = torch.sqrt(x**2 + y**2)
            # 中心恒星
            star = torch.exp(-50 * r**2)
            # 行星轨道
            for orbit_r in [0.3, 0.5, 0.7]:
                orbit = torch.exp(-100 * (r - orbit_r)**2)
                star += 0.3 * orbit
            layer[:, :, :] = torch.stack([star, star * 0.8, star * 0.6], dim=-1)
            
        elif layer_idx == 2:  # 近距离 - 行星表面
            # 创建地形模式
            y, x = torch.meshgrid(torch.linspace(-2, 2, self.H), torch.linspace(-2, 2, self.W), indexing='ij')
            terrain = torch.sin(x * 3) * torch.sin(y * 3) + 0.5 * torch.sin(x * 7) * torch.sin(y * 7)
            layer[:, :, 0] = 0.4 + 0.2 * terrain  # 红色 - 土壤
            layer[:, :, 1] = 0.6 + 0.3 * terrain  # 绿色 - 植被
            layer[:, :, 2] = 0.2 + 0.1 * terrain  # 蓝色
            
        else:  # 最近距离 - 微观结构
            # 创建细节纹理
            y, x = torch.meshgrid(torch.linspace(-4, 4, self.H), torch.linspace(-4, 4, self.W), indexing='ij')
            noise = torch.randn_like(x) * 0.1
            texture = torch.sin(x * 10 + noise) * torch.sin(y * 10 + noise)
            layer[:, :, 0] = 0.5 + 0.4 * texture
            layer[:, :, 1] = 0.3 + 0.2 * texture
            layer[:, :, 2] = 0.2 + 0.1 * texture
        
        # 确保值在 [0, 1] 范围内
        layer = torch.clamp(layer, 0, 1)
        return layer
    
    def visualize_layers(self, save_path="zoom_stack_demo.png"):
        """可视化所有层"""
        fig, axes = plt.subplots(1, self.N, figsize=(4 * self.N, 4))
        if self.N == 1:
            axes = [axes]
        
        layer_names = ["Galaxy", "Star System", "Planet Surface", "Microscopic"]
        
        for i, layer in enumerate(self.layers):
            axes[i].imshow(layer.numpy())
            name = layer_names[i] if i < len(layer_names) else f"Layer {i}"
            axes[i].set_title(f"L_{i}: {name}\n(zoom factor: {self.zoom_factors[i]}x)")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"可视化结果已保存为: {save_path}")
    
    def print_info(self):
        """打印缩放栈信息"""
        print(f"\n=== 缩放栈信息 ===")
        print(f"层数: {self.N}")
        print(f"图像尺寸: {self.H}x{self.W}")
        print(f"缩放因子: {self.zoom_factors}")
        
        for i, (layer, zoom) in enumerate(zip(self.layers, self.zoom_factors)):
            print(f"  L_{i}: 缩放={zoom}x, 形状={layer.shape}, 数据范围=[{layer.min():.3f}, {layer.max():.3f}]")

def demo_zoom_stack():
    """演示缩放栈功能"""
    print("=== 缩放栈数据结构演示 ===\n")
    
    # 创建缩放栈
    zoom_factors = [1, 2, 4, 8]
    zoom_stack = ZoomStackDemo(zoom_factors, H=128, W=128)
    
    # 打印信息
    zoom_stack.print_info()
    
    # 可视化
    print(f"\n正在生成可视化...")
    zoom_stack.visualize_layers()
    
    # 演示数据结构特性
    print(f"\n=== 缩放栈特性演示 ===")
    print(f"1. 所有层都是全分辨率 ({zoom_stack.H}x{zoom_stack.W}x3)")
    print(f"2. 每层代表不同的缩放级别：{zoom_factors}")
    print(f"3. 层数 = 缩放级别数 = {zoom_stack.N}")
    print(f"4. 缩放因子遵循 2^i 模式：{[2**i for i in range(len(zoom_factors))]}")
    
    # 展示张量属性
    print(f"\n=== 张量属性 ===")
    for i, layer in enumerate(zoom_stack.layers):
        print(f"L_{i}: 类型={type(layer)}, 形状={layer.shape}, dtype={layer.dtype}")

if __name__ == "__main__":
    demo_zoom_stack() 