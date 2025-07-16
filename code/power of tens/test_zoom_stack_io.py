#!/usr/bin/env python3
"""
缩放栈输入输出测试脚本
展示 ZoomStack 的具体输入输出格式
"""

import torch
import numpy as np

def test_zoom_stack_io():
    """测试缩放栈的输入输出"""
    
    print("=== 缩放栈输入输出测试 ===\n")
    
    # ==================== 输入参数 ====================
    print("📥 输入参数:")
    zoom_factors = [1, 2, 4, 8]
    H, W = 256, 256  # 使用较小分辨率便于演示
    device = "cpu"   # 使用 CPU 便于演示
    
    print(f"  - zoom_factors: {zoom_factors}")
    print(f"  - 图像尺寸: {H}×{W}")
    print(f"  - 设备: {device}")
    print(f"  - 层数: {len(zoom_factors)}")
    
    # ==================== 创建缩放栈 ====================
    print(f"\n🔄 创建缩放栈...")
    
    # 简化版缩放栈（不依赖外部库）
    class SimpleZoomStack:
        def __init__(self, zoom_factors, H, W, device="cpu"):
            self.zoom_factors = zoom_factors
            self.N = len(zoom_factors)
            self.H, self.W = H, W
            self.device = device
            
            # 初始化层 - 从高斯噪声开始
            self.layers = []
            for i, zoom in enumerate(zoom_factors):
                # 每层都是 H×W×3 张量
                layer = torch.randn(H, W, 3) * 0.1  # 小幅度噪声
                self.layers.append(layer)
    
    zoom_stack = SimpleZoomStack(zoom_factors, H, W, device)
    
    # ==================== 输出分析 ====================
    print(f"\n📤 输出结构:")
    print(f"  - 缩放栈对象: {type(zoom_stack)}")
    print(f"  - 总层数: {zoom_stack.N}")
    print(f"  - 层列表长度: {len(zoom_stack.layers)}")
    
    print(f"\n📊 每层详细信息:")
    for i, (layer, zoom) in enumerate(zip(zoom_stack.layers, zoom_factors)):
        print(f"  L_{i}: 形状={layer.shape}, dtype={layer.dtype}, 缩放={zoom}x")
        print(f"       数据范围=[{layer.min():.3f}, {layer.max():.3f}]")
        print(f"       内存使用={layer.numel() * 4 / 1024:.1f} KB")  # float32 = 4 bytes
    
    # ==================== 实际数据示例 ====================
    print(f"\n🔍 数据示例（左上角 3×3 像素）:")
    for i, layer in enumerate(zoom_stack.layers):
        print(f"\n  L_{i} (缩放={zoom_factors[i]}x):")
        sample = layer[:3, :3, 0]  # 只看红色通道的 3×3 区域
        print(f"    {sample.numpy()}")
    
    # ==================== 使用场景 ====================
    print(f"\n🎯 典型使用场景:")
    
    # 1. 获取单层
    layer_0 = zoom_stack.layers[0]
    print(f"  1. 获取第0层: 形状={layer_0.shape}")
    
    # 2. 修改单层
    zoom_stack.layers[1] = torch.zeros_like(zoom_stack.layers[1])
    print(f"  2. 修改第1层为零: 新范围=[{zoom_stack.layers[1].min():.3f}, {zoom_stack.layers[1].max():.3f}]")
    
    # 3. 批量处理
    all_layers = zoom_stack.layers
    total_memory = sum(layer.numel() * 4 for layer in all_layers) / (1024*1024)
    print(f"  3. 总内存使用: {total_memory:.2f} MB")
    
    # ==================== 与论文对应关系 ====================
    print(f"\n📚 与论文的对应关系:")
    print(f"  - 论文符号 L = [L_0, L_1, ..., L_{{N-1}}]")
    print(f"  - 实际实现: zoom_stack.layers = {[f'L_{i}' for i in range(zoom_stack.N)]}")
    print(f"  - 缩放因子 p_i = 2^i: {[f'p_{i}=2^{i}={2**i}' for i in range(zoom_stack.N)]}")
    
    # ==================== 数据流示例 ====================
    print(f"\n🔄 数据流示例:")
    print(f"  输入: zoom_factors={zoom_factors}")
    print(f"  处理: 创建 {zoom_stack.N} 个 {H}×{W}×3 张量")
    print(f"  输出: 缩放栈对象，包含 {len(zoom_stack.layers)} 层")
    
    return zoom_stack

def demonstrate_zoom_stack_operations():
    """演示缩放栈的基本操作"""
    print(f"\n\n=== 缩放栈操作演示 ===")
    
    zoom_stack = test_zoom_stack_io()
    
    print(f"\n🛠️ 基本操作:")
    
    # 读取操作
    print(f"  - 读取: layer = zoom_stack.layers[i]")
    example_layer = zoom_stack.layers[0]
    print(f"    示例: layer.shape = {example_layer.shape}")
    
    # 写入操作  
    print(f"  - 写入: zoom_stack.layers[i] = new_layer")
    new_layer = torch.ones_like(example_layer) * 0.5
    zoom_stack.layers[0] = new_layer
    print(f"    示例: 设置为常数 0.5，范围=[{zoom_stack.layers[0].min():.1f}, {zoom_stack.layers[0].max():.1f}]")
    
    # 批量操作
    print(f"  - 批量: all_layers = zoom_stack.layers")
    all_shapes = [layer.shape for layer in zoom_stack.layers]
    print(f"    示例: 所有形状 = {all_shapes}")

if __name__ == "__main__":
    demonstrate_zoom_stack_operations() 