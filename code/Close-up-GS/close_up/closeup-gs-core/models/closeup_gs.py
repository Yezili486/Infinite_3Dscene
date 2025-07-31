import torch
import torch.nn as nn
import kornia as K
from einops import rearrange

class CloseupGaussianEnhancer(nn.Module):
    """Closeup GS 专用的高斯点细节增强网络"""
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        # 细节特征提取（针对近景物体纹理）
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, input_dim, kernel_size=3, padding=1)
        )
        # 空间注意力（突出物体边缘细节）
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, gaussian_points):
        """
        输入：gaussian_points (N, 3) - 3DGS 初始点云
        输出：enhanced_points (N, 3) - 增强细节后的点云
        """
        # 提取局部特征
        x = rearrange(gaussian_points, "n d -> 1 d n")  # 适配 Conv1d 输入格式
        features = self.feature_extractor(x)
        # 注意力加权
        attn = self.attention(features)
        enhanced = gaussian_points + 0.1 * (features * attn).squeeze(0).T
        return enhanced
    