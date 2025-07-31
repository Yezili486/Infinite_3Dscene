import torch
import torch.nn as nn
import torch.nn.functional as F

class CloseupGaussianEnhancer(nn.Module):
    """Closeup GS 专用的高斯点云细节增强网络"""
    def __init__(self, feature_dim=32):
        super().__init__()
        # 对近景区域的高斯点进行精细化调整
        self.conv1 = nn.Conv1d(6, feature_dim, kernel_size=1)  # 输入：x,y,z + 颜色(r,g,b)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim*2, kernel_size=1)
        self.conv3 = nn.Conv1d(feature_dim*2, 6, kernel_size=1)  # 输出：位置偏移 + 颜色微调
        self.norm1 = nn.BatchNorm1d(feature_dim)
        self.norm2 = nn.BatchNorm1d(feature_dim*2)
        
    def forward(self, gaussians):
        """
        输入: gaussians - [N, 6] 高斯点云 (x,y,z,r,g,b)
        输出: enhanced_gaussians - 增强细节后的高斯点云
        """
        x = gaussians.transpose(1, 0).unsqueeze(0)  # [1, 6, N]
        
        # 特征提取与增强
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        delta = self.conv3(x).squeeze(0).transpose(1, 0)  # [N, 6] 偏移量
        
        # 对原始高斯点云进行微调（增强细节）
        enhanced = gaussians + delta * 0.1  # 控制调整幅度
        return enhanced
    