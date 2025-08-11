import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class EnhancedFeatureExtractor(nn.Module):
    """增强型特征提取器，用于提取输入图像的多层次特征"""
    def __init__(self, depth=5, channels=[64, 128, 256, 512, 1024], use_attention=True):
        super().__init__()
        self.depth = depth
        self.use_attention = use_attention
        
        # 初始卷积层
        self.initial_conv = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(channels[0])
        
        # 特征提取块
        self.features = nn.ModuleList()
        for i in range(depth):
            in_channels = channels[i]
            out_channels = channels[i+1] if i+1 < depth else channels[i]
            
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            self.features.append(block)
            
            # 添加注意力机制
            if use_attention and i > 0:
                self.features.append(ChannelAttention(out_channels))
    
    def forward(self, x):
        """
        前向传播函数
        x: 输入图像张量，形状为 [B, 3, H, W]
        返回: 特征列表，包含不同层次的特征图
        """
        features = []
        
        # 初始卷积
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)
        features.append(x)
        
        # 通过特征提取块
        for block in self.features:
            x = block(x)
            features.append(x)
            
        return features

class ChannelAttention(nn.Module):
    """通道注意力模块，用于增强重要特征通道"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DetailEnhancementModule(nn.Module):
    """细节增强模块，用于保留和增强图像的细微纹理"""
    def __init__(self, in_channels, out_channels, num_res_blocks=8):
        super().__init__()
        
        # 残差块序列
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(in_channels) for _ in range(num_res_blocks)]
        )
        
        # 特征融合和输出
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, high_level_feat, features):
        """
        前向传播函数
        high_level_feat: 高级特征图（来自特征提取器的最后一层）
        features: 所有层次的特征图列表
        返回: 增强后的特征图
        """
        # 取最低级特征进行融合（包含最丰富的细节）
        low_level_feat = features[1]
        
        # 上采样高级特征以匹配低级特征尺寸
        high_level_upsampled = F.interpolate(
            high_level_feat, 
            size=low_level_feat.shape[2:], 
            mode='bilinear', 
            align_corners=True
        )
        
        # 融合高低级特征
        x = torch.cat([low_level_feat, high_level_upsampled], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 通过残差块增强细节
        x = self.res_blocks(x)
        
        # 最终输出
        x = self.conv2(x)
        return x

class ResidualBlock(nn.Module):
    """残差块，用于特征增强网络"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        return x

class GaussianHead(nn.Module):
    """高斯头部模块，用于生成3D高斯点云"""
    def __init__(self, point_density=1024, feature_dim=512):
        super().__init__()
        self.point_density = point_density
        
        # 预测高斯点云参数
        self.conv_mean = nn.Conv2d(feature_dim, 3 * point_density, kernel_size=1)  # 3D坐标
        self.conv_cov = nn.Conv2d(feature_dim, 6 * point_density, kernel_size=1)   # 协方差矩阵参数
        self.conv_color = nn.Conv2d(feature_dim, 3 * point_density, kernel_size=1) # 颜色
        
    def forward(self, features, depth_map=None):
        """
        前向传播函数
        features: 增强后的特征图
        depth_map: 可选的深度图，用于引导点云生成
        返回: 高斯点云参数
        """
        # 全局平均池化获取全局特征
        x = F.adaptive_avg_pool2d(features, 1)
        
        # 预测高斯点云参数
        mean = self.conv_mean(x).view(-1, self.point_density, 3)  # [B, N, 3]
        cov = self.conv_cov(x).view(-1, self.point_density, 6)    # [B, N, 6]
        color = self.conv_color(x).view(-1, self.point_density, 3) # [B, N, 3]
        
        # 如果有深度图，使用深度信息调整3D坐标
        if depth_map is not None:
            depth = F.adaptive_avg_pool2d(depth_map, 1).view(-1, 1, 1)  # [B, 1, 1]
            mean = mean * depth  # 基于深度缩放坐标
        
        # 确保协方差矩阵正定
        cov = torch.exp(cov)
        
        # 颜色归一化到[0, 1]
        color = torch.sigmoid(color)
        
        return {
            'means3D': mean,
            'cov3D': cov,
            'colors': color,
            'num_points': self.point_density
        }

class CloseupGS(nn.Module):
    """Closeup GS模型主类，整合特征提取、细节增强和高斯点云生成"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = EnhancedFeatureExtractor(
            depth=config.feature_depth, 
            channels=config.feature_channels,
            use_attention=config.use_attention
        )
        self.detail_enhancer = DetailEnhancementModule(
            in_channels=config.feature_channels[-1],
            out_channels=3,
            num_res_blocks=config.num_res_blocks
        )
        self.gaussian_head = GaussianHead(
            point_density=config.point_density,
            feature_dim=config.feature_channels[-1]
        )
        
        # 加载预训练权重
        if config.pretrained_weights:
            self.load_pretrained(config.pretrained_weights)
        
    def forward(self, x, depth_map=None):
        """
        前向传播函数
        x: 输入图像 [B, 3, H, W]
        depth_map: 可选的深度图 [B, 1, H, W]
        返回: 3D高斯点云参数
        """
        # 1. 特征提取
        features = self.feature_extractor(x)
        
        # 2. 细节增强
        enhanced_features = self.detail_enhancer(features[-1], features)
        
        # 3. 生成高斯点云
        point_cloud = self.gaussian_head(enhanced_features, depth_map)
            
        return point_cloud
        
    def load_pretrained(self, weights_path):
        """加载预训练权重"""
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained weights from {weights_path}")
        
    def save_model(self, save_path, epoch=0, loss=0.0):
        """保存模型权重"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
        }, save_path)
        print(f"Model saved to {save_path}")
