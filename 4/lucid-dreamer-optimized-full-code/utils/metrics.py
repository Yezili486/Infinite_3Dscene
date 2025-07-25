import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, data_range=1.0):
    """
    计算峰值信噪比(PSNR)
    
    参数:
        img1: 张量形式的图像1 (C, H, W)
        img2: 张量形式的图像2 (C, H, W)
        data_range: 数据范围（默认为1.0，适用于归一化到[0,1]的图像）
    
    返回:
        psnr: PSNR值
    """
    if img1.shape != img2.shape:
        # 调整图像尺寸以匹配
        img2 = F.interpolate(img2.unsqueeze(0), size=img1.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
    
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse)

def calculate_ssim(img1, img2, data_range=1.0, window_size=11):
    """
    计算结构相似性指数(SSIM)
    
    参数:
        img1: 张量形式的图像1 (C, H, W)
        img2: 张量形式的图像2 (C, H, W)
        data_range: 数据范围（默认为1.0）
        window_size: 高斯窗口大小
    
    返回:
        ssim: SSIM值
    """
    if img1.shape != img2.shape:
        # 调整图像尺寸以匹配
        img2 = F.interpolate(img2.unsqueeze(0), size=img1.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
    
    # 确保图像是4D张量 (N, C, H, W)
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # 创建高斯窗口
    device = img1.device
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*(window_size//2)**2)) for x in range(window_size)]).to(device)
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    # 计算方差和协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def evaluate_model(rendered_img, target_img):
    """
    评估模型生成图像的质量
    
    参数:
        rendered_img: 模型渲染的图像 (C, H, W)
        target_img: 目标参考图像 (C, H, W)
    
    返回:
        metrics: 包含PSNR和SSIM的字典
    """
    # 确保图像在相同设备上
    target_img = target_img.to(rendered_img.device)
    
    # 计算指标
    psnr = calculate_psnr(rendered_img, target_img)
    ssim = calculate_ssim(rendered_img, target_img)
    
    return {
        'psnr': psnr.item(),
        'ssim': ssim.item()
    }
