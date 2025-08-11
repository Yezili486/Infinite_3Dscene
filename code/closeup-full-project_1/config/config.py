class Config:
    """项目配置类，包含所有模型和训练的参数设置"""
    
    def __init__(self):
        # 特征提取器参数
        self.feature_depth = 5  # 特征提取器深度
        self.feature_channels = [64, 128, 256, 512, 1024]  # 各层特征通道数
        self.use_attention = True  # 是否使用注意力机制
        
        # 细节增强模块参数
        self.num_res_blocks = 8  # 残差块数量
        
        # 高斯点云参数
        self.point_density = 2048  # 点云密度（点的数量）
        
        # 预训练权重路径
        self.pretrained_weights = None  # 可以设置为预训练权重文件路径
        
        # 训练参数
        self.learning_rate = 1e-4
        self.batch_size = 8
        self.num_epochs = 100
        self.val_interval = 5  # 验证间隔（epoch）
        self.save_interval = 10  # 模型保存间隔（epoch）
        
        # 数据参数
        self.image_size = (1024, 768)  # 输入图像尺寸
        self.train_data_path = "../data/train"  # 训练数据路径
        self.val_data_path = "../data/val"      # 验证数据路径
        
        # 日志和输出参数
        self.log_dir = "../logs"       # 日志目录
        self.checkpoint_dir = "../checkpoints"  # 模型 checkpoint 目录
        self.output_dir = "../outputs"  # 输出目录
        
        # 设备参数
        self.use_cuda = True  # 是否使用CUDA
        self.seed = 42  # 随机种子，用于结果复现
        
        # 相机参数
        self.camera_fov = 60  # 相机视场角（度）
        self.camera_near = 0.1  # 近平面
        self.camera_far = 1000  # 远平面
        
        # Diffusion模型参数
        self.diffusion_model_name = "stable-diffusion-v1-5"
        self.diffusion_num_inference_steps = 50
        self.diffusion_guidance_scale = 7.5
        
        # 相机对齐参数
        self.camera_alignment_iterations = 20
        self.camera_alignment_lr = 0.1
        
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"配置中不存在参数: {key}")
                
    def to_dict(self):
        """将配置转换为字典"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('__') and not callable(key)
        }
    
    def __str__(self):
        """返回配置的字符串表示"""
        config_str = "Config:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str
