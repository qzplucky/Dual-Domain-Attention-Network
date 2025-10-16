# config.py

train_config = {
    'data_dir': '/home/zthao_/noise2/train_fits/double',
    'val_dir': '/home/zthao_/noise2/val_train/val',
    'batch_size': 16,
    'num_workers': 4,
    'epochs': 50,
    'lr': 0.001,
    'lr_step_size': 30,
    'lr_gamma': 0.5,
    'save_dir': './checkpoints',
    'save_freq': 10,
    'log_dir': './logs',

    # 输入图像尺寸已从 (1, 1, 2048) 改为 512x512
    'input_size': (1, 512, 512),
    'spectrum_length': 2048,       # 依然保留，用于 Ca 检测中的像素计算
    'resize_method': 'interpolate',
    'normalize': True,
    'standardize': False,

    # Noise2Noise 训练范式，仅保留高斯噪声
    'training_paradigm': 'noise2noise',
    'noise_levels': [0, 0],      # 高斯噪声 sigma 范围
    'noise_augmentation': True,

    # 改进方法的基础权重
    'use_spectral_loss': True,
    'base_spectral_weight': 0.05,
    'use_wavelet_loss': True,
    'base_wavelet_weight': 0.05, #此处原始值0.2
    # 'use_local_filter': True,

    # 自适应权重控制
    'adaptive_weights': True,
    'weight_update_freq': 3,
    'performance_threshold': 0.01,

    # # 局部滤波参数
    # 'local_kernel_size': 5,
    # 'line_threshold': 0.9,

    # 学习率自适应控制
    'use_lr_scheduler': True,
    'reduce_on_plateau': True,
    'plateau_patience': 5,
    'plateau_factor': 0.5,

    'use_attention': True,
    # HybridAttention 内部也可独立控制空间或小波注意力：
    'use_spatial_attention': True,
    'use_wavelet_attention': True,

    # # 数据增强
    # # 'use_spectral_augmentation': True,
    # 'augment_absorption_lines': True,
    # 'absorption_line_prob': 0.3,

    # 评估指标
    'eval_metrics': ['psnr', 'ssim', 'mse', 'sre'],
    'early_stopping': True,
    'early_stop_patience': 15,
}
