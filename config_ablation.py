# config_ablation.py
from config import train_config

# 基础配置
base_config = train_config.copy()

# 消融实验配置组
ablation_configs = {
    "baseline": base_config,
    
    # 损失函数消融
    "no_spectral_loss": {**base_config, "use_spectral_loss": False},
    "no_wavelet_loss": {**base_config, "use_wavelet_loss": False},
    "no_adaptive_weights": {**base_config, "adaptive_weights": False},
    

    "no_attention": {**base_config, "use_spectral_attention": False},
   
    # 组合消融
    "minimal": {
        **base_config,
        "use_spectral_loss": False,
        "use_wavelet_loss": False,
        # "use_local_filter": False,
        "use_spectral_attention": False,
        # "use_spectral_augmentation": False
    }
}
