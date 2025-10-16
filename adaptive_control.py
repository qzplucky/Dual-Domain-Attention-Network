# adaptive_control.py

import torch
import numpy as np

class AdaptiveWeightController:
    """自适应调整不同损失函数的权重"""
    def __init__(self, config):
        self.config = config
        self.spectral_weight = config['base_spectral_weight']
        self.wavelet_weight = config['base_wavelet_weight']
        self.performance_history = []
        self.weight_history = []

    def update_weights(self, epoch, val_metric):
        """
        根据验证集性能（如 PSNR）动态调整损失权重。
        val_metric: 当前 epoch 的 PSNR 或其他指标（数值越大越好）。
        """
        if not self.config['adaptive_weights']:
            return self.spectral_weight, self.wavelet_weight

        # 将 PSNR 转换成越小越好的“伪损失”方便比较
        current_loss_like = -val_metric
        self.performance_history.append(current_loss_like)

        # 仅在指定频率更新
        if epoch % self.config['weight_update_freq'] != 0:
            return self.spectral_weight, self.wavelet_weight

        if len(self.performance_history) < 2:
            return self.spectral_weight, self.wavelet_weight

        prev = self.performance_history[-2]
        curr = self.performance_history[-1]
        # 变化率（负数表示指标下降）
        change = (curr - prev) / (abs(prev) + 1e-8)

        # 如果指标变化幅度超过阈值，则调整权重
        if change < -self.config['performance_threshold']:
            # 指标下降，减少正则化强度
            self.spectral_weight = max(0.01, self.spectral_weight * 0.9)
            self.wavelet_weight = max(0.01, self.wavelet_weight * 0.9)
        elif change > self.config['performance_threshold']:
            # 指标提升，增加正则化强度
            self.spectral_weight = min(0.5, self.spectral_weight * 1.1)
            self.wavelet_weight = min(0.5, self.wavelet_weight * 1.1)

        self.weight_history.append((self.spectral_weight, self.wavelet_weight))
        return self.spectral_weight, self.wavelet_weight
