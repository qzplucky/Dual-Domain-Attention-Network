import torch
import torch.nn as nn
import numpy as np
import pywt


class SpectralSmoothnessLoss(nn.Module):
    """
    计算谱向（二阶差分）平滑度损失，惩罚波长方向快速变化。
    """
    def __init__(self, config=None):
        super(SpectralSmoothnessLoss, self).__init__()
        self.config = config

    def forward(self, output, target=None, mask=None):
        # output: [B, C=1, H, W]
        # 计算波长方向(宽度方向, W)的二阶差分
        diff1 = output[:, :, :, 2:] - output[:, :, :, 1:-1]
        diff2 = output[:, :, :, 1:-1] - output[:, :, :, :-2]
        second_diff = diff1 - diff2  # 形状 [B, C, H, W-2]

        loss = torch.mean(second_diff ** 2)
        return loss


class WaveletLoss(nn.Module):
    """
     db8 小波基、逐行多层分解，只对高频子带做 MSE，
    并最终平均到“行数×层数”。
    """
    def __init__(self, wavelet='db8', level=3):
        super(WaveletLoss, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, output, target):
        """
        Args:
          output, target: Tensor of shape [B, C=1, H, W]
        Returns:
          标量 loss
        """
        b, c, H, W = output.shape
        device = output.device

        total_loss = 0.0
        total_rows = b * c * H  # 总行数

        # 对每一行独立做一维多层小波分解
        for bi in range(b):
            for ci in range(c):
                for hi in range(H):
                    # 提取一行，pywt 分解
                    x_row = output[bi, ci, hi].detach().cpu().numpy()
                    y_row = target[bi, ci, hi].detach().cpu().numpy()
                    coeffs_x = pywt.wavedec(x_row, self.wavelet, level=self.level)
                    coeffs_y = pywt.wavedec(y_row, self.wavelet, level=self.level)

                    # 对每一个高频子带计算 MSE
                    for lvl in range(1, len(coeffs_x)):
                        diff = (coeffs_x[lvl] - coeffs_y[lvl]).astype(np.float32)
                        total_loss += np.mean(diff ** 2)

        # 归一化：除以（总行数 × 高频层数）
        loss_np = total_loss / (total_rows * self.level)
        return torch.tensor(loss_np, device=device, dtype=torch.float32)


class CombinedLoss(nn.Module):
    """
    组合 MSELoss、SpectralSmoothnessLoss、WaveletLoss，支持自适应权重更新。
    """
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = SpectralSmoothnessLoss(config)
        # 使用已去除 mask 的 WaveletLoss
        self.wavelet_loss = WaveletLoss(wavelet='db8', level=config.get('wavelet_level', 3))

        # 初始权重
        self.spectral_weight = config.get('base_spectral_weight', 1.0)
        self.wavelet_weight = config.get('base_wavelet_weight', 1.0)

    def set_weights(self, spectral_weight, wavelet_weight):
        self.spectral_weight = spectral_weight
        self.wavelet_weight = wavelet_weight

    def forward(self, output, target):
        # 基本重建损失
        mse = self.mse_loss(output, target)

        # 谱向平滑损失
        spectral = self.spectral_loss(output) if self.config.get('use_spectral_loss', False) else 0.0

        # 小波域损失
        wavelet = self.wavelet_loss(output, target) if self.config.get('use_wavelet_loss', False) else 0.0

        total_loss = mse + self.spectral_weight * spectral + self.wavelet_weight * wavelet
        return total_loss
