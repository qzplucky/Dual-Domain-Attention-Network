import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from astropy.io import fits
import matplotlib.pyplot as plt


class FITSDataset(Dataset):
    def __init__(self, file_list, crop_size=256, noise_level_range=(0.05, 0.2),
                 resize_size=(512, 512), augment=False, debug_output_dir=None):
        self.file_list = file_list
        self.crop_size = crop_size
        self.noise_level_range = noise_level_range
        self.resize_size = (resize_size[0], resize_size[1])
        self.augment = augment
        self.debug_output_dir = debug_output_dir
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(np.float32)  # 加载原始光谱数据

        # —— 1. 调整尺寸 —— #
        data_resized = cv2.resize(data, self.resize_size, interpolation=cv2.INTER_LINEAR)

        # —— 2. 生成带噪数据 —— #
        data_final = np.ascontiguousarray(data_resized)
        noise_level = random.uniform(*self.noise_level_range)
        noisy1 = self.add_gaussian_noise(data_final, noise_level)
        noisy2 = self.add_gaussian_noise(data_final, noise_level)

        # —— 3. 归一化（统一到 [0,1] 范围） —— #
        def normalize(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-8)  # 避免除零

        raw_t = torch.from_numpy(data_final).unsqueeze(0).float()  # [1, H, W]
        n1_t = torch.from_numpy(noisy1).unsqueeze(0).float()
        n2_t = torch.from_numpy(noisy2).unsqueeze(0).float()

        raw_t = normalize(raw_t)
        n1_t = normalize(n1_t)
        n2_t = normalize(n2_t)

        # —— 4. 调试可视化（可选） —— #
        if self.debug_output_dir and idx < 5:
            self._save_debug_visualization(data, data_resized, noisy1, idx)

        return {
            'noisy1': n1_t,
            'noisy2': n2_t,
            'raw':    raw_t,
            'noise_level': noise_level
        }

    @staticmethod
    def add_gaussian_noise(data, sigma):
        """添加高斯噪声（保持非负）"""
        noisy = data + np.random.normal(0, sigma, data.shape).astype(np.float32)
        return np.clip(noisy, 0, None)  # 确保像素值非负

    def _save_debug_visualization(self, raw_data, resized_data, noisy_data, idx):
        """保存调试图像"""
        plt.figure(figsize=(12, 6))
        # 1. 原始数据 vs 调整大小后的数据
        plt.subplot(221)
        plt.imshow(raw_data, cmap='gray')
        plt.title('原始数据')
        plt.colorbar()

        plt.subplot(222)
        plt.imshow(resized_data, cmap='gray')
        plt.title('调整大小后的数据')
        plt.colorbar()

        # 2. 带噪数据
        plt.subplot(223)
        plt.imshow(noisy_data, cmap='gray')
        plt.title(f'带噪数据（σ={np.round(np.std(noisy_data - resized_data), 3)}）')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_output_dir, f'sample_{idx}_debug.png'))

        plt.close()

