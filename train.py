import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim_metric
from astropy.io import fits
# 导入前两个代码中的类
from dataset import FITSDataset  # 对应第二个代码的数据集
from unet import *   # 对应第一个代码的模型
# 导入新增的损失函数
from losses import CombinedLoss, SpectralSmoothnessLoss, WaveletLoss
from adaptive_control import AdaptiveWeightController

class Train:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 目录创建
        self.log_dir = config['log_dir']
        self.correlation_dir = os.path.join(self.log_dir, 'correlation_maps')
        self.power_spectrum_dir = os.path.join(self.log_dir, 'power_spectra')
        self.controller = AdaptiveWeightController(config)
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.correlation_dir, exist_ok=True)
        os.makedirs(self.power_spectrum_dir, exist_ok=True)

        # 模型初始化
        self.model = UnifiedDenoisingUNet(
            n_channels=1,
            n_classes=1,
            bilinear=config.get('bilinear', True),
            wavelet=config.get('wavelet', 'db8'),
            use_attention=config.get('use_attention', True)
        ).to(self.device)

        # 损失函数（使用CombinedLoss）
        self.criterion = CombinedLoss(config)

        # 优化器与调度器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = None
        if config['use_lr_scheduler']:
            if config['reduce_on_plateau']:
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=config['plateau_factor'],
                    patience=config['plateau_patience'], verbose=True
                )
            else:
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma']
                )

        # 数据加载
        train_files = [os.path.join(config['data_dir'], f)
                      for f in os.listdir(config['data_dir']) if f.lower().endswith('.fits')]
        val_files = [os.path.join(config['val_dir'], f)
                    for f in os.listdir(config['val_dir']) if f.lower().endswith('.fits')]

        train_dataset = FITSDataset(
            file_list=train_files,
            noise_level_range=tuple(config['noise_levels']),
            augment=config.get('augment', False),
            resize_size=(config['input_size'][1], config['input_size'][2])
        )
        val_dataset = FITSDataset(
            file_list=val_files,
            noise_level_range=tuple(config['noise_levels']),
            augment=False,
            resize_size=(config['input_size'][1], config['input_size'][2])
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=config['num_workers'], pin_memory=True
        )

        # 日志与指标记录
        self.writer = SummaryWriter(self.log_dir)
        self.best_val_psnr = -float('inf')  # PSNR越大越好，初始值设为负无穷
        self.best_val_ssim = -float('inf')    # 最佳SSIM（越大越好）
        self.best_val_loss = -float('inf')     # 最佳损失（越小越好）
        self.best_epoch = 0                   # 最佳指标对应的epoch
        self.train_losses = []
        self.val_losses = []
        self.val_psnrs = []
        self.val_ssims = []

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        progress = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')

        for i, batch in enumerate(progress):
            noisy1 = batch['noisy1'].to(self.device)
            target = batch['raw'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(noisy1)
            loss = self.criterion(output, target)  # 损失函数计算（移除mask参数）
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})

            step = epoch * len(self.train_loader) + i
            self.writer.add_scalar('Train/BatchLoss', loss.item(), step)

        avg_loss = train_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        self.writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        psnrs, ssims = [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')):
                noisy1 = batch['noisy1'].to(self.device)
                raw = batch['raw'].to(self.device)

                # 前向传播：获取去噪结果 + 注意力图
                output, attention_maps = self.model.forward_with_attention(noisy1) 
                loss = self.criterion(output, raw)  # 损失函数计算（移除mask参数）
                val_loss += loss.item()

                # 全图PSNR计算
                mn, mx = raw.min(), raw.max()
                out_denorm = output * (mx - mn) + mn
                raw_denorm = raw * (mx - mn) + mn
                mse = torch.mean((out_denorm - raw_denorm) **2)
                psnr = 20 * torch.log10((mx - mn) / torch.sqrt(mse + 1e-8))
                psnrs.append(psnr.item())

                # 全图SSIM计算
                o_np = output[0, 0].clamp(0, 1).cpu().numpy()
                r_np = raw[0, 0].clamp(0, 1).cpu().numpy()
                ss = ssim_metric(o_np, r_np, data_range=1.0)
                ssims.append(ss)

                # 每10个epoch保存样本及可视化
                if epoch % 10 == 0 and i == 0:
                    self.save_val_samples(raw, noisy1, output, epoch)
                    self.plot_correlation(raw, output, epoch)
                    self.plot_power_spectrum(raw, noisy1, output, epoch)
                    # 新增：绘制注意力热力图
                    self.visualize_attention_heatmaps(raw, noisy1, output, attention_maps, epoch)

        # 计算平均指标
        avg_loss = val_loss / len(self.val_loader)
        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)

        # 更新最佳指标
        if avg_psnr > self.best_val_psnr:
            self.best_val_psnr = avg_psnr
            self.best_val_ssim = avg_ssim
            self.best_val_loss = avg_loss
            self.best_epoch = epoch

        # 自适应权重更新
        if self.config.get('adaptive_weights', False):
            spectral_weight, wavelet_weight = self.controller.update_weights(epoch, avg_psnr)
            self.criterion.set_weights(spectral_weight, wavelet_weight)

        self.val_losses.append(avg_loss)
        self.val_psnrs.append(avg_psnr)
        self.val_ssims.append(avg_ssim)

        # 记录到TensorBoard
        self.writer.add_scalar('Val/EpochLoss', avg_loss, epoch)
        self.writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Val/SSIM', avg_ssim, epoch)

        # 模型保存
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            torch.save(self.model.state_dict(),
                       os.path.join(self.config['save_dir'], 'best_model.pth'))
        if (epoch + 1) % self.config['save_freq'] == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(self.config['save_dir'], f'model_epoch_{epoch+1}.pth'))

        print(f'[Epoch {epoch}] Val Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}')

        # 调度器步骤
        if self.config['use_lr_scheduler'] and self.scheduler:
            if self.config['reduce_on_plateau']:
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

        return avg_loss, avg_psnr, avg_ssim

    def save_val_samples(self, raw, noisy, output, epoch):
        """保存验证样本：带噪输入 vs 干净GT vs 去噪输出"""
        out_dir = os.path.join(self.log_dir, 'val_samples')
        os.makedirs(out_dir, exist_ok=True)

        # 转换为numpy
        raw_np = raw[0, 0].cpu().numpy()
        noisy_np = noisy[0, 0].cpu().numpy()
        out_np = output[0, 0].cpu().numpy()

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(noisy_np, cmap='gray')
        plt.title('Noisy Input')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(raw_np, cmap='gray')
        plt.title('Clean Ground Truth')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(out_np, cmap='gray')
        plt.title('Denoised Output')
        plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(out_dir, f'epoch_{epoch}_denoise.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_correlation(self, target, output, epoch):
        """绘制输出与GT的相关性散点图"""
        tgt = target[0, 0].clamp(0, 1).cpu().numpy().flatten()
        out = output[0, 0].clamp(0, 1).cpu().numpy().flatten()

        corr_coef = np.corrcoef(tgt, out)[0, 1]
        r2 = r2_score(tgt, out)

        plt.figure(figsize=(6, 6))
        plt.scatter(tgt, out, alpha=0.3, s=1, c='blue')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
        plt.xlabel('Ground Truth')
        plt.ylabel('Denoised Output')
        plt.title(f'Correlation (r={corr_coef:.4f}, R²={r2:.4f})')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        save_path = os.path.join(self.correlation_dir, f'epoch_{epoch}_correlation.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_power_spectrum(self, target, noisy, output, epoch):
        """绘制功率谱：带噪输入 vs GT vs 去噪输出（修复布局警告）"""
        def compute_ps(img):
            fft = fftpack.fft2(img)
            fft_shift = fftpack.fftshift(fft)
            return np.log10(np.abs(fft_shift)** 2 + 1e-8)

        tgt_np = target[0, 0].cpu().numpy()
        nsy_np = noisy[0, 0].cpu().numpy()
        out_np = output[0, 0].cpu().numpy()

        ps_tgt = compute_ps(tgt_np)
        ps_nsy = compute_ps(nsy_np)
        ps_out = compute_ps(out_np)

        # 关键修改：创建图形时启用 constrained_layout
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        im1 = axs[0].imshow(ps_nsy, cmap='viridis')
        axs[0].set_title('Noisy Input Power Spectrum')
        axs[0].axis('off')

        im2 = axs[1].imshow(ps_tgt, cmap='viridis')
        axs[1].set_title('GT Power Spectrum')
        axs[1].axis('off')

        im3 = axs[2].imshow(ps_out, cmap='viridis')
        axs[2].set_title('Denoised Power Spectrum')
        axs[2].axis('off')

        # 添加颜色条（使用 fig.colorbar 并指定 ax 参数，让 constrained_layout 识别）
        cbar = fig.colorbar(im3, ax=axs, label='log10(Power)', shrink=0.8)

        # 移除 tight_layout 调用，改用 constrained_layout
        save_path = os.path.join(self.power_spectrum_dir, f'epoch_{epoch}_power_spectrum.png')
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def plot_metrics(self):
        """绘制训练过程中的指标曲线"""
        epochs = np.arange(len(self.train_losses))
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.log_dir, 'loss_curve.png'))
        plt.close()

        plt.figure()
        plt.plot(epochs, self.val_psnrs, label='Val PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.log_dir, 'psnr_curve.png'))
        plt.close()

        plt.figure()
        plt.plot(epochs, self.val_ssims, label='Val SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.log_dir, 'ssim_curve.png'))
        plt.close()

    def visualize_attention_heatmaps(self, raw, noisy, output, attention_maps, epoch):
        """修正版：空间热力图（归一化+高对比） + 小波通道柱状图"""
        if attention_maps is None:
            return  
    
        spatial_attn, wavelet_attn, alpha = attention_maps
        heatmap_dir = os.path.join(self.log_dir, 'attention_heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)

        # ========== 1. 空间注意力热力图（修正：归一化+高对比） ==========
        raw_np = raw[0, 0].cpu().numpy()  # 原图 [H, W]
        spatial_attn_np = spatial_attn.squeeze().cpu().numpy()  # [H, W]
    
        # 关键：归一化到 [0,1]，强制拉伸数值差异
        spatial_attn_norm = (spatial_attn_np - spatial_attn_np.min()) / (spatial_attn_np.max() - spatial_attn_np.min() + 1e-8)
    
        self._plot_spatial_heatmap(
            original=raw_np, 
            attn_map=spatial_attn_norm, 
            title=f'Spatial Attention (α={alpha.item():.2f})', 
            epoch=epoch, 
            save_dir=heatmap_dir
        )

        # ========== 2. 小波注意力（修正：通道级柱状图） ==========
        self._plot_wavelet_channel_attn(
            wavelet_attn=wavelet_attn, 
            alpha=alpha, 
            epoch=epoch, 
            save_dir=heatmap_dir
        )

    def _plot_spatial_heatmap(self, original, attn_map, title, epoch, save_dir):
        """空间热力图：高对比颜色+合理叠加"""
        plt.figure(figsize=(8, 6))
        # 热力图（turbo 色域对比度更高）
        im = plt.imshow(attn_map, cmap='turbo', vmin=0, vmax=1)  
        # 叠加原图（低透明度，避免掩盖）
        plt.imshow(original, cmap='gray', alpha=0.3)  
        plt.title(title)
        plt.colorbar(im, label='Normalized Attention')
        plt.axis('off')
        save_path = os.path.join(save_dir, f'epoch_{epoch}_spatial_heatmap.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_wavelet_channel_attn(self, wavelet_attn, alpha, epoch, save_dir):
        """小波注意力：通道级权重柱状图（空间无差异，改用通道可视化）"""
        wavelet_attn_np = wavelet_attn.squeeze().cpu().numpy()  # 形状 [C]（通道数）
        num_channels = wavelet_attn_np.shape[0]
    
        plt.figure(figsize=(10, 5))
        plt.bar(range(num_channels), wavelet_attn_np, color='royalblue')
        plt.xlabel('Channel Index')
        plt.ylabel('Attention Weight')
        plt.title(f'Wavelet Attention (α={alpha.item():.2f})')
        save_path = os.path.join(save_dir, f'epoch_{epoch}_wavelet_channel_attn.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    def save_denoised_fits(self):
        """保存去噪结果为FITS文件"""
        out_dir = os.path.join(self.log_dir, 'denoised_fits')
        os.makedirs(out_dir, exist_ok=True)
        self.model.eval()
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if count >= 20:
                    break
                noisy1 = batch['noisy1'].to(self.device)
                output = self.model(noisy1)
                arr = output[0, 0].cpu().numpy().astype(np.float32)
                fits.PrimaryHDU(arr).writeto(os.path.join(out_dir, f'denoised_{count:02d}.fits'), overwrite=True)
                count += 1

    def train(self):
        """启动训练流程"""
        for epoch in range(self.config['epochs']):
            self.train_epoch(epoch)
            self.validate(epoch)
        self.plot_metrics()
        self.save_denoised_fits()
        torch.save(self.model.state_dict(), os.path.join(self.config['save_dir'], 'final_model.pth'))
        self.writer.close()