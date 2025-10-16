import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

# ------------------- 基础模块 -------------------
class DoubleConv(nn.Module):
    """(Conv->BN->ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    """MaxPool + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch,out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    """Upsample + DoubleConv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2)-x1.size(2); dx = x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2,x1], dim=1))

class OutConv(nn.Module):
    """1x1 卷积输出"""
    def __init__(self, in_ch, out_ch): super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,1)
    def forward(self,x): return self.conv(x)

# ------------------- 小波频域注意力 -------------------
class WaveletAttention(nn.Module):
    """基于 2D 小波变换的频域注意力"""
    def __init__(self, in_c, reduction=16, wavelet='db8'):
        super().__init__()
        self.wavelet = wavelet
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_c, in_c//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c//reduction, in_c, bias=False),
            nn.Sigmoid()
        )
        self.attn_map = None  # 新增：保存小波注意力权重图（C×1×1，通道级）
        
    def forward(self, x):
        # 提取低频子带 LL
        coeffs = []
        for b in range(x.size(0)):
            ll_channels = []
            for c in range(x.size(1)):
                arr = x[b,c].cpu().detach().numpy()
                LL, _ = pywt.dwt2(arr, self.wavelet)
                ll_channels.append(torch.from_numpy(LL))
            # stacking
            ll = torch.stack(ll_channels, dim=0)
            coeffs.append(ll)
        LL = torch.stack(coeffs, dim=0).to(x.device)
        # 通道注意力计算
        y = self.avg_pool(LL).view(x.size(0), x.size(1))
        w = self.fc(y).view(x.size(0), x.size(1),1,1)
        self.attn_map = w  # 保存通道级注意力图（后续扩展为空间尺寸）
        # 应用注意力
        return x * w

class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size,padding=kernel_size//2,bias=False)
        self.sigmoid=nn.Sigmoid()
        self.attn_map = None  # 新增：保存空间注意力权重图（1×H×W）
    def forward(self,x):
        avg=torch.mean(x,1,keepdim=True)
        maxv,_=torch.max(x,1,keepdim=True)
        attn=self.sigmoid(self.conv(torch.cat([avg,maxv],1)))
        self.attn_map = attn  # 保存注意力图（后续可提取）
        print(f"空间注意力统计：均值={attn.mean().item():.4f}, 最大值={attn.max().item():.4f}, 最小值={attn.min().item():.4f}")
        return x*attn

class HybridAttention(nn.Module):
    """混合注意力：空间 + 小波频域"""
    def __init__(self, in_c, reduction=16, wavelet='db8'):
        super().__init__()
        self.spatial = SpatialAttention()
        self.wavelet = WaveletAttention(in_c, reduction, wavelet)
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self,x):
        spatial_out = self.spatial(x)  # 空间注意力加权后的特征
        wavelet_out = self.wavelet(x)  # 小波注意力加权后的特征
        alpha = torch.sigmoid(self.weight)
        out = alpha * spatial_out + (1 - alpha) * wavelet_out

        return out, (self.spatial.attn_map, self.wavelet.attn_map, alpha)

# ------------------- 单分支 UNet（全区域处理） -------------------
class UnifiedDenoisingUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, wavelet='db8', use_attention=True):
        super().__init__()
        factor = 2 if bilinear else 1
        
        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024//factor)
        
        # 解码器
        self.up1 = Up(1024, 512//factor, bilinear)
        self.up2 = Up(512, 256//factor, bilinear)
        self.up3 = Up(256, 128//factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 注意力机制（应用于全区域）
        self.use_attention = use_attention
        if use_attention:
            self.attention = HybridAttention(64, reduction=16, wavelet=wavelet)
        
        # 输出层
        self.out = OutConv(64, n_classes)
        
    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码路径
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        feats = self.up4(u3, x1)
        
        # 应用注意力（只取特征张量，忽略注意力图）
        if self.use_attention:
            feats, _ = self.attention(feats)  # 关键：仅保留特征
        
        # 输出预测结果
        return self.out(feats)

    # 2. 带注意力图的前向传播（可视化/分析用）
    def forward_with_attention(self, x):
        # 编码/解码路径与 forward 完全一致（确保逻辑同步）
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        feats = self.up4(u3, x1)
        
        # 应用注意力（同时保留特征和注意力图）
        attention_maps = None
        if self.use_attention:
            feats, attention_maps = self.attention(feats)  # 保留注意力图
        
        # 输出预测结果 + 注意力图
        return self.out(feats), attention_maps