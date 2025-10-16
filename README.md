# Dual-Domain-Attention-Network
# 基于改进UNet的图像去噪模型

该项目实现了一个结合小波变换与注意力机制的UNet去噪模型，支持自适应损失权重调整，特别适用于处理FITS格式的科学图像，NVST天文光谱数据。模型通过混合域注意力机制和多损失函数优化，在保持图像细节的同时有效抑制噪声。


## 项目概述

本项目的核心是一个改进的UNet架构，主要特点包括：
- 融合空间注意力与小波频域注意力的混合注意力机制
- 自适应调整损失函数权重的控制器
- 支持多种损失函数组合（MSE + 谱平滑损失 + 小波损失）
- 针对FITS格式数据的专用处理流程
- 完整的消融实验框架，便于验证各模块有效性


## 环境配置

### 依赖库
```bash
# 核心依赖
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.5
pywt>=1.1.1  # 小波变换
opencv-python>=4.5.3  # 图像处理
astropy>=4.2  # FITS文件处理
scikit-image>=0.18.1  # 图像评估指标
scipy>=1.6.0  # 傅里叶变换等
matplotlib>=3.3.4  # 可视化
tqdm>=4.62.0  # 进度条
pandas>=1.3.0  # 消融实验结果处理
```

## 项目结构

```
├── unet.py               # 核心网络结构（含注意力机制）
├── losses.py             # 损失函数定义（MSE+谱平滑+小波损失）
├── adaptive_control.py   # 自适应损失权重控制器
├── dataset.py            # FITS数据集加载与预处理
├── train.py              # 训练与验证流程
├── config.py             # 基础配置参数
├── config_ablation.py    # 消融实验配置
├── run_ablation.py       # 运行消融实验
├── main.py               # 主程序入口
```

### 关键模块说明
1. **网络结构**（`unet.py`）
   - `UnifiedDenoisingUNet`：基础UNet架构
   - `HybridAttention`：融合空间注意力（`SpatialAttention`）与小波频域注意力（`WaveletAttention`）
   - 支持双线性插值或转置卷积的上采样方式

2. **损失函数**（`losses.py`）
   - `CombinedLoss`：组合损失函数容器
   - `SpectralSmoothnessLoss`：谱向平滑损失（抑制高频噪声）
   - `WaveletLoss`：小波域损失（匹配高频子带分布）

3. **自适应控制**（`adaptive_control.py`）
   - 根据验证集PSNR动态调整损失函数权重
   - 性能提升时增强正则化，性能下降时减弱正则化

4. **数据处理**（`dataset.py`）
   - 加载FITS格式图像并预处理
   - 支持随机高斯噪声添加与数据增强
   - 自动归一化到[0,1]范围


## 使用说明

### 基础训练
1. 修改`config.py`配置参数（数据路径、训练轮次、学习率等）
2. 运行主程序：
```bash
python main.py
```

### 消融实验
1. 配置`config_ablation.py`中的实验组合
2. 运行消融实验脚本：
```bash
python run_ablation.py
```
实验结果将保存为CSV文件和对比图表（PSNR/SSIM/训练时间）


## 可视化结果

训练过程中会自动生成以下可视化内容（保存于`logs`目录）：
- 去噪效果对比（输入噪声图 vs 干净标签 vs 模型输出）
- 注意力热力图（空间注意力分布与小波通道权重）
- 功率谱对比（分析频域去噪效果）
- 相关性散点图（输出与标签的线性相关性）
- 训练指标曲线（损失、PSNR、SSIM随 epoch 变化）


## 模型特点

1. **混合域注意力**：同时关注空间关键区域与频域显著特征
2. **自适应优化**：动态调整损失权重，平衡拟合与正则化
3. **多尺度特征融合**：UNet架构的编码器-解码器结构有效捕捉多尺度信息
4. **科学数据适配**：专为FITS格式设计，支持天文等领域的专业数据处理




