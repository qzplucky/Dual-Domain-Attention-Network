# run_ablation.py
import copy
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from config import train_config
from train import Train

# 定义你想做消融的组件，以及对应要在 config 中关闭的项
ablations = {
    'baseline': {},  # 原始配置
    'no_wavelet_loss':      {'use_wavelet_loss': False},
    'no_spectral_loss':     {'use_spectral_loss': False},
    'no_attention':         {'use_attention': False},  # 需在 config 和 model 中响应地支持
    'no_adaptive_weights':  {'adaptive_weights': False},
}

results = []

for name, delta in ablations.items():
    print(f'\n*** Running ablation: {name} ***')

    start_time = time.time()  # ⏱ 开始计时

    # 1. 复制配置并应用修改
    cfg = copy.deepcopy(train_config)
    cfg.update(delta)

    # 2. 设置唯一保存路径
    cfg['save_dir'] = os.path.join('checkpoints', name)
    cfg['log_dir']  = os.path.join('logs', name)
    os.makedirs(cfg['save_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    # 3. 启动训练
    trainer = Train(cfg)
    trainer.train()

    end_time = time.time()  # ⏱ 结束计时
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'

    print(f'[{name}] Training Time: {formatted_time}')

    # 4. 记录结果
    final_psnr = trainer.val_psnrs[-1]
    final_ssim = trainer.val_ssims[-1]
    final_loss = trainer.val_losses[-1]

    results.append({
        'experiment': name,
        'psnr': final_psnr,
        'ssim': final_ssim,
        'loss': final_loss,
        'time': formatted_time,
        'seconds': elapsed  # 用于画图排序
    })

# 5. 结果保存为 CSV
df = pd.DataFrame(results)
df.to_csv('ablation_results.csv', index=False)
print('\nAll results:')
print(df)

# 6. 绘图：PSNR 对比
plt.figure()
plt.bar(df['experiment'], df['psnr'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Final Val PSNR (dB)')
plt.title('Ablation Study: PSNR Comparison')
plt.tight_layout()
plt.savefig('ablation_psnr.png')
plt.close()

# 7. 绘图：SSIM 对比
plt.figure()
plt.bar(df['experiment'], df['ssim'], color='lightgreen')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Final Val SSIM')
plt.title('Ablation Study: SSIM Comparison')
plt.tight_layout()
plt.savefig('ablation_ssim.png')
plt.close()

# 8. 绘图：耗时对比
plt.figure()
plt.bar(df['experiment'], df['seconds'], color='salmon')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Training Time (seconds)')
plt.title('Ablation Study: Training Time Comparison')
plt.tight_layout()
plt.savefig('ablation_time.png')
plt.close()
