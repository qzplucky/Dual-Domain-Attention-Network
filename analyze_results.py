# analyze_ablation.py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def main():
    # 加载结果
    with open('ablation_results.json') as f:
        results = json.load(f)
    
    # 转换为DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # PSNR对比
    plt.subplot(2, 2, 1)
    df['best_psnr'].sort_values().plot(kind='barh', color='skyblue')
    plt.title('PSNR Comparison (dB)')
    plt.xlabel('PSNR (dB)')
    
    # SSIM对比
    plt.subplot(2, 2, 2)
    df['best_ssim'].sort_values().plot(kind='barh', color='salmon')
    plt.title('SSIM Comparison')
    plt.xlabel('SSIM')
    
    # 损失对比
    plt.subplot(2, 2, 3)
    df['best_val_loss'].sort_values(ascending=False).plot(kind='barh', color='lightgreen')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Loss')
    
    # 时间对比
    plt.subplot(2, 2, 4)
    (df['training_time']/60).sort_values().plot(kind='barh', color='gold')
    plt.title('Training Time Comparison')
    plt.xlabel('Time (minutes)')
    
    plt.tight_layout()
    plt.savefig('ablation_comparison.png')
    plt.close()
    
    # 相关性热力图
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Metrics Correlation')
    plt.tight_layout()
    plt.savefig('metrics_correlation.png')
    plt.close()
    
    # 生成详细报告
    report = f"# Ablation Study Report\n\n"
    report += "## Summary Statistics\n"
    report += df.describe().to_markdown() + "\n\n"
    
    report += "## Detailed Results\n"
    report += df.to_markdown() + "\n\n"
    
    report += "## Visualizations\n"
    report += "\n\n"
    report += "\n"
    
    with open('ablation_report.md', 'w') as f:
        f.write(report)
    
    print("Analysis completed! Report saved as ablation_report.md")

if __name__ == "__main__":
    main()