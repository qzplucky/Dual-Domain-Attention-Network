# main.py

import torch
from train import Train
from config import train_config

def main():
    # 打印配置信息
    print("Training Configuration:")
    for k, v in train_config.items():
        print(f"{k}: {v}")

    # 初始化训练器
    trainer = Train(train_config)

    # 开始训练
    trainer.train()

    print("Training completed!")

if __name__ == "__main__":
    main()
