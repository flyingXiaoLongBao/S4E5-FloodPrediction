import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_missing_values(csv_file, save_dir='../result', filename='缺失值统计.png'):
    """
    统计训练集中21个特征各自的缺失值，并将数据绘制在一张图片上保存
    
    Args:
        csv_file (str): 训练集CSV文件路径
        save_dir (str): 保存图片的目录
        filename (str): 保存的图片文件名
    """
    # 读取数据，将<unset>等特殊值视为缺失值
    data = pd.read_csv(csv_file, na_values=['NaN', 'null', ''])

    # 获取特征列（排除id和FloodProbability）
    feature_columns = data.columns[1:-1]  # 排除第一列(id)和最后一列(FloodProbability)

    # 统计每个特征的缺失值数量
    missing_values = data[feature_columns].isnull().sum()

    # 创建图表
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(feature_columns)), missing_values.values, color='skyblue')

    # 设置图表标题和标签
    plt.title('各特征缺失值统计', fontsize=16)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('缺失值数量', fontsize=12)

    # 设置x轴标签
    plt.xticks(range(len(feature_columns)), feature_columns, rotation=45, ha='right')

    # 在每个柱子上显示具体数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # 调整布局防止标签被截断
    plt.tight_layout()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存图片
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"缺失值统计图已保存至: {save_path}")

    # 打印统计信息
    print("\n各特征缺失值详情:")
    for feature, missing_count in zip(feature_columns, missing_values):
        print(f"{feature}: {missing_count}")

    return missing_values


def plot_training_validation_loss(train_loss, val_loss, title, save_path):
    """
    绘制训练和验证损失随epoch的变化图，并保存到指定路径
    
    Args:
        train_loss (list or array): 训练损失数组
        val_loss (list or array): 验证损失数组
        title (str): 图表标题
        save_path (str): 保存图片的路径
    """
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"损失曲线图已保存至: {save_path}")