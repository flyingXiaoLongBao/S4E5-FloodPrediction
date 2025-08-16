import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import os
import time

from dataset import DatasetTrain
from model import LinearBaselineModel
from utils.devices import get_available_device
from utils.plot import plot_training_validation_loss

# 定义BATCH_SIZE和EPOCH
# 根据测试，Apple M4 16GB设备可以支持高达4096的batch size
BATCH_SIZE = 64
EPOCH = 3
N_SPLITS = 5


class RMSELoss(nn.Module):
    """
    均方根误差损失函数，对小范围数据更敏感
    """

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return torch.sqrt(self.mse(output, target))


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练模型一个epoch
    """
    model.train()
    train_loss = 0.0

    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(features)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate_model(model, val_loader, criterion, device):
    """
    验证模型
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(val_loader):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def load_and_average_model_weights(model, device):
    """
    加载已有模型参数并取平均值
    
    Args:
        model: 要加载参数的模型
        device: 计算设备
        
    Returns:
        bool: 是否成功加载并平均了模型参数
    """
    # 查找所有模型文件
    model_paths = []
    for root, dirs, files in os.walk('result'):
        for file in files:
            if (file.startswith('best_model_fold_') or
                file.startswith('final_model_fold_')) and file.endswith('.pth'):
                model_paths.append(os.path.join(root, file))

    # 如果没有找到之前保存的模型，则使用当前模型
    if not model_paths:
        return False

    print(f"找到 {len(model_paths)} 个模型文件，进行参数平均")

    # 获取当前模型状态字典
    avg_state_dict = model.state_dict()

    # 对每个参数进行平均
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key], dtype=torch.float32)

    # 累加所有模型的参数
    for model_path in model_paths:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            for key in avg_state_dict.keys():
                if key in state_dict:
                    avg_state_dict[key] += state_dict[key].to(device)
        except Exception as e:
            print(f"加载模型 {model_path} 时出错: {e}")
            return False

    # 计算平均值
    for key in avg_state_dict.keys():
        avg_state_dict[key] /= len(model_paths)

    # 加载平均后的参数
    model.load_state_dict(avg_state_dict)
    print("模型参数平均完成")
    return True


def train_linear_model():
    """
    使用线性模型进行训练
    """
    # 获取可用设备
    device = get_available_device()
    print(f"使用设备: {device}")

    # 创建数据集
    dataset = DatasetTrain('data/train.csv')

    # 创建K折交叉验证
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_results = []

    # 创建结果目录
    os.makedirs('result', exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n开始训练第 {fold + 1}/{N_SPLITS} 折")

        # 创建数据加载器
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # 初始化模型
        model = LinearBaselineModel(input_dim=21).to(device)  # 20个原始特征+1个新特征

        # 尝试加载并平均已有模型权重
        load_and_average_model_weights(model, device)

        # 定义损失函数和优化器
        criterion = RMSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # 记录每折的最佳验证损失和损失历史
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        # 训练多个epoch
        for epoch in range(EPOCH):
            # 训练一个epoch
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            # 验证模型
            val_loss = validate_model(model, val_loader, criterion, device)

            # 更新学习率
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch [{epoch + 1}/{EPOCH}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, f'result/best_model_fold_{fold + 1}.pth')
                print(f"第 {fold + 1} 折的最佳模型已保存，验证损失: {best_val_loss:.6f}")

        # 保存每折的最终模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': EPOCH - 1,
            'best_val_loss': best_val_loss
        }, f'result/final_model_fold_{fold + 1}.pth')
        print(f"第 {fold + 1} 折的最终模型已保存")

        # 保存损失历史
        loss_history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        np.save(f'result/loss_history_fold_{fold + 1}.npy', loss_history)

        # 绘制并保存损失曲线
        plot_training_validation_loss(
            train_losses,
            val_losses,
            f'result/loss_curve_fold_{fold + 1}.png',
            f'第 {fold + 1} 折训练和验证损失曲线'
        )

        fold_results.append(best_val_loss)
        print(f"第 {fold + 1} 折训练完成，最佳验证损失: {best_val_loss:.6f}")

    # 计算并打印平均结果
    mean_val_loss = np.mean(fold_results)
    std_val_loss = np.std(fold_results)
    print(f"\nK折交叉验证结果:")
    print(f"平均验证损失: {mean_val_loss:.6f} ± {std_val_loss:.6f}")

    # 保存整体结果
    overall_results = {
        'fold_results': fold_results,
        'mean_val_loss': mean_val_loss,
        'std_val_loss': std_val_loss
    }
    np.save('result/overall_results.npy', overall_results)


if __name__ == '__main__':
    train_linear_model()
