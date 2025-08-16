import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import os
import time

from dataset import DatasetTrain
from model import FloodPredictionModelWithResidual, FloodPredictionModel, FTTransformer
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
        print("未找到任何已有模型参数，使用默认初始化")
        return False

    print(f"找到 {len(model_paths)} 个已有模型参数，进行平均化处理")

    # 获取当前模型状态字典
    avg_state_dict = model.state_dict()

    # 对每个参数进行平均
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key], dtype=torch.float32)

    # 累加所有模型的参数
    for model_path in model_paths:
        try:
            state_dict = torch.load(model_path, map_location=device)
            for key in avg_state_dict.keys():
                if key in state_dict:
                    avg_state_dict[key] += state_dict[key].to(device)
            print(f"成功加载模型参数: {model_path}")
        except Exception as e:
            print(f"加载模型参数失败: {model_path}, 错误: {e}")

    # 计算平均值
    for key in avg_state_dict.keys():
        avg_state_dict[key] /= len(model_paths)

    # 将平均后的参数加载到模型中
    model.load_state_dict(avg_state_dict)
    print("成功对所有模型参数进行平均化处理，作为初始化参数")
    return True


def train_model_with_kfold(device, full_dataset, resume_training=False):
    # 初始化KFold
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 存储每折的验证损失
    fold_results = []

    # 如果启用恢复训练，则创建一个模型实例用于加载平均权重
    if resume_training:
        print("加载并平均所有已有模型参数作为初始化参数")
        # 创建一个临时模型用于加载平均权重
        temp_model = FTTransformer().to(device)
        if load_and_average_model_weights(temp_model, device):
            # 保存平均后的权重状态，供每折训练使用
            avg_state_dict = temp_model.state_dict()
            print("模型参数已准备就绪")
        else:
            avg_state_dict = None
            print("使用默认初始化参数")
    else:
        avg_state_dict = None

    # 进行K折交叉验证
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"正在训练第 {fold + 1}/{N_SPLITS} 折")

        # 创建训练和验证子集
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # 创建数据加载器，添加num_workers和pin_memory优化MPS性能
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2)

        # 定义模型
        # net = FloodPredictionModel().to(device)
        # net = FloodPredictionModelWithResidual().to(device)
        net = FTTransformer().to(device)  # 使用新的FT-Transformer模型

        # 如果启用了恢复训练并且成功加载了平均权重，则应用到当前模型
        if resume_training and avg_state_dict is not None:
            net.load_state_dict(avg_state_dict)
            print(f"第 {fold + 1} 折已加载平均模型参数作为初始化参数")

        # 定义损失函数、优化器、学习率调度
        criterion = RMSELoss().to(device)  # 使用RMSE损失函数替代MSE
        # 降低学习率，从0.001降到0.0001，并增加L2正则化权重从1e-5到1e-4
        optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
        # 调整学习率调度器参数，减小patience并降低factor
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3)

        # 记录最佳验证损失，初始化为无穷大确保第一轮训练后能保存模型
        best_val_loss = float('inf')

        # 定义数组保存每轮训练的损失
        train_losses = []
        val_losses = []

        # 训练循环
        for epoch in range(EPOCH):
            # 训练阶段
            train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)

            # 验证阶段
            val_loss = validate_model(net, val_loader, criterion, device)
            val_losses.append(val_loss)

            # 更新学习率
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存模型到result目录，使用原始文件名格式
                model_filename = f'result/best_model_fold_{fold + 1}.pth'
                torch.save(net.state_dict(), model_filename)
                print(f'  新的最佳模型已保存: {model_filename}')

            print(f'Fold [{fold + 1}/{N_SPLITS}], Epoch [{epoch + 1}/{EPOCH}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存每折的损失历史到result目录
        loss_history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        torch.save(loss_history, f'result/loss_history_fold_{fold + 1}.pth')

        # 绘制并保存损失曲线图
        plot_title = f'Fold {fold + 1} Training and Validation Loss'
        save_path = f'result/loss_curve_fold_{fold + 1}.png'
        plot_training_validation_loss(train_losses, val_losses, plot_title, save_path)

        fold_results.append(best_val_loss)
        print(f'第 {fold + 1} 折最佳验证损失: {best_val_loss:.4f}')

    print(f'K折交叉验证完成')
    print(f'各折验证损失: {fold_results}')
    print(f'平均验证损失: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')

    return fold_results


if __name__ == '__main__':
    # 使用GPU训练
    device = get_available_device()
    print(f"使用设备: {device}")

    # 定义数据
    full_dataset = DatasetTrain('data/train.csv')

    # 确保result目录存在
    os.makedirs('result', exist_ok=True)

    # 执行K折交叉验证，启用恢复训练功能
    fold_results = train_model_with_kfold(device, full_dataset, resume_training=False)

    print("所有模型、损失历史和损失曲线图已保存到 result 目录中，可用于后续推理时取平均值")
