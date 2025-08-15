import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from dataset import DatasetTest
from model import FloodPredictionModel, FloodPredictionModelWithResidual
from utils.devices import get_available_device

# 模型参数
BATCH_SIZE = 64
N_SPLITS = 5


def ensemble_predict(device):
    """
    集成预测：读取所有折的模型，对预测结果取平均值
    """
    # 设置设备
    device = device
    
    # 加载测试数据
    test_dataset = DatasetTest('data/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    
    # 读取测试数据的ID（直接从原始CSV文件读取以确保准确性）
    test_data = pd.read_csv('data/test.csv')
    ids = test_data['id'].values
    
    # 验证数据一致性
    if len(ids) != len(test_dataset):
        raise ValueError(f"测试数据ID数量({len(ids)})与数据集长度({len(test_dataset)})不一致")
    
    # 存储所有模型的预测结果
    all_predictions = []
    
    # 加载每一折的模型并进行预测
    available_models = 0
    for fold in range(1, N_SPLITS + 1):
        model_path = f'result/best_model_fold_{fold}.pth'
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"警告：模型文件 {model_path} 不存在，跳过该模型")
            continue
            
        # 创建模型实例
        # net = FloodPredictionModel().to(device)
        model = FloodPredictionModelWithResidual().to(device)
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # 存储当前模型的预测结果
        fold_predictions = []
        
        # 进行预测
        with torch.no_grad():
            for features in test_loader:
                features = features.to(device)
                outputs = model(features)
                fold_predictions.extend(outputs.cpu().numpy())
        
        all_predictions.append(fold_predictions)
        available_models += 1
        print(f"模型 {fold} 预测完成")
    
    if available_models == 0:
        raise ValueError("没有找到任何模型文件进行预测")
    
    print(f"共使用 {available_models} 个模型进行集成预测")
    
    # 计算所有模型预测结果的平均值
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': ids,
        'FloodProbability': ensemble_predictions.flatten()
    })
    
    # 保存提交文件
    submission.to_csv('result/submission.csv', index=False)
    print(f"提交文件已保存至: result/submission.csv")
    print(f"预测样本数量: {len(submission)}")
    
    # 显示预测结果的统计信息
    print(f"预测结果统计:")
    print(f"  最小值: {submission['FloodProbability'].min():.4f}")
    print(f"  最大值: {submission['FloodProbability'].max():.4f}")
    print(f"  平均值: {submission['FloodProbability'].mean():.4f}")
    print(f"  标准差: {submission['FloodProbability'].std():.4f}")
    
    return submission


if __name__ == '__main__':
    # 设置设备
    device = get_available_device()
    print(f"使用设备: {device}")
    
    ensemble_predict(device)