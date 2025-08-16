import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from dataset import DatasetTest
from model import LinearBaselineModel
from utils.devices import get_available_device

# 模型参数
BATCH_SIZE = 64
N_SPLITS = 5


def average_model_weights(model, device):
    """
    查找并平均所有保存的模型权重
    
    Args:
        model: 模型实例
        device: 计算设备
        
    Returns:
        bool: 是否成功加载并平均了模型参数
    """
    # 查找所有模型文件
    model_paths = []
    for root, dirs, files in os.walk('result'):
        for file in files:
            if file.startswith('best_model_fold_') and file.endswith('.pth'):
                model_paths.append(os.path.join(root, file))
            elif file.startswith('final_model_fold_') and file.endswith('.pth'):
                model_paths.append(os.path.join(root, file))
    
    if not model_paths:
        print("未找到任何模型文件")
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
    
    # 将平均后的参数加载到模型中
    model.load_state_dict(avg_state_dict)
    print(f"成功对模型参数进行平均化处理")
    return True


def ensemble_predict(device):
    """
    集成预测：通过平均模型参数进行单次预测
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
    
    # 创建模型实例
    model = FTTransformer().to(device)
    
    # 平均所有模型的参数
    if not average_model_weights(model, device):
        raise ValueError("无法加载模型参数进行平均")
    
    # 设置模型为评估模式
    model.eval()
    
    # 存储预测结果
    predictions = []
    
    # 进行预测
    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())
    
    print(f"使用平均模型参数完成预测")
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': ids,
        'FloodProbability': np.array(predictions).flatten()
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


def predict_with_linear_model():
    """
    使用线性模型进行预测
    """
    # 获取可用设备
    device = get_available_device()
    print(f"使用设备: {device}")
    
    # 加载测试数据
    test_dataset = DatasetTest('data/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = LinearBaselineModel(input_dim=21).to(device)  # 20个原始特征+1个新特征
    
    # 加载并平均模型权重
    if not average_model_weights(model, device):
        print("无法加载模型权重，使用随机初始化权重进行预测")
    
    # 进行预测
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy().flatten())
    
    # 保存预测结果
    submission = pd.DataFrame({
        'id': range(1117957, 1117957 + len(predictions)),
        'FloodProbability': predictions
    })
    
    # 确保预测值在[0, 1]范围内
    submission['FloodProbability'] = np.clip(submission['FloodProbability'], 0, 1)
    
    submission.to_csv('result/submission_linear_model.csv', index=False)
    print("线性模型预测完成，结果已保存到 result/submission_linear_model.csv")


if __name__ == '__main__':
    predict_with_linear_model()