import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from dataset import DatasetTest
from model import DeepFloodPredictionModel
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
            if file.startswith('best_model_deep_fold_') and file.endswith('.pth'):
                model_paths.append(os.path.join(root, file))
            elif file.startswith('final_model_deep_fold_') and file.endswith('.pth'):
                model_paths.append(os.path.join(root, file))
    
    if not model_paths:
        print("未找到任何深度学习模型文件")
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


def predict_with_deep_model():
    """
    使用深度学习模型进行预测
    """
    # 获取可用设备
    device = get_available_device()
    print(f"使用设备: {device}")
    
    # 加载测试数据
    test_dataset = DatasetTest('data/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = DeepFloodPredictionModel(input_dim=21, d_model=128, num_layers=4, dropout=0.3).to(device)
    
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
    
    submission_path = 'result/submission_deep_model.csv'
    submission.to_csv(submission_path, index=False)
    print(f"深度学习模型预测完成，结果已保存到 {submission_path}")


if __name__ == '__main__':
    predict_with_deep_model()