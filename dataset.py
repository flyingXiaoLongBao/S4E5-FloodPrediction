import torch
from torch.utils.data import Dataset
import pandas as pd


class DatasetTrain(Dataset):
    def __init__(self, csv_file):
        """
        初始化训练数据集
        
        Args:
            csv_file (string): 包含训练数据的csv文件路径
        """
        self.data = pd.read_csv(csv_file)
        self.dtype = torch.float32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取特征（排除id和最后一列FloodProbability）
        features = self.data.iloc[idx, 1:-1].values.astype('float32')

        # 获取标签（最后一列FloodProbability）
        label = self.data.iloc[idx, -1]

        # 转换为张量，但不立即分配到设备
        features = torch.tensor(features, dtype=self.dtype)
        label = torch.tensor(label, dtype=self.dtype)

        return features, label


class DatasetTest(Dataset):
    def __init__(self, csv_file):
        """
        初始化测试数据集
        
        Args:
            csv_file (string): 包含测试数据的csv文件路径
        """
        self.data = pd.read_csv(csv_file)
        self.dtype = torch.float32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取特征（排除id列）
        features = self.data.iloc[idx, 1:].values.astype('float32')

        # 转换为张量，但不立即分配到设备
        features = torch.tensor(features, dtype=self.dtype)

        return features
