import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
import math


class FloodPredictionModel(nn.Module):
    """
    用于洪水概率预测的多层感知机模型
    适用于中等规模的回归任务
    """
    
    def __init__(self, input_size=20, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        """
        初始化模型
        
        Args:
            input_size (int): 输入特征的数量
            hidden_sizes (list): 隐藏层的大小列表
            dropout_rate (float): Dropout比例，已从0.2提高到0.3
        """
        super(FloodPredictionModel, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (Tensor or list of Tensors): 输入特征张量或张量列表
            
        Returns:
            Tensor: 预测结果
        """
        if isinstance(x, list):
            x = torch.cat(x, dim=1)  # 将输入张量列表拼接
            
        # 确保输入是float32类型
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            
        return self.network(x).squeeze()


class FloodPredictionModelWithResidual(nn.Module):
    """
    带残差连接的洪水概率预测模型
    在深层网络中能更好地保留梯度信息
    """
    
    def __init__(self, input_size=20, hidden_sizes=[128, 128, 64, 64, 32], dropout_rate=0.3):
        """
        初始化模型
        
        Args:
            input_size (int): 输入特征的数量
            hidden_sizes (list): 隐藏层的大小列表
            dropout_rate (float): Dropout比例，已从0.2提高到0.3
        """
        super(FloodPredictionModelWithResidual, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.dropout = nn.Dropout(dropout_rate)
        
        # 构建残差块
        self.residual_blocks = nn.ModuleList()
        prev_size = hidden_sizes[0]
        
        for i, hidden_size in enumerate(hidden_sizes):
            self.residual_blocks.append(
                ResidualBlock(prev_size, hidden_size, dropout_rate)
            )
            prev_size = hidden_size
            
        # 输出层
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (Tensor or list of Tensors): 输入特征张量或张量列表
            
        Returns:
            Tensor: 预测结果
        """
        if isinstance(x, list):
            x = torch.cat(x, dim=1)  # 将输入张量列表拼接
            
        # 确保输入是float32类型
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        
        # 通过残差块
        for block in self.residual_blocks:
            x = block(x)
            
        # 输出层
        x = self.output_layer(x)
        return x.squeeze()


class FTTransformer(nn.Module):
    """
    FT-Transformer模型实现，用于结构化数据的分类和回归任务
    结合了Transformer架构和特征嵌入技术
    """
    
    def __init__(self, input_size=20, d_model=128, nhead=8, num_layers=3, dropout_rate=0.3):
        """
        初始化FT-Transformer模型
        
        Args:
            input_size (int): 输入特征的数量
            d_model (int): Transformer模型的维度
            nhead (int): 多头注意力机制的头数
            num_layers (int): Transformer编码器层的数量
            dropout_rate (float): Dropout比例，已从0.2提高到0.3
        """
        super(FTTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # 特征嵌入层 - 每个数值特征通过一个线性层映射到d_model维度
        self.feature_embedding = nn.Linear(1, d_model)
        
        # 类别特征嵌入层 (这里我们假设所有特征都是数值型的)
        # 如果有类别特征，可以添加nn.Embedding层
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加dropout
            nn.Linear(d_model, 1)
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (Tensor or list of Tensors): 输入特征张量或张量列表
            
        Returns:
            Tensor: 预测结果
        """
        if isinstance(x, list):
            x = torch.cat(x, dim=1)  # 将输入张量列表拼接
            
        # 确保输入是float32类型
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
        
        batch_size = x.size(0)
        
        # 将每个特征分离并嵌入到d_model维度
        # x的形状: (batch_size, input_size)
        # 转换为: (batch_size, input_size, 1)
        x = x.unsqueeze(-1)
        
        # 特征嵌入: (batch_size, input_size, d_model)
        x = self.feature_embedding(x)
        
        # 添加位置编码（这里使用可学习的位置编码）
        # 在FT-Transformer中，通常使用可学习的位置编码或不使用位置编码
        # 因为特征的顺序可能不具有实际意义
        
        # Transformer编码器: (batch_size, input_size, d_model)
        x = self.transformer_encoder(x)
        
        # 全局池化: (batch_size, d_model)
        x = torch.mean(x, dim=1)
        
        # 输出层: (batch_size, 1)
        x = self.output_layer(x)
        
        return x.squeeze()


class ResidualBlock(nn.Module):
    """
    残差块实现
    """
    
    def __init__(self, input_size, output_size, dropout_rate=0.3):
        """
        初始化残差块
        
        Args:
            input_size (int): 输入大小
            output_size (int): 输出大小
            dropout_rate (float): Dropout比例，已从0.2提高到0.3
        """
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        # 如果输入和输出大小不同，需要调整维度
        self.shortcut = nn.Linear(input_size, output_size) if input_size != output_size else None
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        residual = x
        
        out = self.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        
        # 如果需要，调整残差的维度
        if self.shortcut is not None:
            residual = self.shortcut(x)
            
        out += residual  # 残差连接
        out = self.relu(out)
        
        return out