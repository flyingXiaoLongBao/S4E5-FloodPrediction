import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
import math
import lightgbm as lgb
import numpy as np


class LinearBaselineModel(nn.Module):
    """
    线性模型作为基准模型
    """
    def __init__(self, input_dim: int = 21):
        """
        初始化线性基准模型
        
        Args:
            input_dim: 输入特征维度，包括新增的Sum72_75特征
        """
        super(LinearBaselineModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征张量，形状为(batch_size, input_dim)
            
        Returns:
            输出预测值，形状为(batch_size, 1)
        """
        x = self.linear(x)
        # 保持输出形状为(batch_size, 1)
        return x.view(-1, 1)


class EnhancedLightGBMModel:
    """
    增强版LightGBM模型
    """
    def __init__(self,
                 objective: str = 'regression',
                 metric: str = 'rmse',
                 boosting: str = 'gbdt',
                 learning_rate: float = 0.05,
                 num_leaves: int = 64,
                 max_depth: int = -1,
                 min_data_in_leaf: int = 20,
                 feature_fraction: float = 0.9,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 min_gain_to_split: float = 0.0,
                 lambda_l1: float = 0.0,
                 lambda_l2: float = 0.0,
                 verbosity: int = -1,
                 random_state: int = 42):
        """
        初始化增强版LightGBM模型
        
        Args:
            objective: 目标函数
            metric: 评估指标
            boosting: 提升类型
            learning_rate: 学习率
            num_leaves: 叶子节点数
            max_depth: 最大深度
            min_data_in_leaf: 叶子节点最少样本数
            feature_fraction: 特征采样比例
            bagging_fraction: 样本采样比例
            bagging_freq: 样本采样频率
            min_gain_to_split: 分裂最小增益
            lambda_l1: L1正则化
            lambda_l2: L2正则化
            verbosity: 日志详细程度
            random_state: 随机种子
        """
        self.params = {
            'objective': objective,
            'metric': metric,
            'boosting': boosting,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'min_gain_to_split': min_gain_to_split,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'verbosity': verbosity,
            'seed': random_state
        }
        self.model = None
        self.best_iteration = None
        
    def fit(self, X, y, X_val=None, y_val=None, num_boost_round=5000, early_stopping_rounds=500, 
            categorical_feature=None):
        """
        训练模型
        
        Args:
            X: 训练特征矩阵，形状为(n_samples, n_features)
            y: 训练目标值，形状为(n_samples,)
            X_val: 验证特征矩阵，形状为(n_samples, n_features)
            y_val: 验证目标值，形状为(n_samples,)
            num_boost_round: 最大迭代次数
            early_stopping_rounds: 早停轮数
            categorical_feature: 分类特征索引列表
        """
        # 创建训练数据集
        train_set = lgb.Dataset(X, label=y, categorical_feature=categorical_feature)
        
        # 如果提供了验证集，则创建验证数据集
        valid_sets = [train_set]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feature)
            valid_sets.append(val_set)
            valid_names.append('valid')
        
        # 训练模型
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.best_iteration = self.model.best_iteration
        
    def predict(self, X):
        """
        预测
        
        Args:
            X: 特征矩阵，形状为(n_samples, n_features)
            
        Returns:
            预测值，形状为(n_samples,)
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法训练模型")
            
        return self.model.predict(X, num_iteration=self.best_iteration)
        
    def get_feature_importance(self, importance_type='split'):
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 ('split' 或 'gain')
            
        Returns:
            特征重要性数组
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法训练模型")
            
        return self.model.feature_importance(importance_type=importance_type)


class DeepFloodPredictionModel(nn.Module):
    """
    深度洪水预测模型
    """
    def __init__(self, input_dim: int = 21, d_model: int = 128, num_layers: int = 4, dropout: float = 0.3):
        """
        初始化深度洪水预测模型
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            num_layers: 网络层数
            dropout: Dropout率
        """
        super(DeepFloodPredictionModel, self).__init__()
        
        # 输入层
        self.input_layer = nn.Linear(input_dim, d_model)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_layers)
        ])
        
        # Batch归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(d_model) for _ in range(num_layers)
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, 1)
        
        # 激活函数
        self.activation = nn.ReLU()
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征张量，形状为(batch_size, input_dim)
            
        Returns:
            输出预测值，形状为(batch_size, 1)
        """
        # 输入层
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 隐藏层
        for layer, batch_norm in zip(self.hidden_layers, self.batch_norms):
            residual = x
            x = layer(x)
            x = batch_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            # 添加残差连接
            x = x + residual
            
        # 输出层
        x = self.output_layer(x)
        return x.view(-1, 1)