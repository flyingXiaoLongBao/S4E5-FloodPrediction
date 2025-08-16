import torch.nn as nn
import lightgbm as lgb


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


class LightGBMComparisonModel:
    """
    LightGBM模型作为对比模型
    """

    def __init__(self,
                 objective: str = 'regression',
                 metric: str = 'rmse',
                 learning_rate: float = 0.05,
                 num_leaves: int = 64,
                 feature_fraction: float = 0.9,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 verbosity: int = -1,
                 random_state: int = 42):
        """
        初始化LightGBM对比模型
        
        Args:
            objective: 目标函数
            metric: 评估指标
            learning_rate: 学习率
            num_leaves: 叶子节点数
            feature_fraction: 特征采样比例
            bagging_fraction: 样本采样比例
            bagging_freq: 样本采样频率
            verbosity: 日志详细程度
            random_state: 随机种子
        """
        self.params = {
            'objective': objective,
            'metric': metric,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbosity': verbosity,
            'seed': random_state
        }
        self.model = None
        self.best_iteration = None

    def fit(self, X, y, X_val=None, y_val=None, num_boost_round=5000, early_stopping_rounds=500):
        """
        训练模型
        
        Args:
            X: 训练特征矩阵，形状为(n_samples, n_features)
            y: 训练目标值，形状为(n_samples,)
            X_val: 验证特征矩阵，形状为(n_samples, n_features)
            y_val: 验证目标值，形状为(n_samples,)
            num_boost_round: 最大迭代次数
            early_stopping_rounds: 早停轮数
        """
        # 创建训练数据集
        train_set = lgb.Dataset(X, label=y)
        
        # 如果提供了验证集，则创建验证数据集
        valid_sets = [train_set]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val)
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

    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
            特征重要性数组
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法训练模型")

        return self.model.feature_importance()
