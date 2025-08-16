import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os

from model import LightGBMComparisonModel
from dataset import DatasetTrain

# 定义训练参数
N_SPLITS = 5
RANDOM_STATE = 42


def train_lgbm_model():
    """
    使用LightGBM模型进行训练
    """
    print("开始训练LightGBM模型...")
    
    # 创建数据集
    dataset = DatasetTrain('data/train.csv')
    
    # 获取全部数据
    all_features = []
    all_labels = []
    for i in range(len(dataset)):
        features, label = dataset[i]
        all_features.append(features.numpy())
        all_labels.append(label.item())
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"数据集大小: {X.shape}")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    # 创建K折交叉验证
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_results = []
    
    # 创建结果目录
    os.makedirs('result', exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n开始训练第 {fold + 1}/{N_SPLITS} 折")
        
        # 分割训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 初始化模型
        model = LightGBMComparisonModel(
            objective='regression',
            metric='rmse',
            learning_rate=0.05,
            num_leaves=64,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbosity=-1,
            random_state=RANDOM_STATE
        )
        
        # 训练模型
        model.fit(
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            num_boost_round=5000,
            early_stopping_rounds=500
        )
        
        # 验证模型
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # 计算RMSE
        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        
        fold_results.append(val_rmse)
        
        print(f'第 {fold + 1} 折训练完成')
        print(f'  训练集 RMSE: {train_rmse:.6f}')
        print(f'  验证集 RMSE: {val_rmse:.6f}')
        print(f'  最佳迭代次数: {model.best_iteration}')
        
        # 保存模型
        model_path = f'result/lgbm_model_fold_{fold+1}.txt'
        model.model.save_model(model_path)
        print(f'  模型已保存到: {model_path}')
        
        # 保存特征重要性
        feature_importance = model.get_feature_importance()
        np.save(f'result/feature_importance_fold_{fold+1}.npy', feature_importance)
        
    # 计算并打印平均结果
    mean_val_rmse = np.mean(fold_results)
    std_val_rmse = np.std(fold_results)
    print(f"\nK折交叉验证结果:")
    print(f"平均验证集 RMSE: {mean_val_rmse:.6f} ± {std_val_rmse:.6f}")
    
    # 保存整体结果
    overall_results = {
        'fold_results': fold_results,
        'mean_val_rmse': mean_val_rmse,
        'std_val_rmse': std_val_rmse
    }
    np.save('result/lgbm_overall_results.npy', overall_results)
    
    return model  # 返回最后一个模型


def main():
    """
    主函数
    """
    try:
        model = train_lgbm_model()
        print("\nLightGBM模型训练完成!")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()