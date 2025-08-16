import numpy as np
import pandas as pd
import os
import lightgbm as lgb

from dataset import DatasetTest

# 模型参数
N_SPLITS = 5


def load_lgbm_models():
    """
    加载所有保存的LightGBM模型
    
    Returns:
        list: LightGBM模型列表
    """
    models = []
    model_files = []
    
    # 查找所有模型文件
    for root, dirs, files in os.walk('result'):
        for file in files:
            if file.startswith('lgbm_model_fold_') and file.endswith('.txt'):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        raise FileNotFoundError("未找到任何LightGBM模型文件")
    
    print(f"找到 {len(model_files)} 个模型文件")
    
    # 加载所有模型
    for model_file in model_files:
        model = lgb.Booster(model_file=model_file)
        models.append(model)
        print(f"已加载模型: {model_file}")
    
    return models


def predict_with_lgbm_models():
    """
    使用LightGBM模型进行预测
    """
    print("开始使用LightGBM模型进行预测...")
    
    # 加载测试数据
    test_dataset = DatasetTest('data/test.csv')
    
    # 获取全部测试数据
    all_features = []
    for i in range(len(test_dataset)):
        features = test_dataset[i]
        all_features.append(features.numpy())
    
    X_test = np.array(all_features)
    print(f"测试集大小: {X_test.shape}")
    
    # 加载模型
    models = load_lgbm_models()
    print(f"成功加载 {len(models)} 个模型")
    
    # 进行预测
    predictions_list = []
    for i, model in enumerate(models):
        pred = model.predict(X_test)
        predictions_list.append(pred)
        print(f"模型 {i+1} 预测完成")
    
    # 对所有模型的预测结果取平均
    predictions = np.mean(predictions_list, axis=0)
    print("已完成模型集成预测")
    
    # 保存预测结果
    submission = pd.DataFrame({
        'id': range(1117957, 1117957 + len(predictions)),
        'FloodProbability': predictions
    })
    
    # 确保预测值在[0, 1]范围内
    submission['FloodProbability'] = np.clip(submission['FloodProbability'], 0, 1)
    
    submission_path = 'result/submission_lgbm_model.csv'
    submission.to_csv(submission_path, index=False)
    print(f"LightGBM模型预测完成，结果已保存到 {submission_path}")


def main():
    """
    主函数
    """
    try:
        predict_with_lgbm_models()
        print("\nLightGBM模型推理完成!")
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()