import pandas as pd
import numpy as np


def add_new_feature(file_path, output_path=None):
    """
    为数据集添加新特征列
    
    新特征：对原本的20个特征值求和，如果和为72、73、74、75则取1，否则取0
    
    参数:
    file_path: 原始数据文件路径
    output_path: 输出文件路径，如果为None则直接修改原文件
    
    返回:
    None
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 检查是否已经存在Sum72_75列
    if 'Sum72_75' in df.columns:
        print(f"文件 {file_path} 已经包含 'Sum72_75' 列，跳过处理")
        return
    
    # 获取特征列（排除id列，剩下的20列就是特征）
    # 训练集有22列（id + 20个特征 + 目标值），测试集有21列（id + 20个特征）
    feature_columns = df.columns[1:21]  # 固定选择第2到21列作为特征列
    
    # 确保我们有20个特征列
    assert len(feature_columns) == 20, f"期望有20个特征列，但找到了{len(feature_columns)}个"
    
    # 计算20个特征的和
    row_sums = df[feature_columns].sum(axis=1)
    
    # 创建新特征：如果和为72、73、74、75则为1，否则为0
    new_feature = ((row_sums == 72) | (row_sums == 73) | (row_sums == 74) | (row_sums == 75)).astype(int)
    
    # 插入新特征列到正确位置（在特征之后，目标值之前，或者在最后）
    insert_position = 21  # 在20个特征之后插入
    df.insert(insert_position, 'Sum72_75', new_feature)
    
    # 保存文件
    output_file = output_path if output_path else file_path
    df.to_csv(output_file, index=False)
    print(f"已处理文件: {file_path}")
    print(f"新增特征列 'Sum72_75' 已添加")
    print(f"新特征中值为1的样本数: {new_feature.sum()}")


def modify_train_and_test_data():
    """
    修改训练集和测试集，添加新特征
    """
    # 修改训练数据
    train_file = 'data/train.csv'
    add_new_feature(train_file)
    
    # 修改测试数据
    test_file = 'data/test.csv'
    add_new_feature(test_file)


if __name__ == '__main__':
    modify_train_and_test_data()