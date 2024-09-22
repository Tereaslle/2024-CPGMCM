import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np

if __name__ == '__main__':
    # 定义文件路径
    file_path = r"../appendix_1.csv"

    # pandas.read_excel('file path')读取Excel文件
    # pandas.read_csv('file path')读取csv文件
    df = pd.read_csv(file_path)
    # 修改列名，inplace=True表示在原数据上修改
    df.rename(columns={"温度，oC": "温度",
                       "频率，Hz": "频率",
                       "磁芯损耗，w/m3": "磁芯损耗",
                       "0（磁通密度B，T）": "0"}, inplace=True)
    # 只读取磁通密度曲线作为特征
    features = df.iloc[:, 4:]  # 第5列到最后一列为磁通密度曲线

    # 计算每个样本的统计特征：均值、标准差、最大值、最小值、幅度、能量、偏度、峰度
    features['mean'] = features.mean(axis=1)
    features['std'] = features.std(axis=1)
    features['max'] = features.max(axis=1)
    features['min'] = features.min(axis=1)
    features['amplitude'] = features['max'] - features['min']
    features['energy'] = np.sum(np.square(features), axis=1)

    # 计算偏度和峰度
    features['skewness'] = features.apply(lambda row: skew(row), axis=1)
    features['kurtosis'] = features.apply(lambda row: kurtosis(row), axis=1)
    print(features)


