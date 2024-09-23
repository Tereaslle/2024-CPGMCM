"""
    斯坦麦茨方程 k alpha beta 系数最小二乘拟合
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义斯坦麦茨方程
def SE_func(x, k, alpha, beta):
    """
    斯坦麦茨方程
    :param x: 变量 [频率，磁通密度峰值]
    :param k: 参数 k
    :param alpha: 参数 alpha
    :param beta: 参数 beta
    :return:
    """
    f, B_m = x
    return k * np.power(f, alpha) * np.power(B_m, beta)

if __name__ == '__main__':
    # 实验变量参数
    material_type = 0   # 选择材料 i + 1
    temperature = None  # None 表示不选择温度条件
    data_path_list = [r"../appendix1_m1.csv", r"../appendix1_m2.csv", r"../appendix1_m3.csv", r"../appendix1_m4.csv"]

    # 定义文件路径,注意需要区分材料
    file_path = data_path_list[material_type]
    # 读取Excel文件
    df = pd.read_csv(file_path)
    # 替换列名
    df.rename(columns={"温度，oC": "温度",
                       "频率，Hz": "频率",
                       "磁芯损耗，w/m3": "磁芯损耗",
                       "0（磁通密度B，T）": "0",
                       "0（磁通密度，T）": "0"}, inplace=True)

    # 定义三类波形
    signal_shape = df['励磁波形'].unique()
    # 定义温度筛选条件
    temp_condition = df['温度'] == 90
    if temperature is not None:
        temp_condition = df['温度'] == temperature
    # 定义波形筛选条件
    shape_condition = df['励磁波形'] == '正弦波'
    # 筛选条件选择
    if temperature is not None:
        filtered_df = df[temp_condition & shape_condition]
    else:
        filtered_df = df[shape_condition]

    # 获取所有磁通密度
    # 获取所有磁通密度 所有列号
    B_col = filtered_df.columns[4:]

    # 算出磁通密度最大值
    filtered_df['Bm'] = filtered_df[B_col].max(axis=1)
    # 筛选出方程 x ，与 P
    X = filtered_df[['频率', 'Bm']].to_numpy().T
    P = filtered_df['磁芯损耗'].to_numpy()

    # 拟合模型，p0是给参数初始值
    # 设定k, alpha和beta的下界和上界,防止运算溢出。其中k为负无穷到正无穷，alpha为1到3, beta为2到3
    param_bounds = ([-np.inf, 1, 2], [np.inf, 3, 3])
    params, covariance = curve_fit(SE_func, X, P, p0=[1, 1, 2], bounds=param_bounds)
    print(f"拟合参数结果: {params}")
    print(f"协方差矩阵: \n{covariance}")
    # 计算拟合的平均绝对误差
    # filtered_df['预测磁芯损耗'] = SE_func(x=filtered_df[['频率', 'Bm']].to_numpy(), *params)
    # print(filtered_df.head())

