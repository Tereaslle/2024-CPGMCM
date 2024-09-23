"""
    改进的斯坦麦茨方程系数最小二乘拟合
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize,curve_fit


def SE_improved_func(x, delta_k, gamma, eta, k, delta_alpha, alpha, delta_beta, beta):
    """
    引入温度因素的斯坦麦茨方程
    :param x: 变量 [频率，磁通密度峰值，温度]
    :param delta_k: 温度修正参数
    :param gamma: 温度修正参数
    :param eta: 温度修正参数
    :param k: 原参数
    :param delta_alpha: 温度修正参数
    :param alpha: 原参数
    :param delta_beta: 温度修正参数
    :param beta: 原参数
    :return:
    """
    f, B_m, T = x
    k = k + -delta_k * np.log(-T * gamma + eta)
    alpha = alpha + delta_alpha * T
    beta = beta + delta_beta * T
    return k * np.power(f, alpha) * np.power(B_m, beta)


def objective(params, f, Bm, T, P):
    """
    定义目标函数：平方误差
    """
    P_pred = SE_improved_func([f, Bm, T], *params)
    return np.sum((P_pred - P) ** 2)


if __name__ == '__main__':
    # 实验变量参数
    material_type = 0  # 选择材料 i + 1
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

    # 获取原始数据
    B_col = filtered_df.columns[4:]

    # 获取输入参数
    f_data = filtered_df['频率'].values
    filtered_df['Bm'] = filtered_df[B_col].max(axis=1)
    Bm_data = filtered_df['Bm']
    T_data = filtered_df['温度'].values
    X = filtered_df[['频率','Bm','温度']].to_numpy().T
    P_data = filtered_df['磁芯损耗'].values

    # 初始猜测的参数值  delta_k, gamma, eta, k, delta_alpha, alpha, delta_beta, beta
    initial_guess = [5, 0.05, 5, 5,
                     0.03, 5, 0.03, 5]

    # 优化约束条件  delta_k > 0 0<gamma<0.1 eta 无约束 k 无约束
    #    0<delta_alpha<0.5  alpha无约束   0<delta_beta<0.5 beta无约束
    bounds = [(0, None), (0, 0.1), (None, None), (None, None),
              (0, 0.5), (None, None), (0, 0.5), (None, None)]
    # 最小二乘的约束条件
    param_bounds = [[0,0,-np.inf,-np.inf,
                     0,-np.inf,0,-np.inf],
                    [np.inf,0.1,np.inf,np.inf,
                     0.5,np.inf,0.5,np.inf]]

    params, covariance = curve_fit(SE_improved_func, X, P_data, p0=initial_guess, bounds=param_bounds)
    print(f"拟合参数结果: {params}")
    print(f"协方差矩阵: \n{covariance}")