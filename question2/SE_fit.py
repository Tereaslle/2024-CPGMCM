"""
    斯坦麦茨方程k alpha beta 系数拟合
"""

import pandas as pd
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义斯坦麦茨方程
def SE_func(x, k, alpha, beta):
    """
    斯坦麦茨方程
    :param x: 变量
    :param k: 参数 k
    :param alpha: 参数 alpha
    :param beta: 参数 beta
    :return:
    """
    f, B_m = x
    return k * np.power(f, alpha) * np.power(B_m, beta)

if __name__ == '__main__':
    # 定义文件路径,注意需要区分材料
    file_path = r"../appendix1_m1.csv"
    # 读取Excel文件
    df = pd.read_csv(file_path)
    # 替换列名
    df.rename(columns={"温度，oC": "温度",
                       "频率，Hz": "频率",
                       "磁芯损耗，w/m3": "磁芯损耗",
                       "0（磁通密度B，T）": "0",
                       "0（磁通密度，T）": "0"}, inplace=True)
    # 定义四类温度
    temperature = df['温度'].unique()
    # 定义三类波形
    signal_shape = df['励磁波形'].unique()
    # 定义温度筛选条件
    temp_condition = df['温度'] == 90
    # 定义波形筛选条件
    shape_condition = df['励磁波形'] == '正弦波'
    # 只筛选正弦波
    filtered_df = df[temp_condition & shape_condition]
    # 算出磁通密度最大值
    filtered_df['Bm'] = filtered_df.iloc[:, 4:].max(axis=1)
    # 晒选出方程 x ，与 P
    X = filtered_df[['频率', 'Bm']].to_numpy()
    X = X.T
    P = filtered_df['磁芯损耗'].to_numpy()

    # 拟合模型，p0是给参数初始值
    # 设定k, alpha和beta的下界和上界,防止运算溢出。其中k为负无穷到正无穷，alpha为1到3, beta为2到3
    param_bounds = ([-np.inf, 1, 2], [np.inf, 3, 3])
    params, covariance = curve_fit(SE_func, X, P, p0=[1, 1, 2], bounds=param_bounds)

    print(params)

