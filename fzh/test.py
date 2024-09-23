import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt


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

    material_type = 3  # 选择材料 i + 1
    temperature = None  # None 表示不选择温度条件
    data_path_list = [r"../appendix1_m1.csv", r"../appendix1_m2.csv", r"../appendix1_m3.csv", r"../appendix1_m4.csv"]

    # 定义文件路径, 注意需要区分材料
    file_path = data_path_list[material_type]
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 替换列名
    df.rename(columns={"温度，oC": "温度", "频率，Hz": "频率", "磁芯损耗，w/m3": "磁芯损耗", "0（磁通密度B，T）": "0"},
              inplace=True)

    # 定义温度筛选条件
    temp_condition = df['温度'] == 90 if temperature is None else df['温度'] == temperature
    # 定义波形筛选条件
    shape_condition = df['励磁波形'] == '正弦波'
    filtered_df = df[temp_condition & shape_condition]

    # 算出磁通密度最大值
    filtered_df['Bm'] = filtered_df.iloc[:, 4:].max(axis=1)

    # 筛选出方程的输入 X 和目标 P
    X = filtered_df[['频率', 'Bm']].to_numpy().T
    P = filtered_df['磁芯损耗'].to_numpy()

    # 拟合模型，p0是参数初始值，param_bounds设定了alpha和beta的边界
    param_bounds = ([-np.inf, 1, 2], [np.inf, 3, 3])
    params, covariance = curve_fit(SE_func, X, P, p0=[1, 1, 2], bounds=param_bounds)

    # 打印拟合的参数和协方差矩阵
    print("拟合参数: ", params)
    print("协方差矩阵: \n", covariance)

    # 计算拟合结果
    filtered_df['预测磁芯损耗'] = SE_func(X, *params)  # 这里直接传递X作为第一个参数
    print(filtered_df.head())


    # 绘制协方差矩阵的热力图并保存
    def plot_covariance_matrix(cov_matrix, param_names):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cov_matrix, annot=True, fmt='.2e', cmap='coolwarm', xticklabels=param_names,
                    yticklabels=param_names)
        plt.title('Covariance Matrix Heatmap')
        plt.xlabel('Parameters')
        plt.ylabel('Parameters')
        plt.savefig('covariance_matrix_heatmap3.png', dpi=300, bbox_inches='tight')
        plt.show()


    # 参数名列表
    param_names = ['k', 'alpha', 'beta']

    # 绘制并保存协方差矩阵
    plot_covariance_matrix(covariance, param_names)
