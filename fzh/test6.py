import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# 定义斯坦麦茨方程
def SE_func(x, k, alpha, beta):
    f, B_m = x
    return k * np.power(f, alpha) * np.power(B_m, beta)


def plot_covariance_matrices(cov_matrices, param_names):
    fig = plt.figure(figsize=(20, 5))  # 设置整体图的大小
    gs = GridSpec(1, 4, figure=fig, wspace=0.1)  # 创建一个1行4列的网格，调整间距

    for i, cov_matrix in enumerate(cov_matrices):
        ax = fig.add_subplot(gs[i])  # 在网格中添加子图
        sns.heatmap(cov_matrix, annot=True, fmt='.2e', cmap='coolwarm',
                    xticklabels=param_names, yticklabels=param_names if i == 0 else '', ax=ax, cbar=False)
        ax.set_title(f'Covariance Matrix {i + 1}', loc='center')  # 标题居中
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Parameters' if i == 0 else '')  # 只在第一个子图上显示Y轴标签

    # 添加统一的颜色条
    cbar_ax = fig.add_axes([0.93, 0.11, 0.01, 0.8])  # 创建颜色条的轴
    sns.heatmap(cov_matrices[-1], ax=ax, cbar_ax=cbar_ax, cbar=True, cmap='coolwarm',
                annot=True, fmt='.2e', xticklabels=param_names, yticklabels='')
    ax.set_xlabel('Parameters')

    plt.tight_layout()
    plt.savefig('covariance_matrices_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 实验变量参数
    data_path_list = [r"../appendix1_m1.csv", r"../appendix1_m2.csv",
                      r"../appendix1_m3.csv", r"../appendix1_m4.csv"]

    param_names = ['k', 'alpha', 'beta']
    cov_matrices = []

    for material_type in range(4):  # 遍历四种材料
        # 读取CSV文件
        df = pd.read_csv(data_path_list[material_type])
        df.rename(columns={"温度，oC": "温度", "频率，Hz": "频率",
                           "磁芯损耗，w/m3": "磁芯损耗"}, inplace=True)

        # 筛选条件
        temp_condition = df['温度'] == 90
        shape_condition = df['励磁波形'] == '正弦波'
        filtered_df = df[temp_condition & shape_condition]

        filtered_df['Bm'] = filtered_df.iloc[:, 4:].max(axis=1)
        X = filtered_df[['频率', 'Bm']].to_numpy().T
        P = filtered_df['磁芯损耗'].to_numpy()

        # 拟合模型
        param_bounds = ([-np.inf, 1, 2], [np.inf, 3, 3])
        params, covariance = curve_fit(SE_func, X, P, p0=[1, 1, 2], bounds=param_bounds)

        cov_matrices.append(covariance)  # 保存协方差矩阵

    # 绘制所有协方差矩阵
    plot_covariance_matrices(cov_matrices, param_names)
