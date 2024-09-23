import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import platform
# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

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


# 定义斯坦麦茨方程
def SE_func2(x, k, alpha, beta):
    """
    斯坦麦茨方程
    :param x: 变量 [频率，磁通密度峰值]
    :param k: 参数 k
    :param alpha: 参数 alpha
    :param beta: 参数 beta
    :return:
    """
    f, B_m = x
    k = 1.0874580380827605
    alpha = 1.41249400132955234
    beta = 2.0910299441530465
    return k * np.power(f, alpha) * np.power(B_m, beta)

def SE_improved_func(x, delta_k, gamma, k, alpha, beta):
    """
    引入温度因素的斯坦麦茨方程
    :param x: 变量 [频率，磁通密度峰值，温度]
    :param delta_k: 温度修正参数
    :param gamma: 温度修正参数
    :param k: 原参数
    :param alpha: 原参数
    :param beta: 原参数
    :return:
    """
    f, B_m, T = x
    k = k + delta_k * np.log(gamma*T)
    return k * np.power(f, alpha) * np.power(B_m, beta)

def objective(params, f, Bm, T, P):
    """
    定义目标函数：平方误差
    """
    P_pred = SE_improved_func([f, Bm, T], *params)
    return np.sum((P_pred - P) ** 2)
# 计算误差并绘制柱状图
def error_plot():
    material_types = [0, 1, 2, 3]  # 材料种类
    temperatures = [25, 50, 70, 90]  # 温度列表
    data_path_list = [r"../appendix1_m1.csv", r"../appendix1_m2.csv", r"../appendix1_m3.csv", r"../appendix1_m4.csv"]

    # 用于存储每个材料和温度组合的平均误差
    mae_data = []
    material_labels = ['材料 1', '材料 2', '材料 3', '材料 4']
    colors = ['#87CEFA', '#20B2AA', '#66CDAA', '#F0E68C', '#FF6347', '#EEE8AA', '#AFEEEE']

    fig, ax = plt.subplots()

    for i, material_type in enumerate(material_types):
        material_mae = []  # 存储当前材料的误差
        for temperature in temperatures:
            # 定义文件路径
            file_path = data_path_list[material_type]
            # 读取CSV文件
            df = pd.read_csv(file_path)
            # 替换列名
            df.rename(columns={"温度，oC": "温度",
                               "频率，Hz": "频率",
                               "磁芯损耗，w/m3": "磁芯损耗",
                               "0（磁通密度B，T）": "0",
                               "0（磁通密度，T）": "0"}, inplace=True)

            # 定义温度筛选条件
            temp_condition = df['温度'] == temperature
            # 定义波形筛选条件
            shape_condition = df['励磁波形'] == '正弦波'
            # 筛选出符合温度和波形条件的行
            filtered_df = df[temp_condition & shape_condition]
            # 计算磁通密度最大值
            filtered_df['Bm'] = filtered_df.iloc[:, 4:].max(axis=1)
            # 筛选出频率和Bm作为X，磁芯损耗作为P
            X = filtered_df[['频率', 'Bm']].to_numpy().T
            P = filtered_df['磁芯损耗'].to_numpy()

            # 拟合斯坦麦茨方程
            param_bounds = ([-np.inf, 1, 2], [np.inf, 3, 3])
            params, _ = curve_fit(SE_func, X, P, p0=[1, 1, 2], bounds=param_bounds)

            # 计算预测值和误差
            predicted_losses = np.array([SE_func([freq, Bm], *params) for freq, Bm in filtered_df[['频率', 'Bm']].to_numpy()])
            mae = np.mean(np.abs(P - predicted_losses))  # 计算平均绝对误差 (MAE)
            material_mae.append(mae)

        # 存储误差数据
        mae_data.append(material_mae)

        # 绘制柱状图：温度作为X轴，误差作为Y轴
        bar_positions = np.arange(len(temperatures)) + (i * 0.2)  # 不同材料的偏移量
        ax.bar(bar_positions, material_mae, width=0.2, label=material_labels[i], color=colors[i])

    # 设置X轴标签为温度
    ax.set_xticks(np.arange(len(temperatures)) + 0.3)  # 0.3 偏移使得刻度居中
    ax.set_xticklabels([str(temp) + "°C" for temp in temperatures])
    ax.set_xlabel('温度 (°C)')
    ax.set_ylabel('平均绝对误差 (MAE)')
    ax.set_title('不同材料不同温度下正弦波磁芯损耗平均绝对误差')

    # 添加图例到右上角
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('error_plot.png', dpi=300)

def improved_error_plot():
    material_types = [0, 1, 2, 3]  # 材料种类
    temperatures = [25, 50, 70, 90]  # 温度列表
    data_path_list = [r"../appendix1_m1.csv", r"../appendix1_m2.csv", r"../appendix1_m3.csv", r"../appendix1_m4.csv"]

    # 用于存储每个材料和温度组合的平均误差
    mae_data = []
    material_labels = ['材料 1', '材料 2', '材料 3', '材料 4']
    colors = ['#87CEFA', '#20B2AA', '#66CDAA', '#F0E68C', '#FF6347', '#EEE8AA', '#AFEEEE']

    fig, ax = plt.subplots()

    for i, material_type in enumerate(material_types):
        material_mae = []  # 存储当前材料的误差
        for temperature in temperatures:
            # 定义文件路径
            file_path = data_path_list[material_type]
            # 读取CSV文件
            df = pd.read_csv(file_path)
            # 替换列名
            df.rename(columns={"温度，oC": "温度",
                               "频率，Hz": "频率",
                               "磁芯损耗，w/m3": "磁芯损耗",
                               "0（磁通密度B，T）": "0",
                               "0（磁通密度，T）": "0"}, inplace=True)

            # 定义温度筛选条件
            temp_condition = df['温度'] == temperature
            # 定义波形筛选条件
            shape_condition = df['励磁波形'] == '正弦波'
            # 筛选出符合温度和波形条件的行
            filtered_df = df[temp_condition & shape_condition]
            # 计算磁通密度最大值
            filtered_df['Bm'] = filtered_df.iloc[:, 4:].max(axis=1)
            # 筛选出频率和Bm作为X，磁芯损耗作为P
            X = filtered_df[['频率', 'Bm', '温度']].to_numpy().T
            P = filtered_df['磁芯损耗'].to_numpy()

            # 初始猜测的参数值  delta_k, gamma, k, alpha, beta
            initial_guess = [5, 5, 5, 1.4, 2.1]

            # 优化约束条件  delta_k > 0 0<gamma<0.1 eta 无约束 k 无约束
            #    0<delta_alpha<0.5  alpha无约束   0<delta_beta<0.5 beta无约束
            bounds = [(None, None), (None, None), (None, None), (1, 3), (2, 3)]
            # 最小二乘的约束条件
            param_bounds = [[i if i is not None else -np.inf for i, _ in bounds],
                            [i if i is not None else np.inf for _, i in bounds]]

            params, covariance = curve_fit(SE_improved_func, X, P, p0=initial_guess, bounds=param_bounds)

            # 计算预测值和误差
            predicted_losses = np.array([SE_improved_func([freq, Bm, temperature], *params) for freq, Bm in filtered_df[['频率', 'Bm']].to_numpy()])
            mae = np.mean(np.abs(P - predicted_losses))  # 计算平均绝对误差 (MAE)
            material_mae.append(mae)

        # 存储误差数据
        mae_data.append(material_mae)

        # 绘制柱状图：温度作为X轴，误差作为Y轴
        bar_positions = np.arange(len(temperatures)) + (i * 0.2)  # 不同材料的偏移量
        ax.bar(bar_positions, material_mae, width=0.2, label=material_labels[i], color=colors[i])

    # 设置X轴标签为温度
    ax.set_xticks(np.arange(len(temperatures)) + 0.3)  # 0.3 偏移使得刻度居中
    ax.set_xticklabels([str(temp) + "°C" for temp in temperatures])
    ax.set_xlabel('温度 (°C)')
    ax.set_ylabel('平均绝对误差 (MAE)')
    ax.set_title('不同材料不同温度下正弦波磁芯损耗平均绝对误差')

    # 添加图例到右上角
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('improved_error_plot.png', dpi=300)

if __name__ == '__main__':
    improved_error_plot()
