import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import pandas as pd
import warnings
import seaborn as sns
from scipy.stats import pearsonr
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import temp_SE,curve_fit_SE

# 忽略所有的警告
warnings.filterwarnings("ignore")

# 设置字体为 SimHei（黑体）以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号
matplotlib.use('TkAgg')


# 定义斯坦麦茨方程模型
def steinmetz_eq(params, f, Bm):
    k1, alpha1, beta1 = params
    return k1 * f ** alpha1 * Bm ** beta1


# 定义目标函数：平方误差
def objective(params, f, Bm, P):
    P_pred = steinmetz_eq(params, f, Bm)
    return np.sum((P_pred - P) ** 2)


# 定义评价指标
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def evaluate_temp_mae(y_true, y_pred, index):
    temp25 = mean_absolute_error(y_true[index[0]], y_pred[index[0]])
    temp50 = mean_absolute_error(y_true[index[1]], y_pred[index[1]])
    temp70 = mean_absolute_error(y_true[index[2]], y_pred[index[2]])
    temp90 = mean_absolute_error(y_true[index[3]], y_pred[index[3]])
    return temp25, temp50, temp70, temp90

if __name__ == '__main__':
    # 定义文件路径
    file_path = r"../appendix1_m1.csv"
    # 读取Excel文件
    df = pd.read_csv(file_path)
    # 替换列名
    df.rename(columns={"温度，oC": "温度",
                       "频率，Hz": "频率",
                       "磁芯损耗，w/m3": "磁芯损耗",
                       "0（磁通密度B，T）": "0",
                       "0（磁通密度，T）": "0"}, inplace=True)
    # 定义波形筛选条件
    shape_condition = df['励磁波形'] == '正弦波'
    data = df[shape_condition]
    # data = df
    # 选择所有磁通密度的列名
    col = data.columns[4:]

    # 假设我们有一些已测量的数据
    f_data = data['频率'].values
    Bm_data = data[col].max(axis=1)
    T_data = data['温度'].values
    P_data = data['磁芯损耗'].values

    # 初始猜测的参数值
    initial_guess = [5, 1.6, 2.7]

    # 约束条件
    bounds = [(0, None), (1, 3), (2, 3)]  # k1 > 0, 1 < alpha1 < 3, 2 < beta1 < 3

    # 使用SLSQP进行优化
    result = minimize(objective, initial_guess, args=(f_data, Bm_data, P_data), bounds=bounds, method='L-BFGS-B')

    # 输出优化结果
    k1_ste_opt, alpha1_ste_opt, beta1_ste_opt = result.x
    print("优化后的系数:")
    print(f"k1: {k1_ste_opt}, alpha1: {alpha1_ste_opt}, beta1: {beta1_ste_opt}")
    print("优化后的目标函数:", result.fun / 1e12)

    data['斯坦麦茨方程'] = [curve_fit_SE(f, Bm) for f, Bm in zip(f_data, Bm_data)]
        # [steinmetz_eq([k1_ste_opt, alpha1_ste_opt, beta1_ste_opt], f, Bm) for f, Bm in zip(f_data, Bm_data)]
    data['不同温度斯坦麦茨方程'] = [temp_SE(f, Bm, T) for f, Bm, T in
                      zip(f_data, Bm_data, T_data)]

    y_true = data['磁芯损耗'].values
    y_pred_model1 = data['斯坦麦茨方程'].values
    y_pred_model2 = data['不同温度斯坦麦茨方程'].values

    temp_index = [data[data['温度'] == 25].index,
                  data[data['温度'] == 50].index,
                  data[data['温度'] == 70].index,
                  data[data['温度'] == 90].index]

    # 计算模型1和模型2的评价指标
    metrics_model1 = evaluate_temp_mae(y_true, y_pred_model1, temp_index)
    metrics_model2 = evaluate_temp_mae(y_true, y_pred_model2, temp_index)

    # 创建DataFrame保存结果
    df = pd.DataFrame({
        '温度': ['25', '50', '70', '90'],
        '斯坦麦茨方程': metrics_model1,
        '修正后斯坦麦茨方程': metrics_model2
    })

    colors = ['#40A0FF', '#99FF33', '#7A40FF', '#FF3399']
    # 可视化：分组直方图
    df.set_index('温度').plot(kind='bar', figsize=(10, 6), color=colors)
    plt.title('使用多段温度修正前后的效果对比', fontsize=12)
    plt.ylabel('MAE误差值', fontsize=12)
    plt.xticks(rotation=0)
    plt.savefig(f'./使用多段温度修正前后的效果对比.png', dpi=500)
    plt.show()

    # fig, ax = plt.subplots(figsize=(17, 10))
    # x = range(len(y_true))
    # colors = ['#40A0FF', '#99FF33','#7A40FF','#FF3399']
    # for i, t in enumerate([25,50,70,90]):
    #     # 筛选出不同温度的预测效果
    #     temp = data[data['温度'] == t]
    #     y_pred_model1_temp = temp['斯坦麦茨方程'].values
    #     y_true_temp = temp['磁芯损耗'].values
    #     ax.scatter(y_true_temp, y_pred_model1_temp, color=colors[i], s=8, label=f"温度{t}")
    # # ax.scatter(y_true, y_pred_model1, color='#40A0FF', s=8)
    # ax.plot(y_true, y_true,  color='red', linewidth=2)
    # plt.legend(loc='best', fontsize=12, frameon=False, ncol=1)
    # # 设置图表标题和轴标签
    # plt.title('材料一斯坦麦茨方程预测分布')
    # plt.xlabel('真实损耗')
    # plt.ylabel('预测损耗')
    # plt.savefig(f'./不同温度下斯坦麦茨方程预测分布.png', dpi=400)
    # plt.show()

