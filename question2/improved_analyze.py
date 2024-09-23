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
# 忽略所有的警告
warnings.filterwarnings("ignore")

# 设置字体为 SimHei（黑体）以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号
matplotlib.use('TkAgg')

# 定义斯坦麦茨方程模型
def steinmetz_eq(params, f, Bm):
    k1, alpha1, beta1 = params
    return k1 * f ** alpha1 * Bm ** beta1


# 定义目标函数：平方误差
def objective(params, f, Bm, P):
    P_pred = steinmetz_eq(params, f, Bm)
    return np.sum((P_pred - P) ** 2)

# 改进的方程
def steinmetz_eq_adjust(params, f, Bm, T):
    k1, alpha1, beta1, gamma1 = params
    return k1 * f**alpha1 * Bm**beta1 * np.log(T)**gamma1

# 定义目标函数：平方误差
def objective_adjust(params, f, Bm, P, T):
    P_pred = steinmetz_eq_adjust(params, f, Bm, T)
    return np.sum((P_pred - P)**2)

# 定义评价指标
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

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

    # 初始猜测的参数值
    initial_guess = [1.08, 1.4, 2.1,0]

    # 约束条件
    bounds = [(0, None), (1, 3), (2, 3),(None,None)]  # k1 > 0, 1 < alpha1 < 3, 2 < beta1 < 3

    # 使用SLSQP进行优化
    result = minimize(objective_adjust, initial_guess, args=(f_data, Bm_data, P_data,T_data), bounds=bounds, method='L-BFGS-B')

    # 输出优化结果
    k1_adj_opt, alpha1_adj_opt, beta1_adj_opt, gamma1_adj_opt = result.x
    print("优化后的系数:")
    print(f"k1: {k1_adj_opt}, alpha1: {alpha1_adj_opt}, beta1: {beta1_adj_opt}, gamma1:{gamma1_adj_opt}")
    print("优化后的目标函数:",result.fun/1e12)

    data['斯坦麦茨方程'] = [steinmetz_eq([k1_ste_opt, alpha1_ste_opt, beta1_ste_opt], f, Bm) for f, Bm in
                      zip(f_data, Bm_data)]
    data['调整后斯坦麦茨方程'] = [steinmetz_eq_adjust([k1_adj_opt, alpha1_adj_opt, beta1_adj_opt, gamma1_adj_opt], f, Bm, T) for
                         f, Bm, T in zip(f_data, Bm_data, T_data)]
    y_true = data['磁芯损耗'].values
    y_pred_model1 = data['斯坦麦茨方程'].values
    y_pred_model2 = data['调整后斯坦麦茨方程'].values

    fig, ax = plt.subplots(figsize=(21, 9))
    x = range(len(y_true))
    ax.plot(x, y_true, 'r-', label='真实磁芯损耗', linewidth=1)
    ax.plot(x, y_pred_model1, 'b-', label='原斯坦麦茨方程预测', linewidth=2)
    ax.plot(x, y_pred_model2, 'g-', label='修正斯坦麦茨方程预测', linewidth=2)
    # Erase 上面 the data by filling with white
    ax.fill_between(x, y_true, max(y_true), color='white')
    # 设置图例列宽：columnspacing=float (upper left)
    plt.legend(loc='best', fontsize=12, frameon=False, ncol=1)
    # fig.autofmt_xdate()
    plt.show()


    # 计算模型1和模型2的评价指标
    metrics_model1 = evaluate_model(y_true, y_pred_model1)
    metrics_model2 = evaluate_model(y_true, y_pred_model2)

    # 创建DataFrame保存结果
    df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        '斯坦麦茨方程': metrics_model1,
        '调整后斯坦麦茨方程': metrics_model2
    })

    # df.to_excel('问题2\\对比结果.xlsx')

    df_guiyi = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        '斯坦麦茨方程': [i / (10 ** len(str(int(i)))) for i in metrics_model1],
        '调整后斯坦麦茨方程': [i / (10 ** len(str(int(i)))) for i in metrics_model2]
    })
    # 可视化：分组直方图
    df_guiyi.set_index('Metric').plot(kind='bar', figsize=(10, 6))
    plt.title('调整前后的效果对比（映射到0-1）', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.xticks(rotation=0)
    plt.savefig(f'./调整前后的效果对比（映射到0-1）.png', dpi=500)
    plt.show()

