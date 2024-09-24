"""
    比较最小二乘法与L-BFGS-B算法参数拟合效果
"""
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
from scipy.optimize import minimize, curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 忽略所有的警告
warnings.filterwarnings("ignore")

# 设置字体为 SimHei（黑体）以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号
matplotlib.use('TkAgg')

# 定义斯坦麦茨方程
def SE_func(x, k, alpha, beta):
    f, B_m = x
    return k * np.power(f, alpha) * np.power(B_m, beta)

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
    filtered_df = df[shape_condition]
    # 选择所有磁通密度的列名
    B_col = filtered_df.columns[4:]

    # 算出磁通密度最大值
    filtered_df['Bm'] = filtered_df[B_col].max(axis=1)
    # 假设我们有一些已测量的数据
    f_data = filtered_df['频率'].values
    Bm_data = filtered_df[B_col].max(axis=1)
    P_data = filtered_df['磁芯损耗'].values
    X = filtered_df[['频率', 'Bm']].to_numpy().T

    # 初始猜测的参数值
    initial_guess = [5, 1.6, 2.7]

    # 约束条件
    bounds = [(0, None), (1, 3), (2, 3)]  # k1 > 0, 1 < alpha1 < 3, 2 < beta1 < 3

    # 使用L-BFGS-B进行优化以拟合参数
    result = minimize(objective, initial_guess, args=(f_data, Bm_data, P_data), bounds=bounds, method='L-BFGS-B')
    # 输出优化结果
    k1_ste_opt, alpha1_ste_opt, beta1_ste_opt = result.x
    print("优化后的系数:")
    print(f"k1: {k1_ste_opt}, alpha1: {alpha1_ste_opt}, beta1: {beta1_ste_opt}")
    print("优化后的目标函数:", result.fun / 1e12)

    # 最小二乘的约束条件
    param_bounds = [[i if i is not None else -np.inf for i, _ in bounds],
                    [i if i is not None else np.inf for _, i in bounds]]
    # 使用最小二乘法拟合参数
    params, covariance = curve_fit(SE_func, X, P_data, p0=initial_guess, bounds=param_bounds)
    print(f"最小二乘法拟合参数结果: {params}")



    filtered_df['L-BFGS算法拟合的斯坦麦茨方程'] = [steinmetz_eq([k1_ste_opt, alpha1_ste_opt, beta1_ste_opt], f, Bm) for f, Bm in
                                          zip(f_data, Bm_data)]
    filtered_df['最小二乘法拟合的斯坦麦茨方程'] = [SE_func((f, Bm), *params) for f, Bm in zip(f_data, Bm_data)]

    y_true = filtered_df['磁芯损耗'].values
    y_pred_lBFGSb = filtered_df['L-BFGS算法拟合的斯坦麦茨方程'].values
    y_pred_ols = filtered_df['最小二乘法拟合的斯坦麦茨方程'].values

    # 计算模型1和模型2的评价指标
    error_metrics_lBFGSb = evaluate_model(y_true, y_pred_lBFGSb)
    error_metrics_ols = evaluate_model(y_true, y_pred_ols)

    # 创建DataFrame保存结果
    df = pd.DataFrame({
        '误差评价指标': ['MSE', 'RMSE', 'MAE', 'R2'],
        'L-BFGS算法拟合的误差结果': error_metrics_lBFGSb,
        '最小二乘法拟合的误差结果': error_metrics_ols
    })

    # df.to_excel('问题2\\对比结果.xlsx')

    df_norm = pd.DataFrame({
        '误差评价指标': ['MSE', 'RMSE', 'MAE', 'R2'],
        'L-BFGS算法拟合的误差结果': [i / (10 ** len(str(int(i)))) for i in error_metrics_lBFGSb],
        '最小二乘法拟合的误差结果': [i / (10 ** len(str(int(i)))) for i in error_metrics_ols]
    })
    # 可视化：分组直方图

    # 定义配色
    colors = ['#FF6347','#66CDAA']
    df_norm.set_index('误差评价指标').plot(kind='bar', figsize=(10, 6), color=colors)
    plt.title('最小二乘法与L-BFGS优化算法拟合效果对比', fontsize=12)
    plt.ylabel('误差值', fontsize=12)
    plt.xticks(rotation=0)
    plt.savefig(f'./最小二乘法与L-BFGS优化算法拟合效果对比.png', dpi=500)
    plt.show()




