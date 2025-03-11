import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
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
    filtered_df = df[shape_condition]
    # filtered_df = df
    # 选择所有磁通密度的列名
    col = filtered_df.columns[4:]

    # 假设我们有一些已测量的数据
    f_data = filtered_df['频率'].values
    Bm_data = filtered_df[col].max(axis=1)
    T_data = filtered_df['温度'].values
    P_data = filtered_df['磁芯损耗'].values

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

    filtered_df['斯坦麦茨方程'] = [steinmetz_eq([k1_ste_opt, alpha1_ste_opt, beta1_ste_opt], f, Bm) for f, Bm in zip(f_data, Bm_data)]
    filtered_df['多段温度修正的斯坦麦茨方程'] = [temp_SE(f, Bm, T) for f, Bm, T in zip(f_data, Bm_data, T_data)]

    y_true = filtered_df['磁芯损耗'].values
    y_pred_origin = filtered_df['斯坦麦茨方程'].values
    y_pred_new = filtered_df['多段温度修正的斯坦麦茨方程'].values

    # 计算原方程和修正方程的总体误差评价指标
    error_metrics_origin = evaluate_model(y_true, y_pred_origin)
    error_metrics_new = evaluate_model(y_true, y_pred_new)

    # 创建DataFrame保存结果
    df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        '斯坦麦茨方程': error_metrics_origin,
        '修正后的斯坦麦茨方程': error_metrics_new
    })

    # df.to_excel('问题2\\对比结果.xlsx')

    norm_len = []
    for i in range(len(error_metrics_origin)):
        norm_len.append(min(len(str(int(error_metrics_origin[i]))), len(str(int(error_metrics_new[i])))))
    # min(len(str(int(i))) for i in error_metrics_origin)

    df_norm = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
        '斯坦麦茨方程': [e / (10 ** norm_len[i]) for i, e in enumerate(error_metrics_origin)],
        '修正后的斯坦麦茨方程': [e / (10 ** norm_len[i]) for i, e in enumerate(error_metrics_new)]
    })
    # # 可视化：分组直方图
    # df_norm.set_index('Metric').plot(kind='bar', figsize=(10, 6))
    # plt.title('修正前后的误差指标对比直方图', fontsize=20)
    # plt.ylabel('误差值', fontsize=20)
    # plt.xlabel('误差指标', fontsize=20)
    # plt.xticks(rotation=0)
    # plt.savefig(f'./修正前后的误差指标对比直方图.png', dpi=500)
    # plt.show()
    #
    # # 画散点图
    # fig, ax = plt.subplots(figsize=(17, 10))
    # x = range(len(y_true))
    # ax.scatter(y_true, y_pred_origin, color='#0066CC', s=12, label="原方程拟合点")
    # ax.scatter(y_true, y_pred_new, color='#00CC00', s=12, label="修正后的方程拟合点")
    # ax.plot(y_true, y_true,  color='red', linewidth=2, label="真实值参考线")
    # # loc='best'：这个参数指定图例的最佳位置
    # # fontsize=12 指定图例大小
    # # frameon=False：这个参数决定是否在图例周围绘制一个边框
    # # ncol=1：这个参数指定图例中的条目应该被排列成多少列
    # plt.legend(loc='best', fontsize=18, frameon=False, ncol=1)
    # # 设置图表标题和轴标签
    # plt.title('修正前后拟合分布散点图', fontsize=18)
    # plt.xlabel('真实损耗', fontsize=18)
    # plt.ylabel('预测损耗', fontsize=18)
    # plt.savefig(f'./修正前后预测分布散点图.png', dpi=400)
    # plt.show()

    # # --------------------------多段温度分析------------------------
    # temp_index = [filtered_df[filtered_df['温度'] == 25].index,
    #               filtered_df[filtered_df['温度'] == 50].index,
    #               filtered_df[filtered_df['温度'] == 70].index,
    #               filtered_df[filtered_df['温度'] == 90].index]
    #
    # # 计算原方程和修正方程的多段温度平均绝对误差评价指标
    # different_temp_error_metrics_origin = evaluate_temp_mae(y_true, y_pred_origin, temp_index)
    # different_temp_error_metrics_new = evaluate_temp_mae(y_true, y_pred_new, temp_index)
    #
    # # 创建DataFrame保存结果
    # df = pd.DataFrame({
    #     '温度': ['25', '50', '70', '90'],
    #     '斯坦麦茨方程': different_temp_error_metrics_origin,
    #     '修正后斯坦麦茨方程': different_temp_error_metrics_new
    # })
    #
    # colors = ['#40A0FF', '#99FF33', '#7A40FF', '#FF3399']
    # # 可视化：分组直方图
    # df.set_index('温度').plot(kind='bar', figsize=(10, 6), color=colors)
    # plt.title('使用多段温度修正前后的效果对比', fontsize=12)
    # plt.ylabel('MAE误差值', fontsize=12)
    # plt.xticks(rotation=0)
    # plt.savefig(f'./使用多段温度修正前后的效果对比.png', dpi=500)
    # plt.show()

    # 画不同温度下预测结果的散点图
    fig, ax = plt.subplots(figsize=(17, 10))
    x = range(len(y_true))
    colors = ['#40A0FF', '#99FF33','#7A40FF','#FF3399']
    for i, t in enumerate([25,50,70,90]):
        # 筛选出不同温度的预测效果
        temp = filtered_df[filtered_df['温度'] == t]
        y_pred_model1_temp = temp['斯坦麦茨方程'].values
        y_true_temp = temp['磁芯损耗'].values
        ax.scatter(y_true_temp, y_pred_model1_temp, color=colors[i], s=8, label=f"温度{t}拟合点")
    # ax.scatter(y_true, y_pred_model1, color='#40A0FF', s=8)
    ax.plot(y_true, y_true,  color='red', linewidth=2)
    plt.legend(loc='best', fontsize=12, frameon=False, ncol=1)
    # 设置图表标题和轴标签
    plt.title('材料一不同温度下斯坦麦茨方程拟合分布散点图', fontsize=18)
    plt.xlabel('真实损耗', fontsize=18)
    plt.ylabel('预测损耗', fontsize=18)
    plt.legend(loc='best', fontsize=18, frameon=False, ncol=1)
    plt.savefig(f'./不同温度下斯坦麦茨方程拟合分布散点图.png', dpi=400)
    plt.show()
