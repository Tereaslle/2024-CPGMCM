import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import pandas as pd
import warnings
import seaborn as sns
from scipy.stats import pearsonr
from scipy.optimize import minimize
# 忽略所有的警告
warnings.filterwarnings("ignore")

# 设置字体为 SimHei（黑体）以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号

#import pandas as pd

file_path = '../附件一（训练集）.xlsx'






# 定义 sheet 名字列表，分别为 材料1, 材料2, 材料3, 材料4
sheets = ['材料1', '材料2', '材料3', '材料4']

# 初始化一个空的 DataFrame 来存储所有材料的数据
all_data = pd.DataFrame()

# 读取每个 sheet 的前四列，并添加材料列
for sheet in sheets:
    # 读取 sheet 前四列
    data = pd.read_excel(file_path, sheet_name=sheet)

    # 添加一列材料信息
    data['磁芯材料'] = sheet

    # 将当前 sheet 的数据添加到总的 DataFrame 中
    all_data = pd.concat([all_data, data], ignore_index=True)
#print(all_data)

col = data.columns[4:]

# 假设我们有一些已测量的数据
f_data = data['频率，Hz'].values
Bm_data = data[col].max(axis=1).values
P_data = data['磁芯损耗，w/m3'].values
T_data = data['温度，oC'].values




# 改进的方程
def separate(params, f, Bm):
    a1, a2, a3, a4 = params
    return a1 * f * Bm ** a2 + a3 * f ** 2 * Bm ** 2 + (a4 * 0.1356) ** 0.5 * f ** 1.5 * Bm ** 1.5


# 定义目标函数：平方误差
def objective_sep(params, f, Bm, P):
    P_pred = separate(params, f, Bm)
    return np.sum((P_pred - P) ** 2)


def steinmetz_eq_adjust(params, f, Bm, T):
    k1, alpha1, beta1, gamma1 = params
    return k1 * f ** alpha1 * Bm ** beta1 * np.log(T) ** gamma1


# 定义目标函数：平方误差
def objective_adjust(params, f, Bm, P, T):
    P_pred = steinmetz_eq_adjust(params, f, Bm, T)
    return np.sum((P_pred - P) ** 2)


ls = []
bo = []
mat = []
af = []
bf = []
for i in all_data['磁芯材料'].unique():
    for o in all_data['励磁波形'].unique():
        tem = all_data[(all_data.磁芯材料 == i) & (all_data.励磁波形 == o)]

        f_data = tem['频率，Hz'].values
        Bm_data = tem[col].max(axis=1).values
        P_data = tem['磁芯损耗，w/m3'].values
        T_data = tem['温度，oC'].values

        # 损耗分离模型
        initial_guess = [-2, 0, 0, 0]
        bounds = [(None, None), (None, None), (None, None), (None, None)]  # k1 > 0, 1 < alpha1 < 3, 2 < beta1 < 3
        result = minimize(objective_sep, initial_guess, args=(f_data, Bm_data, P_data), bounds=bounds,
                          method='L-BFGS-B')
        b1, b2, b3, b4 = result.x
        a_func = result.fun
        # 改进的斯坦麦茨方程
        initial_guess = [0, 0, 0, 0]
        bounds = [(0, None), (1, 3), (2, 3), (None, None)]  # k1 > 0, 1 < alpha1 < 3, 2 < beta1 < 3
        result = minimize(objective_adjust, initial_guess, args=(f_data, Bm_data, P_data, T_data), bounds=bounds,
                          method='L-BFGS-B')
        a1, a2, a3, a4 = result.x
        b_func = result.fun

        ls.append([a1, a2, a3, a4, b1, b2, b3, b4])
        bo.append(o)
        mat.append(i)
        af.append(a_func)
        bf.append(b_func)
result = pd.DataFrame(ls, columns=['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4'])
result['励磁波形'] = bo
result['材料'] = mat
result['损耗分离模型'] = af
result['修正斯坦麦茨方程损失'] = bf


def ensemble(params, a1, a2, a3, a4, b1, b2, b3, b4, f, Bm, T):
    a, b = params
    return a * steinmetz_eq_adjust([a1, a2, a3, a4], f, Bm, T) + b * separate([b1, b2, b3, b4], f, Bm)


# 定义目标函数：平方误差
def objectivet(params, a1, a2, a3, a4, b1, b2, b3, b4, f, Bm, P, T):
    P_pred = ensemble(params, a1, a2, a3, a4, b1, b2, b3, b4, f, Bm, T)
    return np.sum((P_pred - P) ** 2)


als = []
bls = []
total_cost = []
for i in all_data['磁芯材料'].unique():
    for o in all_data['励磁波形'].unique():
        tem = all_data[(all_data.磁芯材料 == i) & (all_data.励磁波形 == o)]
        a1, a2, a3, a4, b1, b2, b3, b4 = \
        result[(result.材料 == i) & (result.励磁波形 == o)][['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4']].values[0]

        f_data = tem['频率，Hz'].values
        Bm_data = tem[col].max(axis=1).values
        P_data = tem['磁芯损耗，w/m3'].values
        T_data = tem['温度，oC'].values

        # 损耗分离模型
        initial_guess = [0.5, 0.5]
        bounds = [(0, 1), (0, 1)]  # k1 > 0, 1 < alpha1 < 3, 2 < beta1 < 3
        total_result = minimize(objectivet, initial_guess,
                                args=(a1, a2, a3, a4, b1, b2, b3, b4, f_data, Bm_data, P_data, T_data), bounds=bounds,
                                method='L-BFGS-B')
        a, b = total_result.x
        func = total_result.fun

        als.append(a)
        bls.append(b)
        total_cost.append(func)

result['a'] = als
result['b'] = als
# result.to_excel('问题4\\不同材料不同波形的损耗分离模型和修正方程模型拟合结果.xlsx',index=False)
print(result)

#print(df.columns)
print(col)

df = pd.read_excel('附件三（测试集）.xlsx')

f_data = df['频率，Hz'].values
Bm_data = df[col].max(axis=1).values
T_data = df['温度，oC'].values

bo = df['励磁波形'].values
mat = df['磁芯材料'].values

pred_ls = []
for b,m,f,bm,t in zip(bo,mat,f_data,Bm_data,T_data):
    a1,a2,a3,a4,b1,b2,b3,b4,a,b = result[(result.材料==m)&(result.励磁波形==b)][['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4','a1','a2']].values[0]
    pred = ensemble([a,b], a1,a2,a3,a4,b1,b2,b3,b4,f, bm, t)
    pred_ls.append(pred)
print(pred_ls)

print(len(pred_ls))