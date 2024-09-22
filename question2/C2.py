import pandas as pd
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
# 定义文件路径
file_path = r"../附件一（训练集）.xlsx"

# 读取Excel文件
data = pd.read_excel(file_path)
# 输出数据的前几行查看
# 假设 `waveform_data` 是一个 numpy 数组，代表正弦波形数据
# 同时假设我们知道每个采样点之间的时间间隔 dt（采样周期）
i_signal=0
# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
selected_data = data.iloc[i_signal, 4:]
x = range(len(selected_data))  # 生成一个和selected_data长度相同的索引列表
# 读取第一行的前四列
param = data.iloc[i_signal, :4]
# 将这些值合并成一个标题字符串，这里使用空格分隔
title = ' '.join(str(x) for x in param)
# 绘制数据
plt.figure(figsize=(5, 3))  # 设置图像大小
plt.plot(x, selected_data, marker='o')  # 绘制折线图，并添加数据点
plt.xlabel('时间')  # x轴标签
plt.ylabel('磁通密度')  # y轴标签
plt.title(title)  # 图像标题
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()

i_signal=0
demo_data = data.iloc[i_signal, 4:]
# 假设提供的时序数据、频率和温度
B_max = max(demo_data)  # 最大磁通密度，单位 T (Tesla)
f = data.iloc[i_signal, 1]    # 频率，单位 Hz
T = data.iloc[i_signal, 0]  # 温度，单位 °C
T_0 = 25  # 基准温度，单位 °C
P_real = data.iloc[i_signal, 2]   # 真实磁芯损耗功率，单位 W

# 斯坦麦茨方程参数
alpha = 1.5  # 频率指数
beta = 2.3  # 磁通密度指数
# 斯坦麦茨方程损耗常数k计算
k_f = P_real / ((f ** alpha) * (B_max ** beta))

# 温度修正参数
gamma = -0.02  # 温度对 k_f 的修正系数
delta_alpha = 0.001  # 温度对 alpha 的修正系数
delta_beta = 0.002  # 温度对 beta 的修正系数


i_signal=350
demo_data = data.iloc[i_signal, 4:]
# 假设提供的时序数据、频率和温度
B_max = max(demo_data)  # 最大磁通密度，单位 T (Tesla)
f = data.iloc[i_signal, 1]    # 频率，单位 Hz
T = data.iloc[i_signal, 0]  # 温度，单位 °C
T_0 = 25  # 基准温度，单位 °C
P_real = data.iloc[i_signal, 2]   # 真实磁芯损耗功率，单位 W


# 斯坦麦茨方程损耗计算
P_SE = k_f * (f ** alpha) * (B_max ** beta)

# 温度修正方程损耗计算
k_f_T = k_f * (1 + gamma * (T - T_0))
alpha_T = alpha + delta_alpha * (T - T_0)
beta_T = beta + delta_beta * (T - T_0)
P_T_correction = k_f_T * (f ** alpha_T) * (B_max ** beta_T)

# 计算误差
error_SE = P_SE - P_real
error_T_correction = P_T_correction - P_real

# 绘制对比图
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# 图1：损耗值对比
axs[0].bar(['真实值', '斯坦麦茨方程', '温度修正方程'], [P_real, P_SE, P_T_correction], color=['green', 'blue', 'orange'])
axs[0].set_title('损耗值对比')
axs[0].set_ylabel('损耗功率 (W)')

# 图2：误差对比
axs[1].bar(['斯坦麦茨方程误差', '温度修正方程误差'], [error_SE, error_T_correction], color=['blue', 'orange'])
axs[1].set_title('损耗预测误差对比')
axs[1].set_ylabel('误差 (W)')

plt.tight_layout()
plt.show()
