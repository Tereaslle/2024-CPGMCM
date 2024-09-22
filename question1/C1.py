import pandas as pd
from typing import List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
import platform

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # 设置字体,MacOS 使用'STHeiti'，Windows使用'SimHei'
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示不了的问题


if __name__ == '__main__':
    # 定义文件路径
    file_path = r"./appendix_1.csv"

    # pandas.read_excel('file path')读取Excel文件
    # pandas.read_csv('file path')读取csv文件
    df = pd.read_csv(file_path)
    # 修改列名，inplace=True表示在原数据上修改
    df.rename(columns={"温度，oC": "温度",
                       "频率，Hz": "频率",
                       "磁芯损耗，w/m3": "磁芯损耗",
                       "0（磁通密度B，T）": "0"}, inplace=True)
    # 定义四类温度
    temperature = df['温度'].unique()
    # 定义三类波形
    signal_shape = df['励磁波形'].unique()
    # 定义温度筛选条件
    temp_condition = df['温度'] == 25
    # 定义波形筛选条件
    shape_condition = df['励磁波形'] == signal_shape[0]
    filtered_df = df[(temp_condition) & (shape_condition)]
    # 从第五列开始才是磁通密度的波形
    signal_df = df.iloc[[0,49,99,149,199], 4:df.shape[1]]
    x = [list(range(signal_df.shape[1])) for _ in range(signal_df.shape[0])]  # 生成一个和signal_df列数相同的索引列表
    # 将这些值合并成一个标题字符串，这里使用空格分隔
    title = f"温度{25}, {signal_shape[0]} 波形图"
    # 绘制数据
    plt.figure(figsize=(16, 9))  # 设置图像大小
    # 绘制折线图，颜色固定为蓝色，防止画出来一团
    plt.plot(x, signal_df, color='b')
    plt.xlabel('时间')   # x轴标签
    plt.ylabel('磁通密度')  # y轴标签
    plt.title(title)           # 图像标题
    plt.tight_layout()         # 自动调整子图参数, 使之填充整个图像区域
    plt.show()