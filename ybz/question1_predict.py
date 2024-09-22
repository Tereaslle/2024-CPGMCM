import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
import os

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    for file_name in ['appendix2.xlsx']:
        has = {"三角波": "2sj", "梯形波": "3tx", "正弦波": "1zx"}
        # 定义文件路径
        file_path = r"./" + file_name
        folder_name = r"./imgs/" + file_name.split('.')[0] + '/pred/'
        # 如果文件夹不存在，则创建
        os.makedirs(folder_name, exist_ok=True)
        # 读取Excel文件
        df = pd.read_excel(file_path)
        # 修改列名
        df.rename(columns={"温度，oC": "温度",
                           "频率，Hz": "频率",
                           "磁芯损耗，w/m3": "磁芯损耗",
                           "0（磁通密度B，T）": "0"}, inplace=True)
        # 定义三类波形
        signal_shape = df['预测'].unique()

        # 设置1x3的图像布局
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1行3列的布局
        colors = ['#87CEFA', '#20B2AA', '#66CDAA', '#F0E68C', '#FF6347', '#EEE8AA', '#AFEEEE']  # 颜色列表
        for idx, shape in enumerate(["正弦波", "三角波", "梯形波"]):
            # 筛选条件
            shape_condition = df['预测'] == shape
            filtered_df = df[shape_condition]

            # 获取磁通密度数据
            signal_df = filtered_df.iloc[:, 5:df.shape[1]]
            x = list(range(signal_df.shape[1]))  # x 轴为从 0 到 1023 的范围

            # 绘制每种波形的图像
            ax = axs[idx]
            for i in range(signal_df.shape[0]):
                y_values = signal_df.iloc[i]  # 获取当前曲线的y值
                ax.plot(x, y_values, color=colors[i % len(colors)], linestyle='-')

            # 设置子图标题和标签
            ax.set_xlabel('时间')
            if idx == 0:
                ax.set_ylabel('磁通密度')
            ax.set_title(f"{shape} 波形图预测")

        plt.tight_layout()  # 自动调整布局
        combined_image_path = folder_name + 'combined_waveforms.png'
        plt.savefig(combined_image_path, dpi=300)  # 保存为高分辨率图像
        plt.show()


