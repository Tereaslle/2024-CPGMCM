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
    for file_name in ['appendix1_m1.csv', 'appendix1_m2.csv', 'appendix1_m3.csv', 'appendix1_m4.csv']:
        has = {"三角波":"2sj", "梯形波":"3tx", "正弦波":"1zx"}
        # 定义文件路径
        file_path = r"../" + file_name
        folder_name = r"./imgs/" + file_name.split('_')[0] + '/freq/' + file_name.split('_')[1][:2] + '/'
        # 如果文件夹不存在，则创建
        os.makedirs(folder_name, exist_ok=True)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 修改列名
        df.rename(columns={"温度，oC": "温度",
                           "频率，Hz": "频率",
                           "磁芯损耗，w/m3": "磁芯损耗",
                           "0（磁通密度B，T）": "0"}, inplace=True)
        # 定义四类温度
        temperature = df['温度'].unique()
        # 定义三类波形
        signal_shape = df['励磁波形'].unique()
        for temp in temperature:
            for shape in signal_shape:
                # 定义筛选条件
                temp_condition = df['温度'] == temp
                shape_condition = df['励磁波形'] == shape
                filtered_df = df[temp_condition & shape_condition]

                # 从特定行获取磁通密度的波形
                signal_df = filtered_df.iloc[:, 4:df.shape[1]]
                signal_df = signal_df.to_numpy()
                signal_df = np.abs(np.fft.fft(signal_df))
                signal_df = signal_df[:, :signal_df.shape[1] // 50]
                x = list(range(signal_df.shape[1]))  # x 是从 0 到 1023 的列表

                # 设置颜色列表
                colors = ['#87CEFA', '#20B2AA', '#66CDAA', '#F0E68C', '#FF6347', '#EEE8AA', '#AFEEEE']
                # 绘制数据
                plt.figure(figsize=(4, 3))  # 设置图像大小
                for i in range(signal_df.shape[0]):
                    plt.plot(x, signal_df[i, :], color=colors[i % len(colors)], linestyle='-',)  # 使用不同的颜色和透明度

                # plt.xlabel('时间')  # x轴标签
                # plt.ylabel('磁通密度')  # y轴标签
                # plt.title(f"温度{temp}, {shape} 波形图")  # 图像标题
                plt.tight_layout()  # 自动调整子图参数
                plt.savefig(folder_name + str(has[shape] + '_' + str(temp)) + '.png', dpi=300)
                plt.close()  # 每次保存后关闭图像，释放内存
