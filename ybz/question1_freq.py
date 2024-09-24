import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
from scipy.stats import skew, kurtosis
from mpl_toolkits.mplot3d import Axes3D

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    metrics = ['max', 'min', 'mean', 'std', 'amplitude', 'energy', 'skewness', 'kurtosis']
    metrics_has = {
        'max': '频率最大值',
        'min': '频率最小值',
        'mean': '频率均值',
        'std': '频率标准差',
        'amplitude': '频率振幅',
        'energy': '频率能量',
        'skewness': '频率方差',
        'kurtosis': '频率密度'
    }
    has = {"三角波": "2sj", "梯形波": "3tx", "正弦波": "1zx"}
    colors = ['b', 'g', 'r']  # 为三种频率分配不同的颜色

    # 遍历每个文件
    for file_name in ['appendix1_m1.csv', 'appendix1_m2.csv', 'appendix1_m3.csv', 'appendix1_m4.csv']:
        file_path = r"../" + file_name
        df = pd.read_csv(file_path)
        df.rename(columns={"温度，oC": "温度", "频率，Hz": "频率", "磁芯损耗，w/m3": "磁芯损耗", "0（磁通密度B，T）": "0"}, inplace=True)

        # 创建2x4的3D散点图
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('不同频率特征分布', fontsize=20)

        # 遍历每个统计量
        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(2, 4, idx + 1, projection='3d')
            ax.set_title(metrics_has[metric])  # 使用哈希映射更新子标题

            for i, shape in enumerate(df['励磁波形'].unique()):
                temp_values = []  # 保存每个样本的温度
                freq_values = []  # 保存每个样本的频率
                z_values = []  # 保存统计量的z值

                for temp in df['温度'].unique():
                    for freq in df['频率'].unique():
                        temp_condition = (df['温度'] == temp)
                        freq_condition = (df['频率'] == freq)
                        shape_condition = (df['励磁波形'] == shape)

                        filtered_df = df[temp_condition & freq_condition & shape_condition]

                        if len(filtered_df) == 0:
                            continue

                        # 从特定行获取磁通密度的波形
                        signal_df = filtered_df.iloc[:, 4:df.shape[1]].to_numpy()
                        signal_df = np.abs(np.fft.fft(signal_df))  # 计算傅里叶变换

                        # 保持完整数据
                        signal_df = signal_df[:, :signal_df.shape[1]]

                        # 计算统计量
                        if metric == 'max':
                            z_value = np.max(signal_df)
                        elif metric == 'min':
                            z_value = np.min(signal_df)
                        elif metric == 'mean':
                            z_value = np.mean(signal_df)
                        elif metric == 'std':
                            z_value = np.std(signal_df)
                        elif metric == 'amplitude':
                            z_value = np.max(signal_df) - np.min(signal_df)
                        elif metric == 'energy':
                            z_value = np.sum(np.square(signal_df))
                        elif metric == 'skewness':
                            z_value = skew(signal_df.flatten())
                        elif metric == 'kurtosis':
                            z_value = kurtosis(signal_df.flatten())

                        # 记录温度、频率和统计量
                        temp_values.append(temp)
                        freq_values.append(freq)
                        z_values.append(z_value)

                # 绘制散点图
                if len(temp_values) > 0:
                    ax.scatter(temp_values, freq_values, z_values, color=colors[i], alpha=0.7, label=shape)
            if (i > 3):
                ax.set_xlabel('温度 (°C)')
                ax.set_ylabel('频率 (Hz)')
            ax.set_zlabel(metrics_has[metric])  # 使用哈希映射更新z轴标签
            ax.legend(loc='upper left')  # 只显示三种波形的图例

        plt.tight_layout()
        plt.savefig('3d_scatter_combined_styled_fixed.png', dpi=500)
        plt.close()  # 关闭图像，释放内存
