import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew, kurtosis

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    file_name = 'appendix1_m1.csv'
    file_path = r"../" + file_name
    df = pd.read_csv(file_path)

    # 修改列名
    df.rename(columns={"温度，oC": "温度",
                       "频率，Hz": "频率",
                       "磁芯损耗，w/m3": "磁芯损耗",
                       "0（磁通密度B，T）": "磁通密度"}, inplace=True)

    # 定义波形和对应颜色
    colors = {'三角波': '#87CEFA', '梯形波': '#20B2AA', '正弦波': '#FF6347'}
    metrics = ['max', 'min', 'mean', 'std', 'amplitude', 'energy', 'skewness', 'kurtosis']
    metrics_has = {'max':'波形最大值', 'min':'波形最小值', 'mean':'波形均值', 'std':'波形标准差',
                   'amplitude':'波形振幅', 'energy':'波形能量', 'skewness':'波形偏度', 'kurtosis':'波形峰度'}
    # 创建一个2x4的子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': '3d'})
    fig.suptitle('不同时序特征分布', fontsize=20)

    for i, metric in enumerate(metrics):
        ax = axes[i // 4, i % 4]  # 计算子图的位置

        # 计算统计量
        if metric == 'max':
            df['stat'] = df.groupby(['温度', '频率'])['磁通密度'].transform('max')
        elif metric == 'min':
            df['stat'] = df.groupby(['温度', '频率'])['磁通密度'].transform('min')
        elif metric == 'mean':
            df['stat'] = df.groupby(['温度', '频率'])['磁通密度'].transform('mean')
        elif metric == 'std':
            df['stat'] = df.groupby(['温度', '频率'])['磁通密度'

            ].transform('std')
        elif metric == 'amplitude':
            df['stat'] = df.groupby(['温度', '频率'])['磁通密度'].transform(lambda x: x.max() - x.min())
        elif metric == 'energy':
            df['stat'] = df.groupby(['温度', '频率'])['磁通密度'].transform(lambda x: np.sum(np.square(x)))
        elif metric == 'skewness':
            df['stat'] = df.groupby(['温度', '频率'])['磁通密度'].transform(lambda x: skew(x))
        elif metric == 'kurtosis':
            df['stat'] = df.groupby(['温度', '频率'])['磁通密度'].transform(lambda x: kurtosis(x))

        # 绘制每个波形的散点图
        for shape, color in colors.items():
            filtered_df = df[df['励磁波形'] == shape]
            x = filtered_df['温度']
            y = filtered_df['频率']
            z = filtered_df['stat']
            ax.scatter(x, y, z, color=color, label=shape)

        # 设置标签和标题
        ax.set_title(f'{metrics_has[metric]}')
        if (i > 3):
            ax.set_xlabel('温度 (°C)')
            ax.set_ylabel('频率 (Hz)')

        ax.set_zlabel('磁通密度 (T)')

        # 添加图例
        ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以留出总标题空间
    plt.savefig('3D_Scatter_Plots_Combined.png', dpi=300)
    plt.show()
