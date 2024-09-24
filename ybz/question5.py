import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import platform
import matplotlib

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False

# 定义读取数据的方法
def readdata(data_path: str = '../appendix1_all.csv') -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df

if __name__ == '__main__':
    # 读取数据
    df = readdata()

    # 计算最大磁通密度
    df['max_B'] = df.iloc[:, 5:].max(axis=1)  # 假设磁通密度数据从第六列开始

    # 计算传输磁能
    df['transmission_energy'] = df['频率'] * df['max_B']

    # 提取磁性损耗
    z_loss = df['磁芯损耗']  # 假设磁性损耗的列名为'磁芯损耗'

    # 提取需要的坐标数据
    x = df['温度']
    y = df['频率']
    z_energy = df['transmission_energy']

    # 创建组合图
    fig = plt.figure(figsize=(18, 8))

    # 绘制传输磁能的3D散点图
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(x, y, z_energy, c=z_energy, cmap='viridis', marker='o')
    ax1.set_xlabel('温度 (°C)')
    ax1.set_ylabel('频率 (Hz)')
    ax1.set_zlabel('传输磁能 (W)')
    ax1.set_title('传输磁能分布')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('传输磁能')

    # 绘制磁性损耗的3D散点图
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(x, y, z_loss, c=z_loss, cmap='plasma', marker='o')
    ax2.set_xlabel('温度 (°C)')
    ax2.set_ylabel('频率 (Hz)')
    ax2.set_zlabel('磁芯损耗 (W/m3)')
    ax2.set_title('磁芯损耗分布')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('磁芯损耗')

    plt.tight_layout()  # 调整布局
    plt.savefig('q5_.png', dpi=500)
    plt.show()
