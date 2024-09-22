"""
绘图相关函数
"""
from math import log10
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
plt.rcParams.update({"font.size": 29})  # 设置字体大小


def multiplot_2D(y: List[np.ndarray], x=None, data_label=None, xlabel="", ylabel="", title=""):
    """
    绘制二维多曲线图像的通用函数
    :param dataset: 数据组（包含多个对比数据）
    :param data_label: 数据标签
    :param x: x轴数据，未指定则默认为数组长度
    :param xlabel: x轴坐标名
    :param ylabel: y轴坐标名
    :param title: 图像标题名
    :return:
    """
    if x is None:
        x = [[i for i in range(len(y[0]))] for _ in range(len(y))]
    if data_label is None:
        data_label = list(map(str, range(len(y))))
    fig = plt.figure(figsize=(17, 13))  # figsize为画布的长宽比
    ax = fig.add_subplot(1, 1, 1)  # 把画布分成1行1列，取第一个
    for i, data in enumerate(y):
        ax.plot(x[i], data, label=data_label[i])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    fig.show()


def plot_2D(data: np.ndarray, x=None, xlabel="", ylabel="", title=""):
    """
    绘制二维图像的通用函数
    :param data: y轴数据
    :param x: x轴数据，未指定则默认为数组长度
    :param xlabel: x轴坐标名
    :param ylabel: y轴坐标名
    :param title: 图像标题名
    :return:
    """
    if x is None:
        x = range(len(data))
    fig = plt.figure(figsize=(17, 13))  # figsize为画布的长宽比
    ax = fig.add_subplot(1, 1, 1)  # 把画布分成1行1列，取第一个
    ax.plot(x, data)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.show()


def plot_3D(data: np.ndarray, x=None, y=None, xlabel="", ylabel="", zlable="", title=""):
    if x is None:
        x = list(range(data.shape[1]))
    if y is None:
        y = list(range(data.shape[0]))
    X, Y = np.meshgrid(x, y)
    # -----------------绘制3D图像-----------------------
    fig = plt.figure(figsize=(17, 14))  # figsize为画布的长宽比
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, data)
    ax.set_xlabel(ylabel, labelpad=20)  # labelpad设置轴标题与轴的距离
    ax.set_ylabel(xlabel, labelpad=20)
    ax.set_zlabel(zlable, labelpad=25)
    ax.set_title(title, fontsize=35, pad=0.5)
    ax.ticklabel_format(axis="z", style="scientific",
                        scilimits=(-2, 2), useMathText=False)  # 设置为科学计数法表示
    ax.view_init(elev=20, azim=-115)  # 调整3D图的显示视角，elev为垂直旋转角，azim为水平旋转角
    fig.show()


def distance_fft(fft_data: np.ndarray, distance_resolution: float) -> None:
    """
    绘制距离维fft结果
    :param fft_data: fft结果矩阵，一行为一个chirp
    :param distance_resolution: 距离分辨率
    :return:
    """
    # 定义分贝转换公式，fft后将幅度值转为分贝值缩小范围，转换公式为：20*log10(幅度值)
    db = np.frompyfunc(lambda x: 20 * log10(abs(x)), 1, 1)
    # 定义chirp数与采样数
    chirps_num, samples_per_chirp = fft_data.shape  # 只要不声明为global，就为局部作用域的变量，两个变量不同
    # 将采样数映射到距离维
    distance_range = list(map(lambda x: x * distance_resolution, range(samples_per_chirp)))
    # 绘制一个chirp的距离维fft
    fig = plt.figure(figsize=(17, 13))  # figsize为画布的长宽比
    # fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(13, 13))  # ncols，nrows用于划分子图,figsize为画布的长宽比
    ax = fig.add_subplot(1, 1, 1)  # 把画布分成1行1列，取第一个
    ax.plot(distance_range, db(fft_data[0, :]))
    ax.set_ylabel('幅度(db)')
    ax.set_xlabel('距离(m)')
    ax.set_title('距离维FFT（1个chirp）')
    fig.show()
    # 做笛卡尔积，生成3D图像的所有x,y点对
    X, Y = np.meshgrid(distance_range, range(chirps_num))
    # --------------------绘制3D图像--------------------
    fig = plt.figure(figsize=(17, 14))  # figsize为画布的长宽比
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, fft_data)
    ax.set_xlabel('距离(m)', labelpad=20)  # labelpad设置轴标题与轴的距离
    ax.set_ylabel('chirp序号', labelpad=20)
    ax.set_zlabel('幅度', labelpad=25)
    ax.set_title('距离维 1D FFT结果')
    ax.ticklabel_format(axis="z", style="scientific",
                        scilimits=(-2, 2), useMathText=False)  # 设置为科学计数法表示
    ax.view_init(elev=20, azim=-115)  # 调整3D图的显示视角，elev为垂直旋转角，azim为水平旋转角
    fig.show()


# 绘制降噪效果对比图
def denoise_contrast(fft_data, fft_data_denoise, distance_resolution: float):
    """
    绘制降噪效果对比图
    :param fft_data: 原始数据,一行为一个chirp
    :param fft_data_denoise: 降噪后的数据,一行为一个chirp
    :param distance_resolution: 距离分辨率
    :return:
    """
    chirps_num, samples = fft_data.shape  # 定义：调频脉冲数, 每个脉冲的采样点数
    distance_range = list(map(lambda x: x * distance_resolution, range(samples)))  # 根据采样点数计算出对应的距离范围
    # ---------------绘制1个chirp的2D对比图----------------
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(13, 13))  # ncols，nrows用于划分子图,figsize为画布的长宽比
    axs[0].plot(distance_range, np.abs(fft_data[1, :]))
    axs[0].set_xlabel('距离(m)')
    axs[0].set_ylabel('幅度')
    axs[0].set_title('距离维FFT（1个chirp）')
    axs[0].ticklabel_format(axis="y", style="scientific",
                            scilimits=(-2, 2), useMathText=True)  # y轴设置为科学计数法表示
    axs[1].plot(distance_range, np.abs(fft_data_denoise[1, :]))
    axs[1].set_xlabel('距离(m)')
    axs[1].set_ylabel('幅度')
    axs[1].set_title('静态杂波滤除后')
    axs[1].ticklabel_format(axis="y", style="scientific",
                            scilimits=(-2, 2), useMathText=True)  # y轴设置为科学计数法表示
    plt.tight_layout()  # tight_layout会自动调整子图参数，使之填充整个图像区域。
    fig.show()
    # ---------------绘制1个chirp的2D对比图结束----------------
    # 做笛卡尔积，生成3D图像的所有x,y点对
    X, Y = np.meshgrid(distance_range, range(chirps_num))
    # -----------------绘制3D对比图-----------------------
    fig, axs = plt.subplots(ncols=1, nrows=2,
                            figsize=(13, 13),
                            subplot_kw={"projection": "3d"})  # ncols，nrows用于划分子图,figsize为画布的长宽比
    axs[0].plot_surface(X, Y, np.abs(fft_data))
    axs[0].set_xlabel('距离(m)', labelpad=30)  # labelpad设置轴标题与轴的距离
    axs[0].set_ylabel('chirp序号', labelpad=25)
    axs[0].set_zlabel('幅度', labelpad=25)
    axs[0].set_title('距离维 1D FFT结果')
    axs[0].ticklabel_format(axis="z", style="scientific",
                            scilimits=(-2, 2), useMathText=True)  # 设置为科学计数法表示
    axs[0].view_init(elev=20, azim=-115)  # 调整3D图的显示视角，elev为垂直旋转角，azim为水平旋转角
    axs[1].plot_surface(X, Y, np.abs(fft_data_denoise))
    axs[1].view_init(elev=20, azim=-115)  # 调整3D图的显示视角，elev为垂直旋转角，azim为水平旋转角
    axs[1].set_title('静态杂波滤除后')
    axs[1].ticklabel_format(axis="z", style="scientific",
                            scilimits=(-2, 2), useMathText=True)  # 设置为科学计数法表示
    plt.tight_layout()  # tight_layout会自动调整子图参数，使之填充整个图像区域。
    fig.show()
    # ------------------绘制3D对比图结束-----------------------
