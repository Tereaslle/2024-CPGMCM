"""
    使用多项式拟合、对数函数拟合、幂函数拟合多温度参数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import pyswarms as ps
import platform

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # 设置字体,MacOS 使用'STHeiti'，Windows使用'SimHei'
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示不了的问题
matplotlib.use('TkAgg')

# 定义对数函数模型
def log_func(x, a, b):
    return a + b * np.log(x)

# 定义幂函数模型
def power_func(x, a, b):
    return a + np.power(x,b)


if __name__ == '__main__':
    # 生成一些示例数据
    x = [25, 50, 70, 90]
    k = [3.99479398, 0.74065738, 0.23393301, 0.22204269]
    alpha = [1.347, 1.492, 1.591, 1.602]
    beta = [2.305, 2.482, 2.576, 2.655]

    # 创建字典，用于映射变量名
    value_dict = {0: 'k', 1: 'alpha', 2: 'beta'}
    # 创建一个细粒度的x值数组，用于绘制平滑曲线
    x_fine = np.linspace(min(x), max(x), 400)

    # 设置画布大小
    plt.figure(figsize=(16, 9))  # 宽度12英寸，高度8英寸

    for i, y in enumerate([k, alpha, beta]):
        plt.subplot(2, 3, i + 1)  # 2行3列子图中的第i+1个
        plt.plot(x, y, 'o', label=f'原始数据 {value_dict[i]}')  # 原始数据点只需要绘制一次
        plt.grid()  # 加上网格线
        print(f"参数{value_dict[i]}拟合曲线绘制开始\n-------------------------------------\n")
        for degree in range(1, 4):
            # 使用polyfit进行多项式拟合
            coefficients = np.polyfit(x, y, degree)
            polynomial = np.poly1d(coefficients)
            print(f"系数{value_dict[i]}关于温度T的{degree}次多项式方程:\n", polynomial)
            plt.plot(x_fine, polynomial(x_fine), label=f'{degree}次拟合曲线')  # 拟合曲线
            print(f"参数{value_dict[i]}{degree}次拟合结果的平均绝对误差MAE：{mean_squared_error(y, polynomial(x))}")

        # 对数函数拟合
        params, covariance = curve_fit(log_func, x, y)
        print(f"参数{value_dict[i]}关于温度T的对数函数方程a+b*log(x)参数:\n", params)
        plt.plot(x_fine, log_func(x_fine, *params), label=f'对数函数拟合曲线')  # 拟合曲线
        print(f"参数{value_dict[i]}对数函数拟合结果的平均绝对误差MAE：{mean_squared_error(y, log_func(x, *params))}")

        # 幂函数拟合
        params, covariance = curve_fit(power_func, x, y)
        print(f"系数{value_dict[i]}关于温度T的幂函数方程a+x^b参数:\n", params)
        plt.plot(x_fine, power_func(x_fine, *params), label=f'幂函数拟合曲线')  # 拟合曲线
        print(f"参数{value_dict[i]}幂函数拟合结果的平均绝对误差MAE：{mean_squared_error(y, power_func(x, *params))}")

        # 只要第一行子图的标题
        plt.ylabel(f'{value_dict[i]}值')
        plt.title(f'{value_dict[i]}值随温度变化的拟合曲线')
        plt.legend()

        # 再单独画三次多项式方程的拟合结果
        plt.subplot(2, 3, 4+i)  # 2行3列子图中的第4+i个
        plt.plot(x, y, 'o', label=f'原始数据 {value_dict[i]}')  # 原始数据点只需要绘制一次
        plt.grid()  # 加上网格线
        degree = 3
        # 使用polyfit进行多项式拟合
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)
        plt.plot(x_fine, polynomial(x_fine), label=f'{degree}次拟合曲线')  # 拟合曲线

        # 设置坐标参数
        plt.xlabel('温度')
        plt.ylabel(f'{value_dict[i]}值')
        plt.legend()

        print(f"参数{value_dict[i]}拟合曲线绘制结束\n-------------------------------------\n")

    plt.savefig('./多段温度参数多函数拟合结果.png', dpi=500)
    plt.show()

