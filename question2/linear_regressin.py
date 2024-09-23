"""
    使用多项式拟合多温度参数
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
def log_func(x, a, b, c, d):
    return a + b * np.log(c*x + d)

# 定义幂函数模型
def power_func(x, a, b):
    return a + np.power(x,b)

# 目标函数，我们需要最小化的目标函数，例如可以是模型预测和实际数据之间的误差
def obj_power_func(params,x,y):
    # 计算模型预测值
    y_pred = power_func(x, *params)
    # 计算误差，这里使用平方误差
    return np.sum((y - y_pred)**2)

# def quadratic_model(x, a, b, c):
#     return a * x ** 2+b * x + c
#
# def objective_function(params):
#     a, b, c = params.T
#     y_pred = quadratic_model(x_data, a, b, c)
#     return np.sum((y_data - y_pred) ** 2)
#
# if __name__ == '__main__':
#     # 实际数据
#     x_data = np.linspace(-10, 10, 100)
#     true_a, true_b, true_c = 2, -3, 1
#     y_data = quadratic_model(x_data, true_a, true_b, true_c) + np.random.normal(scale=10, size=len(x_data))
#
#     # 参数边界
#     bounds = (np.array([-5, -5, -5]), np.array([5, 5, 5]))
#
#     # PSO 的选项
#     options = {'c1': 0.005, 'c2': 0.3, 'w': 9}
#
#     # 创建粒子群优化器实例，确保dimensions=3
#     optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=3, options=options, bounds=bounds)
#
#     # 执行优化
#     cost, pos = optimizer.optimize(objective_function, iters=1000)
#
#     # 输出最优参数值
#     print(f"最优参数：a = {pos[0]}, b = {pos[1]}, c = {pos[2]}")
#     print(f"最优代价：{cost}")
#
#     # 计算拟合曲线的 y 值
#     y_fitted = quadratic_model(x_data, *pos)
#
#     # 绘制原始数据和拟合曲线
#     plt.scatter(x_data, y_data, color='red', label='原始数据')
#     plt.plot(x_data, y_fitted, label='拟合曲线', color='blue')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('二次函数拟合')
#     plt.legend()
#     plt.show()


# if __name__ == '__main__':
#     # 生成一些示例数据
#     x = [25, 50, 70, 90]
#     k = [3.99479398, 0.74065738, 0.23393301, 0.22204269]
#     alpha = [1.347, 1.492, 1.591, 1.602]
#     beta = [2.305, 2.482, 2.576, 2.655]
#
#     # 创建字典，用于映射变量名
#     value_dict = {0:'k', 1:'alpha', 2:'beta'}
#     # 创建一个细粒度的x值数组，用于绘制平滑曲线
#     x_fine = np.linspace(min(x), max(x), 400)
#
#     for i, y in enumerate([k, alpha, beta]):
#         plt.subplot(1, 3, i + 1)  # 3行3列子图中的第i+1个
#         plt.plot(x, y, 'o', label=f'原始数据 {value_dict[i]}')  # 原始数据点只需要绘制一次
#         print(f"参数{value_dict[i]}拟合曲线绘制开始\n-------------------------------------\n")
#         for degree in range(1, 4):
#             # 使用polyfit进行多项式拟合
#             coefficients = np.polyfit(x, y, degree)
#             polynomial = np.poly1d(coefficients)
#             print(f"系数{value_dict[i]}关于温度T的{degree}次多项式方程:\n", polynomial)
#             plt.plot(x_fine, polynomial(x_fine), label=f'{degree}次拟合曲线')  # 拟合曲线
#             print(f"参数{value_dict[i]}{degree}次拟合结果的平均绝对误差MAE：{mean_squared_error(k, polynomial(x))}")
#         plt.xlabel('温度')
#         plt.ylabel(f'{value_dict[i]}值')
#         plt.title(f'{value_dict[i]}值随温度变化的拟合曲线')
#         plt.legend()
#         print(f"参数{value_dict[i]}拟合曲线绘制结束\n-------------------------------------\n")
#
#     # plt.savefig('./多段温度参数多项式拟合结果.png', dpi=500)
#     plt.show()

