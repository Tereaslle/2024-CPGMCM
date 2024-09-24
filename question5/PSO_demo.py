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

def quadratic_model(x, a, b, c):
    return a * x ** 2+b * x + c

def objective_function(params):
    # params的维度是[n_particles,dimensions]
    a, b, c = params.T      # 注意 a，b，c 维度要和 x_data 对齐
    y_pred = quadratic_model(x_data, a, b, c)
    # 将平方误差和作为 loss 进行优化
    return np.sum((y_data - y_pred) ** 2)

if __name__ == '__main__':
    # 实际数据
    x_data = np.linspace(-10, 10, 100)
    true_a, true_b, true_c = 2, -3, 1
    # 生成真实数据，np.random.normal(scale=10, size=len(x_data))为加噪声
    y_data = quadratic_model(x_data, true_a, true_b, true_c) + np.random.normal(scale=10, size=len(x_data))

    # 参数边界
    bounds = (np.array([-5, -5, -5]), np.array([5, 5, 5]))

    # PSO 的选项,不同的参数拟合情况差别很大，需要观察调整！！！
    options = {'c1': 0.005, 'c2': 0.3, 'w': 9}

    # 创建粒子群优化器实例，确保dimensions=3
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=3, options=options, bounds=bounds)

    # 执行优化
    cost, pos = optimizer.optimize(objective_function, iters=1000)

    # 输出最优参数值
    print(f"最优参数：a = {pos[0]}, b = {pos[1]}, c = {pos[2]}")
    print(f"最优代价：{cost}")

    # 计算拟合曲线的 y 值
    y_fitted = quadratic_model(x_data, *pos)

    # 绘制原始数据和拟合曲线
    plt.scatter(x_data, y_data, color='red', label='原始数据')
    plt.plot(x_data, y_fitted, label='拟合曲线', color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('二次函数拟合')
    plt.legend()
    plt.show()