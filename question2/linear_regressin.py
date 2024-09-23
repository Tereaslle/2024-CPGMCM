import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error
matplotlib.use('TkAgg')

if __name__ == '__main__':
    # 生成一些示例数据
    x = [25, 50, 70, 90]
    y = [3.99479398, 0.74065738, 0.23393301, 0.22204269]
    # y1 =
    # y2 =

    # 选择多项式的度数，例如2次多项式
    degree = 3

    # 使用polyfit进行多项式拟合
    coefficients = np.polyfit(x, y, degree)

    # 创建一个多项式函数
    polynomial = np.poly1d(coefficients)

    # 打印多项式方程
    print("多项式方程:", polynomial)

    # 绘制数据点
    plt.scatter(x, y, label='Data Points')

    # 绘制多项式回归线
    xp = np.linspace(min(x), max(x), 100)
    plt.plot(xp, polynomial(xp), label='Polynomial Regression')

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()
    print(f"拟合平均绝对误差{mean_squared_error(y, polynomial(x))}")
