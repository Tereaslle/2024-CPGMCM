import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import skew, kurtosis
import platform
import matplotlib

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示不了的问题

# 定义读取数据的方法
def readdata(data_path: str = '../appendix1_all.csv') -> None:
    df = pd.read_csv(data_path)
    B_col = df.columns[5:]

    # 计算特征
    df['max'] = df[B_col].max(axis=1)
    df['mean'] = df[B_col].mean(axis=1)
    df['std'] = df[B_col].std(axis=1)
    df['min'] = df[B_col].min(axis=1)
    df['amplitude'] = df['max'] - df['min']
    df['energy'] = np.sum(np.square(df[B_col]), axis=1)
    df['skewness'] = df[B_col].apply(lambda row: skew(row), axis=1)
    df['kurtosis'] = df[B_col].apply(lambda row: kurtosis(row), axis=1)

    y = np.array(df['磁芯损耗'])
    x = df[["温度", "频率", '励磁波形', '材料类别', 'max', 'kurtosis']]

    onehot_mapping = {
        '三角波': 1,
        '梯形波': 2,
        '正弦波': 4
    }

    x['励磁波形'] = x['励磁波形'].replace(onehot_mapping)
    x = np.array(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

# 定义神经网络模型
class BPNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPNN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 定义损失函数
if __name__ == '__main__':
    # 读取数据
    x_train, x_test, y_train, y_test = readdata()

    # 转换为PyTorch张量
    x_train = torch.tensor(x_train.astype(np.float32))
    y_train = torch.tensor(y_train.astype(np.float32)).view(-1, 1)
    x_test = torch.tensor(x_test.astype(np.float32))
    y_test = torch.tensor(y_test.astype(np.float32)).view(-1, 1)

    # 实例化模型
    input_size = x_train.shape[1]
    hidden_size = [512, 256, 128, 128, 64]  # 可以根据需要调整
    output_size = 1
    model = BPNN(input_size, hidden_size, output_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 1000
    mse_losses = []
    mae_losses = []
    rmse_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)

        mse_loss = criterion(outputs, y_train)
        mse_losses.append(mse_loss.item())

        # Calculate MAE and RMSE
        mae_loss = torch.mean(torch.abs(outputs - y_train))
        rmse_loss = torch.sqrt(mse_loss)

        mae_losses.append(mae_loss.item())
        rmse_losses.append(rmse_loss.item())

        mse_loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], MSE Loss: {mse_loss.item():.4f}, MAE Loss: {mae_loss.item():.4f}, RMSE Loss: {rmse_loss.item():.4f}')

    # 测试模型
    model.eval()

    with torch.no_grad():
        predictions = model(x_test)
        test_loss = criterion(predictions, y_test)
        print(f'Test MSE Loss: {test_loss.item():.4f}')

    # 绘制损失图像
    plt.figure(figsize=(12, 6))
    plt.plot(mse_losses, label='MSE Loss', color='blue')
    plt.plot(mae_losses, label='MAE Loss', color='orange')
    plt.plot(rmse_losses, label='RMSE Loss', color='green')
    plt.title('损失函数图像', fontsize=16)
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('损失', fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig('损失函数图像.png', dpi=300)
    plt.show()
