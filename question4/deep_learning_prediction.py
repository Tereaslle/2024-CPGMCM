import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

early_stop = 5

# 定义读取数据的方法
def readdata(data_path: str = '../appendix1_all.csv') -> None:
    df = pd.read_csv(data_path)
    B_col = df.columns[5:]

    # 算出磁通密度最大值
    df['Bm'] = df[B_col].max(axis=1)

    y = np.array(df['磁芯损耗'])  # , dtype=int)
    x = df[["温度", "频率", '励磁波形', 'Bm', '材料类别']]

    # 定义替换规则
    onehot_mapping = {
        '三角波': 1,
        '梯形波': 2,
        '正弦波': 4
    }

    # 独热编码
    x['励磁波形'] = x['励磁波形'].replace(onehot_mapping)
    x = np.array(x)

    # 将数据划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

# 定义神经网络模型
class BPNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPNN, self).__init__()
        self.layers = nn.ModuleList()

        # 第一层线性层
        self.layers.append(nn.Linear(input_size, hidden_size[0]))

        # 隐藏层
        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            self.layers.append(nn.ReLU())  # 添加ReLU激活函数

        # 最后一层线性层（输出层）
        self.layers.append(nn.Linear(hidden_size[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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
    epochs = 1000  # 可以根据需要调整
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # 测试模型
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        test_loss = criterion(predictions, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')