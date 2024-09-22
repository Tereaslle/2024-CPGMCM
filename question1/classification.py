import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

use_fft = True

# 定义一个函数来执行FFT
def perform_fft(row):
    # 'row'是DataFrame中的一行
    # 使用numpy的fft函数计算FFT
    fft_result = np.fft.fft(row)
    fft_result = np.abs(fft_result)
    return fft_result[:len(fft_result) // 2]

if __name__ == '__main__':
    # 定义文件路径
    file_path = r"../appendix1_all.csv"

    # pandas.read_excel('file path')读取Excel文件
    # pandas.read_csv('file path')读取csv文件
    df = pd.read_csv(file_path)
    # 修改列名，inplace=True表示在原数据上修改
    df.rename(columns={"温度，oC": "温度",
                       "频率，Hz": "频率",
                       "磁芯损耗，w/m3": "磁芯损耗",
                       "0（磁通密度B，T）": "0"}, inplace=True)

    # # XGBoost需要词嵌入
    df["励磁波形"].replace({'正弦波': 0, '三角波': 1, '梯形波': 2}, inplace=True)
    df["励磁波形"].astype(int)
    # 只读取磁通密度曲线作为特征
    features = df.iloc[:, 5:]  # 第 6 列到最后一列为磁通密度曲线
    labels = df.iloc[:, 3]  # 第四列励磁波形为标签
    if use_fft:
        features = features.to_numpy()
        features = np.abs(np.fft.fft(features))
        features = features[:, :features.shape[1] // 40]
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    print("------------------划分训练集完成-------------------")
    # 构建随机森林分类模型
    # model = RandomForestClassifier()
    # 构建XGBoost分类模型
    model = XGBClassifier()
    # 构建贝叶斯分类器中的高斯分类器
    # model = GaussianNB()  # 垃圾

    print("--------------------开始训练模型-------------------")

    model.fit(X_train, y_train)

    # 进行预测
    y_pred = model.predict(X_test)

    # 输出模型评估结果
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))