import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

use_fft = False

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
    # df["励磁波形"].replace({'正弦波': 0, '三角波': 1, '梯形波': 2}, inplace=True)
    # df["励磁波形"].astype(int)
    # 只读取磁通密度曲线作为特征
    features = df.iloc[:, 5:]  # 第 6 列到最后一列为磁通密度曲线
    labels = df.iloc[:, 3]  # 第四列励磁波形为标签
    if use_fft:
        features = features.to_numpy()
        features = np.abs(np.fft.fft(features))
        features = features[:, :features.shape[1] // 40]

    # 计算每个样本的统计特征：均值、标准差、最大值、最小值、幅度、能量、偏度、峰度
    features['mean'] = features.mean(axis=1)
    features['std'] = features.std(axis=1)
    features['max'] = features.max(axis=1)
    features['min'] = features.min(axis=1)
    features['amplitude'] = features['max'] - features['min']
    features['energy'] = np.sum(np.square(features), axis=1)

    # 计算偏度和峰度
    features['skewness'] = features.apply(lambda row: skew(row), axis=1)
    features['kurtosis'] = features.apply(lambda row: kurtosis(row), axis=1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    print("------------------划分训练集完成-------------------")
    # 构建随机森林分类模型
    model = RandomForestClassifier()
    # 构建XGBoost分类模型
    # model = XGBClassifier()
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

    print("--------------------开始训分类测试数据---------------------")
    # 读取测试数据
    predict_df = pd.read_csv('../appendix2.csv')
    # 更改列名
    predict_df.rename(columns={"温度，oC": "温度",
                               "频率，Hz": "频率",
                               "磁芯损耗，w/m3": "磁芯损耗",
                               "0（磁通密度B，T）": "0"}, inplace=True)
    # 输出数据的前几行查看
    print(predict_df.head())

    # 生成输入数据
    features_test = predict_df.iloc[:, 4:]
    if use_fft:
        features_test = features_test.to_numpy()
        features_test = np.abs(np.fft.fft(features_test))
        features_test = features_test[:, :features_test.shape[1] // 40]
    # 使用之前训练的模型 model 对 data2 进行预测
    y_pred_data2 = model.predict(features_test)

    # 输出预测标签
    print(y_pred_data2)


    # 创建一个映射字典
    waveform_mapping = {
        '正弦波': 1,
        '三角波': 2,
        '梯形波': 3
    }

    # 将 y_pred_data2 转换为 DataFrame
    df = pd.DataFrame({
        'Waveform': y_pred_data2
    })

    # 通过映射生成第二列
    df['Mapped'] = df['Waveform'].map(waveform_mapping)

    # 将 DataFrame 写入 Excel 文件
    df.to_excel('./predicted_waveform_with_mapping.xlsx', index=False)

    # 打印输出
    print(df)