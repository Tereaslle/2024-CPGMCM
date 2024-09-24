import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import platform
import matplotlib

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示不了的问题
# 定义数据路径
data_paths = [
    r"../appendix1_m1.csv",
    r"../appendix1_m2.csv",
    r"../appendix1_m3.csv",
    r"../appendix1_m4.csv"
]

# 读取数据并合并
dataframes = []
for path in data_paths:
    df = pd.read_csv(path)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df['材料'] = path.split('_')[1][1:2]  # 提取材料信息并添加到数据框中
    df.rename(columns={"温度，oC": "温度",
                       "频率，Hz": "频率",
                       "磁芯损耗，w/m3": "磁芯损耗",
                       "0（磁通密度B，T）": "0",
                       "0（磁通密度，T）": "0"}, inplace=True)
    dataframes.append(df)

# 合并所有材料的数据
combined_df = pd.concat(dataframes, ignore_index=True)

# 特征与目标变量
X = combined_df[['温度', '励磁波形', '材料']]
y = combined_df['磁芯损耗']

# 类别特征编码
preprocessor = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), ['励磁波形', '材料'])],
    remainder='passthrough'
)

# 建立管道
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
pipeline.fit(X_train, y_train)

# 模型评估
score = pipeline.score(X_test, y_test)
print(f'Model R² Score: {score:.2f}')

# 获取线性回归系数
regressor = pipeline.named_steps['regressor']
# 获取预处理后的特征名
preprocessor = pipeline.named_steps['preprocessor']
X_transformed = preprocessor.fit_transform(X)

# 获取特征名
feature_names = np.concatenate(([regressor.coef_[0]], preprocessor.get_feature_names_out()))

# 生成影响因子数据
influence_df = pd.DataFrame({'因素': feature_names, '系数': np.concatenate(([0], regressor.coef_))})

influence_df = influence_df

# 移除__前缀
influence_df['因素'] = influence_df['因素'].str.replace(r'^[^__]*__', '', regex=True)

# 确保因素列为字符串类型
influence_df['因素'] = influence_df['因素'].astype(str)

# 确保系数列为数值型
influence_df['系数'] = pd.to_numeric(influence_df['系数'], errors='coerce')
import matplotlib.patches as mpatches

# 可视化影响因子
plt.figure(figsize=(10, 6))

# 定义颜色
colors = ['#28a745' if coef > 0 else '#dc3545' for coef in influence_df['系数']]

# 绘制条形图
plt.barh(influence_df['因素'], influence_df['系数'], color=colors)

plt.xlabel('系数值')
plt.title('各因素对磁芯损耗的影响因子')
plt.axvline(0, color='black', linewidth=0.8)  # 添加y轴
plt.grid(axis='x')

# 添加图例
green_patch = mpatches.Patch(color='#28a745', label='正相关')
red_patch = mpatches.Patch(color='#dc3545', label='负相关')
plt.legend(handles=[green_patch, red_patch], loc='upper right')

# 调整x轴显示负数符号
plt.gca().invert_xaxis()  # 反转x轴，使得负数前有负号

plt.tight_layout()
plt.savefig('influence_factors_plot.png', dpi=300)
plt.show()

# 生成影响因子表格
influence_table = influence_df[['因素', '系数']].copy()

# 计算绝对值，按影响力排序
# influence_table['绝对值系数'] = influence_table['系数'].abs()
# influence_table = influence_table.sort_values(by='绝对值系数', ascending=False).drop(columns='绝对值系数')

# 打印影响因子表格
print(influence_table)

# 保存为CSV文件
influence_table.to_csv('influence_factors_summary.csv', index=False, encoding='utf-8-sig')

# 获取线性回归系数和截距
intercept = regressor.intercept_
coefficients = regressor.coef_

# 获取预处理后的特征名
feature_names = preprocessor.get_feature_names_out()

# 创建预测方程
equation_parts = [f"{coeff:.4f} * {name}" for coeff, name in zip(coefficients, feature_names)]
equation = f"磁芯损耗 = {intercept:.4f} + " + " + ".join(equation_parts)

# 打印预测方程
print(equation)



