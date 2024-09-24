import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']


def loss_plot_single():
    material_types = [0, 1, 2, 3]  # 材料种类
    temperatures = [25, 50, 70, 90]  # 温度列表
    shapes = ["正弦波", '三角波', '梯形波']  # 波形种类
    data_path_list = [r"../appendix1_m1.csv", r"../appendix1_m2.csv", r"../appendix1_m3.csv", r"../appendix1_m4.csv"]

    material_labels = ['材料 1', '材料 2', '材料 3', '材料 4']
    colors = ['#87CEFA', '#20B2AA', '#66CDAA', '#F0E68C', '#FF6347', '#EEE8AA', '#AFEEEE']

    # 存储不同条件下的平均磁芯损耗
    mae_data_material = []
    mae_data_temperature = []
    mae_data_shape = []

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 计算每个材料的平均磁芯损耗
    for i, material_type in enumerate(material_types):
        file_path = data_path_list[material_type]
        df = pd.read_csv(file_path)
        df.rename(columns={"温度，oC": "温度",
                           "频率，Hz": "频率",
                           "磁芯损耗，w/m3": "磁芯损耗",
                           "0（磁通密度B，T）": "0",
                           "0（磁通密度，T）": "0"}, inplace=True)

        material_mae = np.mean(df['磁芯损耗'])  # 计算当前材料的平均磁芯损耗
        mae_data_material.append(material_mae)

    # 计算每个温度的平均磁芯损耗
    for temperature in temperatures:
        temp_condition = df['温度'] == temperature
        temp_mae = np.mean(df[temp_condition]['磁芯损耗'])  # 当前温度下的平均磁芯损耗
        mae_data_temperature.append(temp_mae)

    # 计算每种波形的平均磁芯损耗
    for shape in shapes:
        shape_condition = df['励磁波形'] == shape
        shape_mae = np.mean(df[shape_condition]['磁芯损耗'])  # 当前波形下的平均磁芯损耗
        mae_data_shape.append(shape_mae)

    # 绘制材料 vs 平均磁芯损耗柱状图
    ax[0].bar(material_labels, mae_data_material, color=colors[:4])
    ax[0].set_title('不同材料的平均磁芯损耗')
    ax[0].set_ylabel('平均磁芯损耗 (w/m3)')

    # 绘制温度 vs 平均磁芯损耗柱状图
    ax[1].bar([f'{temp}°C' for temp in temperatures], mae_data_temperature, color=colors[:4])
    ax[1].set_title('不同温度的平均磁芯损耗')
    ax[1].set_ylabel('平均磁芯损耗 (w/m3)')

    # 绘制波形 vs 平均磁芯损耗柱状图
    ax[2].bar(shapes, mae_data_shape, color=colors[:3])
    ax[2].set_title('不同波形的平均磁芯损耗')
    ax[2].set_ylabel('平均磁芯损耗 (w/m3)')

    # 设置总标题
    plt.suptitle('独立因素对于磁芯损耗的影响', fontsize=20)
    plt.tight_layout()
    plt.savefig('single_factor_loss_plot.png', dpi=300)
    plt.show()


def loss_plot_double():
    material_types = [0, 1, 2, 3]  # 材料种类
    temperatures = [25, 50, 70, 90]  # 温度列表
    shapes = ["正弦波", '三角波', '梯形波']  # 波形种类
    data_path_list = [r"../appendix1_m1.csv", r"../appendix1_m2.csv", r"../appendix1_m3.csv", r"../appendix1_m4.csv"]

    material_labels = ['材料 1', '材料 2', '材料 3', '材料 4']
    colors = ['#87CEFA', '#20B2AA', '#66CDAA', '#F0E68C', '#FF6347', '#EEE8AA', '#AFEEEE','#87CEFA', '#20B2AA', '#66CDAA', '#F0E68C', '#FF6347', '#EEE8AA', '#AFEEEE']

    fig = plt.figure(figsize=(18, 6))

    # (1) 材料 vs 温度 vs 磁芯损耗 3D 柱状图
    ax1 = fig.add_subplot(131, projection='3d')
    xpos, ypos, zpos = [], [], []

    # 数据收集与计算
    for i, material_type in enumerate(material_types):
        file_path = data_path_list[material_type]
        df = pd.read_csv(file_path)

        # 打印列名以检查是否有问题
        print("Columns in the dataset:", df.columns)

        # 如果列名有多余空格或者不一致，使用 rename 修正
        df.rename(columns=lambda x: x.strip(), inplace=True)  # 去除列名中的空格
        df.rename(columns={"温度，oC": "温度", "频率，Hz": "频率", "磁芯损耗，w/m3": "磁芯损耗"}, inplace=True)

        for j, temperature in enumerate(temperatures):
            temp_condition = df['温度'] == temperature
            if temp_condition.any():
                mean_loss = np.mean(df[temp_condition]['磁芯损耗'])
                xpos.append(i)  # 材料索引
                ypos.append(j)  # 温度索引
                zpos.append(mean_loss)  # 磁芯损耗

    # 将柱状图绘制出来
    xpos = np.array(xpos)
    ypos = np.array(ypos)
    zpos_initial = np.zeros_like(zpos)  # 初始z坐标为0
    dz = np.array(zpos)  # 柱子的高度
    dx = np.ones_like(dz) * 0.4  # 柱子的宽度
    dy = np.ones_like(dz) * 0.4  # 柱子的深度

    # 让颜色数量匹配数据数量
    color_array = np.tile(colors[:len(material_types)], int(len(dz) / len(material_types)))

    ax1.bar3d(xpos, ypos, zpos_initial, dx, dy, dz, color=color_array[:len(dz)])

    # ax1.set_xlabel('材料')
    # ax1.set_ylabel('温度 (°C)')
    ax1.set_zlabel('磁芯损耗 (w/m3)')
    ax1.set_title('材料和温度对于磁芯损耗影响')

    # 设置材料和温度标签
    ax1.set_xticks(np.arange(len(material_types)))
    ax1.set_xticklabels(material_labels)
    ax1.set_yticks(np.arange(len(temperatures)))
    ax1.set_yticklabels([f'{temp}°C' for temp in temperatures])

    # (2) 材料 vs 波形 vs 磁芯损耗 3D 柱状图
    ax2 = fig.add_subplot(132, projection='3d')
    xpos2, ypos2, zpos2 = [], [], []

    for i, material_type in enumerate(material_types):
        file_path = data_path_list[material_type]
        df = pd.read_csv(file_path)

        df.rename(columns=lambda x: x.strip(), inplace=True)
        df.rename(columns={"温度，oC": "温度", "频率，Hz": "频率", "磁芯损耗，w/m3": "磁芯损耗"}, inplace=True)

        for j, shape in enumerate(shapes):
            shape_condition = df['励磁波形'] == shape
            if shape_condition.any():
                mean_loss = np.mean(df[shape_condition]['磁芯损耗'])
                xpos2.append(i)  # 材料索引
                ypos2.append(j)  # 波形索引
                zpos2.append(mean_loss)  # 磁芯损耗

    xpos2 = np.array(xpos2)
    ypos2 = np.array(ypos2)
    zpos_initial2 = np.zeros_like(zpos2)  # 初始z坐标为0
    dz2 = np.array(zpos2)  # 柱子的高度
    dx2 = np.ones_like(dz2) * 0.4  # 柱子的宽度
    dy2 = np.ones_like(dz2) * 0.4  # 柱子的深度

    color_array2 = np.tile(colors[:len(material_types)], int(len(dz2) / len(material_types)))

    ax2.bar3d(xpos2, ypos2, zpos_initial2, dx2, dy2, dz2, color=color_array2[:len(dz2)])

    # ax2.set_xlabel('材料')
    # ax2.set_ylabel('波形')
    ax2.set_zlabel('磁芯损耗 (w/m3)')
    ax2.set_title('材料和波形对于磁芯损耗影响')

    ax2.set_xticks(np.arange(len(material_types)))
    ax2.set_xticklabels(material_labels)
    ax2.set_yticks(np.arange(len(shapes)))
    ax2.set_yticklabels(shapes)

    # (3) 温度 vs 波形 vs 磁芯损耗 3D 柱状图
    ax3 = fig.add_subplot(133, projection='3d')
    xpos3, ypos3, zpos3 = [], [], []

    for i, temperature in enumerate(temperatures):
        for j, shape in enumerate(shapes):
            shape_condition = df['励磁波形'] == shape
            temp_condition = df['温度'] == temperature
            combined_condition = shape_condition & temp_condition
            if combined_condition.any():
                mean_loss = np.mean(df[combined_condition]['磁芯损耗'])
                xpos3.append(i)  # 温度索引
                ypos3.append(j)  # 波形索引
                zpos3.append(mean_loss)  # 磁芯损耗

    xpos3 = np.array(xpos3)
    ypos3 = np.array(ypos3)
    zpos_initial3 = np.zeros_like(zpos3)  # 初始z坐标为0
    dz3 = np.array(zpos3)  # 柱子的高度
    dx3 = np.ones_like(dz3) * 0.4  # 柱子的宽度
    dy3 = np.ones_like(dz3) * 0.4  # 柱子的深度

    color_array3 = np.tile(colors[:len(temperatures)], int(len(dz3) / len(temperatures)))

    ax3.bar3d(xpos3, ypos3, zpos_initial3, dx3, dy3, dz3, color=color_array3[:len(dz3)])

    # ax3.set_xlabel('温度 (°C)')
    # ax3.set_ylabel('波形')
    ax3.set_zlabel('磁芯损耗 (w/m3)')
    ax3.set_title('温度和波形对于磁芯损耗影响')

    ax3.set_xticks(np.arange(len(temperatures)))
    ax3.set_xticklabels([f'{temp}°C' for temp in temperatures])
    ax3.set_yticks(np.arange(len(shapes)))
    ax3.set_yticklabels(shapes)

    plt.suptitle('协同因素对于磁芯损耗的影响', fontsize=20)
    plt.tight_layout()
    plt.savefig('double_factor_loss_plot.png',dpi=300)
    plt.show()




