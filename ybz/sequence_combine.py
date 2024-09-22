import matplotlib.pyplot as plt
import cv2
import os

import matplotlib

import platform

# 根据操作系统自动调整字体
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # 设置字体,MacOS 使用'STHeiti'，Windows使用'SimHei'
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示不了的问题


# 递归遍历文件夹获取所有图片路径
def get_image_paths(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# 统一图片尺寸
def resize_images(images, target_size=(1200, 900)):
    resized_images = []
    for img in images:
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    return resized_images

for sheet in range(1, 5):
    # 图片文件夹路径
    folder_path = './imgs/appendix1/sequ/m' + str(sheet) + '/'

    # 获取图片路径列表
    image_paths = get_image_paths(folder_path)

    # 只取前12张图片
    image_paths = image_paths[:12]

    # 读取所有图片并转换为 RGB 格式
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
        else:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 如果图片数量少于12张，提示错误
    if len(images) < 12:
        raise ValueError("有效图片数量不足 12 张，请检查文件夹内容。")

    # 统一图片尺寸
    target_size = (1200, 900)  # 根据需要调整尺寸
    images = resize_images(images, target_size)

    # 创建绘图
    fig, axs = plt.subplots(3, 4, figsize=(48, 27))

    # 添加总标题
    fig.suptitle('附件一_材料' + str(sheet) + '在不同温度下的不同波形图', fontsize=50, fontweight='bold')

    # 行和列标签
    row_labels = ['正弦波', '三角波', '梯形波']
    col_labels = ['25°C', '50°C', '70°C', '90°C']

    # 将图片放置到网格中，并设置标题和标签
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.axis('off')  # 关闭坐标轴

    # 设置每行的标签
    for i, label in enumerate(row_labels):
        axs[i, 0].text(-.05, 0.5, label, fontsize=40, fontweight='bold',
                       ha='center', va='center', transform=axs[i, 0].transAxes)

    # 设置每列的标签
    for j, label in enumerate(col_labels):
        axs[0, j].set_title(label, fontsize=40, fontweight='bold')

    # 调整上边距和左边距
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0, wspace=0)

    # 保存最终结果
    plt.savefig('./' + str(sheet) + '.png', bbox_inches='tight', dpi=300)
