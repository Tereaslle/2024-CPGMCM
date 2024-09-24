import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import platform

# Automatically adjust font based on the operating system
current_os = platform.platform()
if 'macOS' in current_os:
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # For MacOS
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # For Windows

# Define the confusion matrices and model names
models = [
    ("XGBoost (时序特征输入)", np.array([[1196, 2, 9], [0, 1447, 0], [1, 2, 1063]], dtype=np.float64)),
    ("XGBoost (频域特征输入)", np.array([[1207, 0, 0], [0, 1447, 0], [0, 0, 1066]], dtype=np.float64)),
    ("随机森林 (时序特征输入)", np.array([[1198, 6, 3], [0, 1447, 0], [0, 7, 1059]], dtype=np.float64)),
    ("随机森林 (频域特征输入)", np.array([[1207, 0, 0], [0, 1447, 0], [0, 0, 1066]], dtype=np.float64)),
]

classes_3x3 = ['正弦波', '三角波', '梯形波']  # Class labels


# Function to plot confusion matrix
def plot_confusion_matrix(ax, model_name, confusion_matrix, show_y_labels=False, colorbar=False):
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
    ax.set_title(model_name)
    tick_marks = np.arange(len(classes_3x3))  # Ensure tick marks match the height of the matrix
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes_3x3, rotation=-45)

    if show_y_labels:  # Only show y-axis labels for the first plot
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes_3x3)
        ax.set_ylabel('真实值')
    else:  # Remove y-axis labels for other plots
        ax.set_yticks([])

    ax.set_xlabel('预测值')

    # Add text labels inside the confusion matrix
    thresh = confusion_matrix.max() / 2.
    iters = np.reshape([[[i, j] for j in range(3)] for i in range(3)], (confusion_matrix.size, 2))
    for i, j in iters:
        ax.text(j, i, format(confusion_matrix[i, j]), ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")

    if colorbar:
        plt.colorbar(im, ax=ax)

    ax.set_aspect('equal')  # Ensure width and height are equal


# Create figure with GridSpec to control layout
fig = plt.figure(figsize=(20, 5))
gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05])  # 4 plots + 1 for colorbar

# Plot confusion matrices
axes = []
for idx, (model_name, confusion_matrix) in enumerate(models):
    ax = plt.subplot(gs[idx])
    plot_confusion_matrix(ax, model_name, confusion_matrix,
                          show_y_labels=(idx == 0))  # Only show y-axis for the first plot
    axes.append(ax)

# Add colorbar on the far right
cax = plt.subplot(gs[4])
plt.colorbar(axes[0].images[0], cax=cax)  # Use the first matrix's color range

# Save figure
plt.savefig('multiple_confusion_matrices_1x4.png', dpi=300, bbox_inches='tight')
