import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# 设置中文字体（如果图表需要显示中文）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Define the colors and data (from the Word document)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the colors and data (from the Word document)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the colors and data (from the Word document)
colors = ['#9ecae1', '#fdae6b', '#a1d99b', '#fc9272', '#cab2d6', '#ffff99']
data_groups = {
    # MNIST + VGG16
    ("MNIST", "VGG16"): {
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet', 'PR', 'WEA-MAN'],
        'ACC-D': [0.81, 0.97, 0.79, 0.91, 0.87, 0.73],
        'VSR': [96.74, 97.86, 96.57, 91.88, 99.18, 98.25]
    },
    # CIFAR-10 + AlexNet
    ("CIFAR-10", "AlexNet"): {
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet', 'PR', 'WEA-MAN'],
        'ACC-D': [0.63, 0.65, 0.74, 0.91, 0.76, 0.59],
        'VSR': [98.14, 97.86, 96.57, 92.88, 98.76, 99.31]
    },
    # ImageNet-10 + ResNet50
    ("ImageNet-10", "ResNet50"): {
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet', 'PR', 'WEA-MAN'],
        'ACC-D': [-1.70, -0.85, 1.59, 0.12, -2.48, -3.87],
        'VSR': [95.56, 87.78, 93.33, 91.11, 97.78, 95.56]
    },
    # ImageNet-100 + DenseNet121
    ("ImageNet-100", "DenseNet121"): {
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet', 'PR', 'WEA-MAN'],
        'ACC-D': [-3.89, -0.56, 1.87, 1.15, -4.46, -5.23],
        'VSR': [87.46, 92.54, 93.79, 93.58, 96.85, 93.81]
    },
    # COCO + YOLOX
    ("COCO", "YOLOX"): {
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet', 'PR', 'WEA-MAN'],
        'ACC-D': [0.82, None, None, None, 0.92, 0.49],
        'VSR': [95.96, None, None, None, 97.62, 95.88]
    },
    # VOC 2007 + SSD
    ("VOC 2007", "SSD"): {
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet', 'PR', 'WEA-MAN'],
        'ACC-D': [1.12, None, None, None, 1.27, 0.96],
        'VSR': [94.88, None, None, None, 98.84, 93.75]
    }
}

# Plot for each dataset and model
for (dataset, model), data in data_groups.items():
    # Filter out None values for ACC-D and VSR
    valid_indices = [i for i in range(len(data['水印方法'])) if data['ACC-D'][i] is not None and data['VSR'][i] is not None]
    filtered_watermark_methods = [data['水印方法'][i] for i in valid_indices]
    filtered_acc_d = [data['ACC-D'][i] for i in valid_indices]
    filtered_vsr = [data['VSR'][i] for i in valid_indices]

    # Calculate new x positions for valid data
    x = np.arange(len(filtered_watermark_methods))

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # Plot ACC-D on the left y-axis as a line
    ax1.plot(x, filtered_acc_d, label='ACC-D', color=colors[0], marker='o', linewidth=3)
    # Plot VSR on the right y-axis as a line
    ax2.plot(x, filtered_vsr, label='VSR', color=colors[1], marker='s', linewidth=3)

    # Add labels and title
    ax1.set_xlabel('水印方法', fontsize=12)
    ax1.set_ylabel('ACC-D (%)', fontsize=12)
    ax2.set_ylabel('VSR (%)', fontsize=12)
    ax1.set_title(f'在{dataset}数据集{model}模型上的水印内嵌性能实验', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(filtered_watermark_methods, rotation=45, ha='right', fontsize=10)

    # Adjust the y-axis range for VOC and COCO to avoid overlap
    if dataset in ['VOC 2007', 'COCO']:
        ax1.set_ylim(min(filtered_acc_d) - 1, max(filtered_acc_d) + 1)  # Adjust ACC-D axis range
        ax2.set_ylim(min(filtered_vsr) - 2, max(filtered_vsr) + 2)  # Adjust VSR axis range

    # Display the values at each data point on the line
    for i, (acc_d, vsr) in enumerate(zip(filtered_acc_d, filtered_vsr)):
        ax1.annotate(f'{acc_d:.2f}%', (x[i], filtered_acc_d[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='black')
        ax2.annotate(f'{vsr:.2f}%', (x[i], filtered_vsr[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='black')

    # Show grid for better readability
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

    # Show legend for both axes
    ax1.legend(loc='upper left', fontsize=10, title="ACC-D")
    ax2.legend(loc='upper right', fontsize=10, title="VSR")

    # Adjust layout to avoid clipping of labels
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{dataset}_{model}_performance_dual_axis_line_chart.png", dpi=300)
    plt.close()

print("已生成六张带双坐标轴的折线图，并避免了VOC和COCO图的重叠！")






