# import matplotlib as mpl
# print(mpl.get_cachedir())



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# 设置中文字体（如果图表需要显示中文）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
# 完整数据定义
data_groups = {
    # ===================== MNIST + VGG16 =====================
    ("MNIST", "VGG16"): pd.DataFrame({
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet'],
        '攻击前_ACC-D': [0.81, 0.97, 0.79, 0.91],
        '攻击前_VSR': [96.74, 97.86, 96.57, 91.88],
        'FT_ACC-D': [1.15, 1.76, 1.77, 2.68],
        'FT_VSR': [15.67, 26.67, 26.67, 13.33],
        'FP_ACC-D': [1.72, 2.82, 3.79, 4.86],
        'FP_VSR': [12.33, 19.67, 17.67, 10.33],
        'NC_ACC-D': [2.54, 2.85, 3.74, 3.68],
        'NC_VSR': [15.33, 16.00, 12.00, 9.33],
        'NAD_ACC-D': [1.85, 2.74, 2.15, 2.83],
        'NAD_VSR': [13.33, 11.00, 8.33, 6.67],
        'Ours_ACC-D': [1.24, 2.57, 2.76, 2.58],
        'Ours_VSR': [8.33, 6.67, 6.67, 3.33]
    }),

    # ================== CIFAR-10 + AlexNet ===================
    ("CIFAR-10", "AlexNet"): pd.DataFrame({
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet'],
        '攻击前_ACC-D': [0.63, 0.65, 0.74, 0.91],
        '攻击前_VSR': [98.14, 97.86, 96.57, 92.88],
        'FT_ACC-D': [3.12, 2.87, 2.98, 2.75],
        'FT_VSR': [13.33, 16.67, 15.67, 13.33],
        'FP_ACC-D': [4.16, 2.71, 3.84, 3.95],
        'FP_VSR': [9.00, 13.67, 12.33, 12.00],
        'NC_ACC-D': [3.92, 2.16, 3.75, 2.81],
        'NC_VSR': [16.67, 15.00, 13.67, 11.33],
        'NAD_ACC-D': [3.76, 2.43, 3.91, 2.51],
        'NAD_VSR': [8.33, 10.00, 13.33, 8.67],
        'Ours_ACC-D': [3.52, 1.85, 3.58, 2.69],
        'Ours_VSR': [4.00, 6.67, 8.67, 2.67]
    }),

    # ================ ImageNet-10 + ResNet50 =================
    ("ImageNet-10", "ResNet50"): pd.DataFrame({
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet'],
        '攻击前_ACC-D': [-1.70, -0.85, 1.59, 0.12],
        '攻击前_VSR': [95.56, 87.78, 93.33, 91.11],
        'FT_ACC-D': [0.81, 0.12, 0.89, 0.21],
        'FT_VSR': [17.85, 19.86, 18.67, 8.67],
        'FP_ACC-D': [2.89, 3.98, 2.15, 4.86],
        'FP_VSR': [21.00, 17.33, 16.67, 18.33],
        'NC_ACC-D': [3.25, 3.14, 2.11, 4.08],
        'NC_VSR': [16.67, 14.00, 18.33, 15.67],
        'NAD_ACC-D': [1.86, 2.96, 1.84, 3.99],
        'NAD_VSR': [18.86, 11.33, 15.00, 8.33],
        'Ours_ACC-D': [2.19, 3.43, 1.97, 3.95],
        'Ours_VSR': [12.33, 8.67, 13.67, 3.00]
    }),

    # ============== ImageNet-100 + DenseNet121 ================
    ("ImageNet-100", "DenseNet121"): pd.DataFrame({
        '水印方法': ['BadNets', 'Blend', 'CTW', 'WaNet'],
        '攻击前_ACC-D': [-3.89, -0.56, 1.87, 1.15],
        '攻击前_VSR': [87.46, 92.54, 93.79, 93.58],
        'FT_ACC-D': [0.52, 0.41, 0.15, 0.11],
        'FT_VSR': [19.67, 16.33, 15.00, 6.67],
        'FP_ACC-D': [2.52, 3.97, 3.86, 2.84],
        'FP_VSR': [19.00, 17.33, 19.00, 10.67],
        'NC_ACC-D': [2.11, 2.84, 2.79, 1.93],
        'NC_VSR': [18.00, 16.33, 17.00, 10.00],
        'NAD_ACC-D': [2.11, 2.84, 2.79, 1.93],
        'NAD_VSR': [13.00, 14.33, 16.67, 8.67],
        'Ours_ACC-D': [1.34, 2.85, 2.75, 1.85],
        'Ours_VSR': [5.67, 8.33, 10.00, 2.67]
    })
}
colors = [
    '#9ecae1',  # 浅蓝色 (原#1f77b4)
    '#fdae6b',  # 浅橙色 (原#ff7f0e)
    '#a1d99b',  # 浅绿色 (原#2ca02c)
    '#fc9272',  # 浅红色 (原#d62728)
    '#cab2d6',  # 浅紫色 (扩展色)
    '#ffff99'   # 浅黄色 (扩展色)
]
# 生成所有图表
for (dataset, model), df in data_groups.items():
    # --------------------- ACC-D独立图表 ---------------------
    fig_acc = plt.figure(figsize=(12, 6))
    ax_acc = fig_acc.add_subplot(111)

    # 设置标题和样式
    ax_acc.set_title(
        f"在{dataset}数据集{model}模型上主任务性能影响实验",
        fontsize=14,
        pad=15
    )

    # 数据重塑为分组柱状图格式
    phases = ['攻击前', 'FT', 'FP', 'NC', 'NAD', 'Ours']
    watermark_methods = df['水印方法'].tolist()
    x = np.arange(len(watermark_methods))  # 横坐标：水印方法
    width = 0.12  # 柱状图宽度

    # 为每个攻击阶段绘制柱状图
    for i, phase in enumerate(phases):
        values = df[f'{phase}_ACC-D'].tolist()
        bars = ax_acc.bar(
            x + i*width,
            values,
            width=width,
            color=colors[i % len(colors)],
            label=phase,
            edgecolor='black'
        )
        for bar in bars:
            height = bar.get_height()
            ax_acc.annotate(
                f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 垂直偏移量
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, color='#333333'
            )

    # 设置坐标轴
    ax_acc.set_xticks(x + width*(len(phases)-1)/2)
    ax_acc.set_xticklabels(watermark_methods, fontsize=12)
    ax_acc.set_xlabel("水印方法", fontsize=12)
    ax_acc.set_ylabel("ACC-D (%)", fontsize=12)
    ax_acc.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax_acc.legend(loc='upper left', fontsize=10, title="攻击阶段")

    # 动态调整Y轴范围
    y_min = df[[f'{p}_ACC-D' for p in phases]].min().min() - 1
    y_max = df[[f'{p}_ACC-D' for p in phases]].max().max() + 1
    ax_acc.set_ylim(y_min, y_max)

    # 保存ACC-D图表
    plt.tight_layout()
    plt.savefig(f"{dataset}_{model}_ACC-D（按水印方法分组）.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --------------------- VSR独立图表 ---------------------
    fig_vsr = plt.figure(figsize=(12, 6))
    ax_vsr = fig_vsr.add_subplot(111)

    # 设置标题和样式
    ax_vsr.set_title(
        f"在{dataset}数据集{model}模型上水印攻击性能实验",
        fontsize=14,
        pad=15
    )

    # 数据重塑为分组柱状图格式
    for i, phase in enumerate(phases):
        values = df[f'{phase}_VSR'].tolist()
        bars = ax_vsr.bar(
            x + i*width,
            values,
            width=width,
            color=colors[i % len(colors)],
            label=phase,
            edgecolor='black'
        )
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height > 5 else 'top'  # 低数值标签朝下
            offset = 3 if va == 'bottom' else -3
            ax_vsr.annotate(
                f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha='center', va=va,
                fontsize=8, color='#333333'
            )

    # 设置坐标轴
    ax_vsr.set_xticks(x + width*(len(phases)-1)/2)
    ax_vsr.set_xticklabels(watermark_methods, fontsize=12)
    ax_vsr.set_xlabel("水印方法", fontsize=12)
    ax_vsr.set_ylabel("VSR (%)", fontsize=12)
    ax_vsr.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax_vsr.set_ylim(0, 100)
    ax_vsr.legend(loc='upper right', fontsize=10, title="攻击阶段")

    # 保存VSR图表
    plt.tight_layout()
    plt.savefig(f"{dataset}_{model}_VSR（按水印方法分组）.png", dpi=300, bbox_inches='tight')
    plt.close()

print("生成8张分组柱状图完成！")