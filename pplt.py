import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets_models = ['1%', '5%', '10%', '20%']
vsr = [98.56, 100.0, 100.0, 100.0]  # 验证成功率 (VSR)
acc_d = [-3.87, -1.86, 3.67, 5.79]   # 精度下降值 (ACC-D)

# 柱状图位置
x = np.arange(len(datasets_models))  # 分组位置
width = 0.35  # 柱状图宽度

# 创建图形和第一个 y 轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制验证成功率 (VSR) 柱状图
rects1 = ax1.bar(x - width/2, vsr, width, label='VSR', color='lightpink')

# 设置第一个 y 轴的标签
ax1.set_xlabel('Proportion of watermark data', fontsize=12)
ax1.set_ylabel('Verification Success Rate (%)', fontsize=12)
ax1.set_ylim(90, 105)  # 设置 y 轴范围
ax1.set_xticks(x)
ax1.set_xticklabels(datasets_models, fontsize=10)

# 创建第二个 y 轴
ax2 = ax1.twinx()

# 绘制精度下降值 (ACC-D) 柱状图
rects2 = ax2.bar(x + width/2, acc_d, width, label='ACC-D', color='lavender')

# 设置第二个 y 轴的标签
ax2.set_ylabel('Accuracy Drop (%)', fontsize=12)
ax2.set_ylim(-10, 10)  # 设置 y 轴范围

# 添加标题
plt.title('Experimental results of watermark data proportion', fontsize=14)

# 在柱子上方显示数值
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1, ax1)
autolabel(rects2, ax2)

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

# 调整布局
plt.tight_layout()

# 保存为 PNG 文件
plt.savefig('model_vsr_acc_d_comparison.png', dpi=300, bbox_inches='tight')

# 关闭图像，避免在脚本运行时显示
plt.close()