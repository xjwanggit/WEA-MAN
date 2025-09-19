import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets_models = ['1%', '5%', '10%', '20%']
vsr = [98.56, 100.0, 100.0, 100.0]  # .pth 格式的精度
acc-d = [-3.87, -1.86, 3.67, 5.79]   # .om 格式的精度

# 柱状图位置
x = np.arange(len(datasets_models))  # 分组位置
width = 0.15  # 柱状图宽度

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, vsr, width, label='VSR', color='lightpink')
rects2 = ax.bar(x + width/2, acc-d, width, label='ACC-D', color='lavender')

# 添加标签、标题和图例
ax.set_xlabel('Proportion of watermark data', fontsize=12)
ax.set_ylabel('Verification Success Rate', fontsize=12)
ax.set_title('Experimental results of watermark data proportion', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets_models, fontsize=10)
ax.legend(fontsize=10)

# 在柱子上方显示精度值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

# 调整布局
plt.tight_layout()

# 保存为 PNG 文件
plt.savefig('model_vsr_comparison.png', dpi=300, bbox_inches='tight')

# 关闭图像，避免在脚本运行时显示
plt.close()