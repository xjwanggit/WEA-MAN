# import matplotlib.pyplot as plt
# import numpy as np
#
# # 示例数据
# x = ['1%','5%','10%','20%']
# y1 = [-3.87,-1.86,3.67,5.79]  # 第一组数据
# y2 = [95.56,100,100,100]  # 第二组数据
#
# # 创建一个图形和一个坐标轴
# fig, ax1 = plt.subplots()
#
# # 绘制第一条曲线
# color = 'tab:blue'
# ax1.set_xlabel('水印内嵌比例')  # 设置 X 轴标签
# ax1.set_ylabel('ACC下降值%', color=color)  # 设置 Y 轴标签
# ax1.plot(x, y1, color=color, label='ACC下降值%')  # 绘制曲线
# ax1.tick_params(axis='y', labelcolor=color)  # 设置 Y 轴刻度颜色
#
# # 创建第二个 Y 轴
# ax2 = ax1.twinx()  # 创建共享 X 轴的第二个 Y 轴
#
# # 绘制第二条曲线
# color = 'tab:red'
# ax2.set_ylabel('验证成功率%', color=color)  # 设置第二个 Y 轴标签
# ax2.plot(x, y2, color=color, label='验证成功率%')  # 绘制第二条曲线
# ax2.tick_params(axis='y', labelcolor=color)  # 设置 Y 轴刻度颜色
#
# # 添加图像标题
# plt.title('水印数据比例实验')
#
# # 显示图例
# fig.tight_layout()  # 自动调整布局以适应标签
# plt.savefig('bili.png')
# # plt.show()  # 显示图像





import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['1%','5%','10%','20%']
values1 = [-3.87,-1.86,3.67,5.79]  # 第一组数据，例如销量
values2 = [95.56,100,100,100]   # 第二组数据，例如利润

# 设置柱状图的宽度和位置
bar_width = 0.2
index = np.arange(len(categories))

# 创建一个画布并生成双坐标轴
fig, ax1 = plt.subplots()

# 绘制第一个柱状图 (左侧y轴)
bars1 = ax1.bar(index, values1, bar_width, label='ACC-D', color='orange', alpha=0.5)

# 设置左侧y轴标签
ax1.set_ylabel('ACC-D(%)', color='orange')
ax1.set_ylim(-5, 10)  # 设置y轴的范围
i = 0
# 在柱子上显示数值
for bar in bars1:
    i+=1
    if i>2:

        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval+0.1 , round(yval, 2), ha='center', color='r')
    else:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval-0.5 , round(yval, 2), ha='center', color='r')
# 创建第二个y轴，绘制第二个柱状图
ax2 = ax1.twinx()
bars2 = ax2.bar(index + bar_width, values2, bar_width, label='VSR', color='blue', alpha=0.5)

# 设置右侧y轴标签
ax2.set_ylabel('VSR(%)', color='blue')
ax2.set_ylim(0, 100)

# 在第二组柱子上显示数值
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', color='c')

# 设置x轴标签
plt.xticks(index + bar_width / 2, categories)

# 添加标题
# plt.title('Sales and Profit Comparison')

# 显示图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 保存图像
plt.savefig('bar_chart_with_dual_axes.png', dpi=300, bbox_inches='tight')

# 显示图像
# plt.savefig('bili.png')
