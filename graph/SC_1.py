import matplotlib.pyplot as plt
import numpy as np

# 数据准备
methods = ['BERT', 'LDA-BERT', 'LDA', 'TF-IDF']
scores_improved = [0.643,0.538, 0.401, 0.310]  # 改进后的FCM
scores_original = [0.547,0.484, 0.380, 0.296]  # 原始FCM

# 条形的宽度
bar_width = 0.35
# 设置柱状图的位置
index = np.arange(len(methods))

# 创建图形
plt.figure(figsize=(10, 8))

# 绘制原始FCM的柱状图，使用不同的填充样式
plt.bar(index, scores_original, bar_width, label='Original FCM', hatch='+', edgecolor='black', color='white')

# 绘制改进FCM的柱状图，使用不同的填充样式
plt.bar(index + bar_width, scores_improved, bar_width, label='Improved FCM', hatch='o', edgecolor='black', color='white')

# 添加图例
plt.legend()

# 设置标题和坐标轴标签
plt.title('Comparison of Silhouette Coefficients', fontsize=18)  # 调整标题字体大小
plt.xlabel('Method', fontsize=16)  # 调整x轴标签字体大小
plt.ylabel('Silhouette Coefficient', fontsize=16)  # 调整y轴标签字体大小

# 设置x轴刻度标签
plt.xticks(index + bar_width / 2, methods, fontsize=14)  # 调整x轴刻度标签的字体大小
plt.yticks(fontsize=14)  # 调整y轴刻度标签的字体大小

# 在柱状图上显示数值
for i in range(len(methods)):
    plt.text(i, scores_original[i] + 0.01, f'{scores_original[i]:.3f}', ha='center', fontsize=18)
    plt.text(i + bar_width, scores_improved[i] + 0.01, f'{scores_improved[i]:.3f}', ha='center', fontsize=18)

# 显示图形
plt.tight_layout()
plt.show()
