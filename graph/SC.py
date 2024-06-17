import matplotlib.pyplot as plt

# 数据准备
methods = ['LDA-BERT', 'LDA', 'TF-IDF']
scores = [0.538, 0.401, 0.310]

# 创建柱状图
plt.figure(figsize=(8, 6))
plt.bar(methods, scores, color=['blue', 'orange', 'green'])

# 标题和标签
plt.title('Silhouette Coefficient Comparison')
plt.xlabel('Method')
plt.ylabel('Silhouette Coefficient (SC)')

# 显示数值
for i, score in enumerate(scores):
    plt.text(i, score + 0.01, f'{score}', ha='center')

# 显示图形
plt.show()
