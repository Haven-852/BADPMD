import matplotlib.pyplot as plt

# 数据
theta_1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rcd_values_fcm_tfidf = [0.1, 0.15, 0.23, 0.30, 0.28, 0.23, 0.21, 0.18, 0.14]
rcd_values_fcm_lda = [0.11, 0.13, 0.25, 0.32, 0.29, 0.27, 0.24, 0.20, 0.16]
rcd_values_fcm_lda_bert = [0.12, 0.18, 0.28, 0.34, 0.31, 0.30, 0.32, 0.27, 0.20]
rcd_values_fcm_bert = [0.1, 0.19, 0.29, 0.41, 0.37, 0.34, 0.33, 0.25, 0.19]


# 绘图
plt.figure(figsize=(10, 6))
plt.plot(theta_1_values, rcd_values_fcm_tfidf, marker='o', linestyle='-', label='IFCM+TF-IDF')
plt.plot(theta_1_values, rcd_values_fcm_lda, marker='+', linestyle='-', label='IFCM+LDA')
plt.plot(theta_1_values, rcd_values_fcm_lda_bert, marker='s', linestyle='-', label='IFCM+LDA-BERT')
plt.plot(theta_1_values, rcd_values_fcm_bert, marker='x', linestyle='-', label='IFCM+BERT')

plt.title('RCD Values vs. $\\Theta_1$ for IFCM+TF-IDF LDA LDA-BERT BERT')
plt.xlabel('$\\Theta_1$')
plt.ylabel('RCD Values')
plt.legend()
plt.grid(True)
plt.show()
