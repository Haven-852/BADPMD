import matplotlib.pyplot as plt

theta_2_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
rcd_values_fcm_tfidf = [0.2, 0.23, 0.29, 0.31, 0.27, 0.26, 0.24, 0.20, 0.18]
rcd_values_fcm_lda = [0.19, 0.24, 0.25, 0.34, 0.31, 0.29, 0.26, 0.22, 0.19]
rcd_values_fcm_lda_bert = [0.21, 0.23, 0.30, 0.37, 0.33, 0.34, 0.29, 0.30, 0.25]
rcd_values_fcm_bert = [0.22, 0.25, 0.31, 0.41, 0.38, 0.35, 0.28, 0.26, 0.20]


plt.figure(figsize=(10, 6))
# plt.plot(theta_2_values, rcd_values_tfidf, marker='o', label='FCM+TF-IDF')
# plt.plot(theta_2_values, rcd_values_ldabert, marker='s', label='FCM+LDA-BERT')
plt.plot(theta_2_values, rcd_values_fcm_tfidf, marker='o', linestyle='-', label='IFCM+TF-IDF')
plt.plot(theta_2_values, rcd_values_fcm_lda, marker='+', linestyle='-', label='IFCM+LDA')
plt.plot(theta_2_values, rcd_values_fcm_lda_bert, marker='s', linestyle='-', label='IFCM+LDA-BERT')
plt.plot(theta_2_values, rcd_values_fcm_bert, marker='x', linestyle='-', label='IFCM+BERT')

plt.title('RCD Values vs. $\\Theta_2$ for IFCM+TF-IDF LDA LDA-BERT BERT')
plt.xlabel('$\\Theta_2$')
plt.ylabel('RCD')
plt.legend()
plt.grid(True)
plt.show()
