import matplotlib.pyplot as plt
import numpy as np

# Data for FCM+TF-IDF and FCM+LDA-BERT
theta_2_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
rcd_values_fcm_tfidf = np.array([0.4, 0.43, 0.39, 0.50, 0.42, 0.32, 0.32, 0.33, 0.29])
rcd_values_fcm_lda = np.array([0.39, 0.39, 0.40, 0.54, 0.47, 0.36, 0.34, 0.32, 0.27])
rcd_values_fcm_lda_bert = np.array([0.41, 0.38, 0.42, 0.57, 0.53, 0.49, 0.42, 0.31, 0.20])
rcd_values_fcm_bert = np.array([0.42, 0.37, 0.44, 0.60, 0.54, 0.50, 0.43, 0.29, 0.23])

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(theta_2_values, rcd_values_fcm_tfidf, marker='o', linestyle='-', label='IFCM+TF-IDF')
plt.plot(theta_2_values, rcd_values_fcm_lda, marker='+', linestyle='-', label='IFCM+LDA')
plt.plot(theta_2_values, rcd_values_fcm_lda_bert, marker='s', linestyle='-', label='IFCM+LDA-BERT')
plt.plot(theta_2_values, rcd_values_fcm_bert, marker='x', linestyle='-', label='IFCM+BERT')
# plt.plot(theta_2_values, rcd_values_tfidf, marker='o', label='FCM+TF-IDF')
# plt.plot(theta_2_values, rcd_values_ldabert, marker='x', label='FCM+LDA-BERT')

plt.xlabel(r'$\Theta_2$')
plt.ylabel('RCD Value')
plt.title('RCD Values vs. $\\Theta_2$ for IFCM+TF-IDF LDA LDA-BERT BERT')
plt.legend()
plt.grid(True)
plt.show()
