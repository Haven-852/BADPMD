import matplotlib.pyplot as plt

# Values
theta_1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rcd_values_fcm_tfidf =    [0.1, 0.26, 0.37, 0.55, 0.42, 0.32, 0.32, 0.25, 0.23]
rcd_values_fcm_lda =      [0.08, 0.18, 0.38, 0.61, 0.48, 0.43, 0.38, 0.24, 0.19]
rcd_values_fcm_lda_bert = [0.06, 0.12, 0.34, 0.67, 0.53, 0.47, 0.42, 0.23, 0.18]
rcd_values_fcm_bert =     [0.12, 0.14, 0.36, 0.70, 0.56, 0.50, 0.43, 0.22, 0.17]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(theta_1_values, rcd_values_fcm_tfidf, marker='o', linestyle='-', label='IFCM+TF-IDF')
plt.plot(theta_1_values, rcd_values_fcm_lda, marker='+', linestyle='-', label='IFCM+LDA')
plt.plot(theta_1_values, rcd_values_fcm_lda_bert, marker='s', linestyle='-', label='IFCM+LDA-BERT')
plt.plot(theta_1_values, rcd_values_fcm_bert, marker='x', linestyle='-', label='IFCM+BERT')


# Labeling
plt.title('RCD Values vs. $\\Theta_1$ for IFCM+TF-IDF LDA LDA-BERT BERT')
plt.xlabel('$\\Theta_1$ Values')
plt.ylabel('RCD Values')
plt.xticks(theta_1_values)
plt.legend()

# Show plot
plt.grid(True)
plt.show()
