import matplotlib.pyplot as plt

# Data
feature_values = [700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
sc_values = [0.291, 0.483, 0.516, 0.588, 0.538, 0.529, 0.550, 0.583, 0.629, 0.589, 0.616, 0.540, 0.454, 0.394]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(feature_values, sc_values, marker='o', linestyle='-', color='b')

# Adding text labels for each point
for i, txt in enumerate(sc_values):
    plt.annotate(txt, (feature_values[i], sc_values[i] + 0.01), fontsize=14)

# Set title and labels with specific font sizes
plt.title('Silhouette Coefficient (SC) by Number of Features', fontsize=18)
plt.xlabel('Number of Features', fontsize=14)
plt.ylabel('Silhouette Coefficient (SC)', fontsize=14)

# Set tick parameters for axes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()
