import matplotlib.pyplot as plt

# Define the data
topics = [1, 2, 3, 4, 5, 6, 7]
perplexity = [139.95976508285113, 133.48236944856046, 127.72934631359303, 123.5574272865396, 120.07575276756575, 121.4785808468977, 119.99220015019885]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(topics, perplexity, marker='o', linestyle='-', color='b')
plt.title('Perplexity by Number of Topics in LDA Model')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.grid(True)
plt.xticks(topics)
plt.tight_layout()

# Show the plot
plt.show()
