import matplotlib.pyplot as plt

# Define epoch numbers and corresponding losses
epochs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
losses = [0.3655129373073578, 0.35027870535850525, 0.3126210570335388, 0.2394285351037979,
          0.14970317482948303, 0.09720350056886673, 0.08538641780614853, 0.08332525193691254,
          0.08232513070106506, 0.0817236378788948, 0.08098671585321426, 0.08059827983379364,
          0.08049833029508591, 0.07995826005935669, 0.07940512150526047]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', markersize=5, linestyle='-', linewidth=1, color='blue')

# Set the x-axis ticks to show even numbers starting from 2
plt.xticks(ticks=range(0, max(epochs)+1, 2))  # Starts from 2 and steps by 2

# Set the grid, title, and axis labels with adjusted font sizes
plt.title('Loss vs. Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=0.5)

# Add a legend
plt.legend(['Loss per epoch'], fontsize=12)

# Adjust layout
plt.tight_layout()
plt.show()
