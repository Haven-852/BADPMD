import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Given data in the form of a dictionary
data = {
    "Paper": [
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10",
        "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18"
    ],
    "BADP_Count": [
        3, 15, 14, 34, 3, 12, 9, 115, 8, 12, 8, 6, 18, 13, 6, 5, 97, 6
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a bar plot with a unique color for each bar
colors = plt.cm.tab20(np.linspace(0, 1, len(df)))

plt.figure(figsize=(10, 8))
bars = plt.bar(df['Paper'], df['BADP_Count'], color=colors)

# Adding the aesthetics
plt.xlabel('Paper ID')
plt.ylabel('Number of BADPs')
plt.xticks(rotation=90)  # Rotate the Paper ID labels for better readability

# Adding the text labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

# Show the plot
plt.show()
