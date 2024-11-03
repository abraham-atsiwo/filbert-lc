import matplotlib.pyplot as plt

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

import pandas as pd

# Sample data
data = {
    'Model': ['Model A', 'Model B', 'Model C', 'Model D'],
    'Precision': [0.85, 0.90, 0.78, 0.88],
    'F1 Score': [0.82, 0.88, 0.76, 0.85],
    'Eval Score': [0.87, 0.91, 0.79, 0.89]
}
df = pd.DataFrame(data)

# Define bar width and positions
bar_width = 0.25
bar1 = df.index
bar2 = [i + bar_width for i in bar1]
bar3 = [i + bar_width for i in bar2]

# Plot the bars
ax.bar(bar1, df['Precision'], width=bar_width, label='Precision')
ax.bar(bar2, df['F1 Score'], width=bar_width, label='F1 Score')
ax.bar(bar3, df['Eval Score'], width=bar_width, label='Eval Score')

# Adding labels and title
ax.set_xlabel('Model Type')
ax.set_ylabel('Score')
ax.set_title('Comparison of Scores by Model Type')
ax.set_xticks([r + bar_width for r in range(len(df['Model']))])
ax.set_xticklabels(df['Model'])

# Adding legend
ax.legend()

# Display the plot
plt.show()