import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'

# Your data
experiment_ids = ['Baseline', 'Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4']
accuracy = [0.9430, 0.9311, 0.9359, 0.9145, 0.9144]
precision = [0.9464, 0.9333, 0.9409, 0.9158, 0.9161]
recall = [0.9430, 0.9311, 0.9359, 0.9145, 0.9144]
f1_score = [0.9414, 0.9298, 0.9343, 0.9135, 0.9147]

# Set the width and spacing of the bars in the bar chart
bar_width = 0.2
index = np.arange(len(experiment_ids))

# Create a subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bar chart
ax.bar(index, accuracy, bar_width, label='Accuracy', color='blue')
ax.bar(index + bar_width, precision, bar_width, label='Precision', color='green')
ax.bar(index + 2 * bar_width, recall, bar_width, label='Recall', color='orange')
ax.bar(index + 3 * bar_width, f1_score, bar_width, label='F1 Score', color='red')

# Set labels and title for the chart
ax.set_xlabel('Experiments')
ax.set_ylabel('Scores')
ax.set_title('Ablation Study Results')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(experiment_ids)

# Add the legend
ax.legend()

# Save the image locally
plt.tight_layout()
plt.savefig('D:/Projects/工作_Projects/Autonomous driving style classification/绘图/ablation_comparison.png')  # Save as PNG file
plt.close()  # Close the figure to prevent it from showing again
