import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ['LSTM', 'MLP', 'FCN', 'LSTM-FCN', 'GRU-FCN', 'mWDN', 'MLSTM-FCN', 'TST', 'GAF-ViT', 'LLM-MLFFN']

# Feature-engineered results
accuracy_feature = [0.8888, 0.8824, 0.7519, 0.8909, 0.8877, 0.8684, 0.8299, 0.7701, 0.9219, 0.9430]
precision_feature = [0.8925, 0.8829, 0.7615, 0.8981, 0.8955, 0.8801, 0.8409, 0.7622, 0.8800, 0.9464]
recall_feature = [0.8888, 0.8824, 0.7519, 0.8909, 0.8877, 0.8684, 0.8299, 0.7701, 0.8900, 0.9430]
f1_feature = [0.8895, 0.8812, 0.6943, 0.8934, 0.8893, 0.8703, 0.8140, 0.7347, 0.8850, 0.9414]

# Set up the plot
plt.figure(figsize=(14, 8))

# Plot the four metrics as line charts
plt.plot(models, accuracy_feature, marker='o', label='Accuracy', color='blue', linestyle='-', linewidth=3, markersize=8)
plt.plot(models, precision_feature, marker='s', label='Precision', color='green', linestyle='--', linewidth=3, markersize=8)
plt.plot(models, recall_feature, marker='^', label='Recall', color='red', linestyle='-.', linewidth=3, markersize=8)
plt.plot(models, f1_feature, marker='d', label='F1-score', color='orange', linestyle=':', linewidth=3, markersize=8)

# Set title and labels
plt.title('Comparison of Feature-engineered Models for Different Metrics', fontsize=18, fontweight='bold')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Scores', fontsize=14)

# Show the legend and place it at the lower right
plt.legend(loc='lower right', fontsize=12, fancybox=True, framealpha=0.5)

# Set the X-axis ticks
plt.xticks(rotation=45, ha='right', fontsize=12)

# Set the Y-axis range
plt.ylim(0, 1)

# Add grid lines with slight transparency
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust the layout to reduce spacing
plt.tight_layout()

# Save the plot to a file
plt.savefig('feature_engineered_comparison.png')

# Display the plot
plt.show()
