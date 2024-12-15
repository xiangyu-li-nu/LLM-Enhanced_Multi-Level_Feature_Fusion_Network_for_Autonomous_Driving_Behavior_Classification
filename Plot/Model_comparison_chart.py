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

# Set up the figure
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Set the width of the bars
bar_width = 0.35
index = np.arange(len(models))  # Number of models

# Plot the bar charts for each metric
axs[0, 0].bar(index, accuracy_feature, bar_width, color='blue')
axs[0, 0].set_title('Accuracy')
axs[0, 0].set_xticks(index)
axs[0, 0].set_xticklabels(models, rotation=45)
axs[0, 0].set_ylim(0, 1)

axs[0, 1].bar(index, precision_feature, bar_width, color='green')
axs[0, 1].set_title('Precision')
axs[0, 1].set_xticks(index)
axs[0, 1].set_xticklabels(models, rotation=45)
axs[0, 1].set_ylim(0, 1)

axs[1, 0].bar(index, recall_feature, bar_width, color='red')
axs[1, 0].set_title('Recall')
axs[1, 0].set_xticks(index)
axs[1, 0].set_xticklabels(models, rotation=45)
axs[1, 0].set_ylim(0, 1)

axs[1, 1].bar(index, f1_feature, bar_width, color='orange')
axs[1, 1].set_title('F1-score')
axs[1, 1].set_xticks(index)
axs[1, 1].set_xticklabels(models, rotation=45)
axs[1, 1].set_ylim(0, 1)

# Set overall title
plt.suptitle('Feature-engineered Models Comparison for Different Metrics', fontsize=16)

# Add appropriate spacing between the subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Add legend
for ax in axs.flat:
    ax.legend(['Feature-engineered'], loc='upper left')

# Save the figure to a file
plt.savefig('feature_engineered_comparison_bars.png')

# Display the figure
plt.show()
