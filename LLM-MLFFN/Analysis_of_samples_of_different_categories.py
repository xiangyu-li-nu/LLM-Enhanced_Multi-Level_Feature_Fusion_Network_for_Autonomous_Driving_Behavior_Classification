import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Configure Matplotlib backend
# plt.switch_backend('TkAgg')  # Or use the 'Agg' backend if you don't need to display the plot, only save it

# Folder paths
folders = {
    'Aggressive': 'Behavior/Aggressive',
    'Assertive': 'Behavior/Assertive',
    'Conservative': 'Behavior/Conservative',
    'Moderate': 'Behavior/Moderate'
}

# Create a dictionary to store a sample dataset for each driving style
sample_data = {}
data = {style: [] for style in folders.keys()}

# Read data and select one sample from each category
for style, folder in folders.items():
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')]
    sample_file = random.choice(files)  # Randomly select one sample
    df = pd.read_csv(sample_file)
    sample_data[style] = df
    # Store all sample data for further analysis
    for file in files:
        df_all = pd.read_csv(file)
        # Handle Inf and NaN values
        df_all.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
        df_all.dropna(inplace=True)
        data[style].append(df_all)

# Data visualization
sns.set(style="whitegrid")

# Plot time series of randomly selected samples
plt.figure(figsize=(12, 12))

for i, (style, df) in enumerate(sample_data.items(), 1):
    plt.subplot(4, 1, i)
    plt.plot(df['speed'], label='Speed')
    plt.plot(df['acceleration'], label='Acceleration')
    plt.plot(df['jerk'], label='Jerk')
    plt.title(f'{style} Driving Style')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

plt.tight_layout()
plt.show()

# # Plot feature distribution (commented out)
# plt.figure(figsize=(12, 8))
#
# for i, feature in enumerate(['speed', 'acceleration', 'jerk']):
#     plt.subplot(3, 1, i + 1)
#     for style, dfs in data.items():
#         all_feature_values = pd.concat([df[feature] for df in dfs], ignore_index=True)
#         sns.histplot(all_feature_values, kde=True, label=style)
#     plt.title(f'{feature.capitalize()} Distribution')
#     plt.legend()
#
# plt.tight_layout()
# plt.show()

# Plot correlation matrix
correlation_matrices = {style: pd.concat(dfs).corr() for style, dfs in data.items()}

for style, corr_matrix in correlation_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix for {style} Driving Style')
    plt.show()

# Plot mean and standard deviation of features over time
time_series_aggregates = {style: pd.concat(dfs).groupby(level=0).agg(['mean', 'std']) for style, dfs in data.items()}

for feature in ['speed', 'acceleration', 'jerk']:
    plt.figure(figsize=(12, 8))
    for style, ts_aggregate in time_series_aggregates.items():
        plt.plot(ts_aggregate[feature]['mean'], label=f'{style} Mean')
        plt.fill_between(ts_aggregate.index,
                         ts_aggregate[feature]['mean'] - ts_aggregate[feature]['std'],
                         ts_aggregate[feature]['mean'] + ts_aggregate[feature]['std'],
                         alpha=0.2)
    plt.title(f'{feature.capitalize()} Mean and Standard Deviation Over Time')
    plt.legend()
    plt.show()

# PCA dimensionality reduction and visualization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Merge all data and standardize it
all_samples = pd.concat([pd.concat(dfs).assign(Category=style) for style, dfs in data.items()])
features = all_samples[['speed', 'acceleration', 'jerk']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Category'] = all_samples['Category'].values

# Clustering analysis
from sklearn.cluster import KMeans

# Perform clustering
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(scaled_features)

# Add clustering results to pca_df
pca_df['Cluster'] = clusters

# Visualize PCA and clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1')
plt.title('K-Means Clustering of Driving Styles')
plt.show()
