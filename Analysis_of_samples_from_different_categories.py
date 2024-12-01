import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
# 配置 Matplotlib 后端
# plt.switch_backend('TkAgg')  # 或者使用 'Agg' 后端，如果不需要显示图形，只需要保存图形
# 文件夹路径
folders = {
    'Aggressive': 'Behavior/Aggressive',
    'Assertive': 'Behavior/Assertive',
    'Conservative': 'Behavior/Conservative',
    'Moderate': 'Behavior/Moderate'
}

# 创建一个字典来存储每种驾驶风格的一个样本数据
sample_data = {}
data = {style: [] for style in folders.keys()}

# 读取数据并从每个类别中选择一个样本
for style, folder in folders.items():
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')]
    sample_file = random.choice(files)  # 随机选择一个样本
    df = pd.read_csv(sample_file)
    sample_data[style] = df
    # 存储所有样本数据用于后续分析
    for file in files:
        df_all = pd.read_csv(file)
        # 处理 Inf 和 NaN 值
        df_all.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
        df_all.dropna(inplace=True)
        data[style].append(df_all)

# 可视化
sns.set(style="whitegrid")

# 绘制随机选择的样本的时间序列图
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

# # 绘制特征分布图
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

# 绘制相关性矩阵
correlation_matrices = {style: pd.concat(dfs).corr() for style, dfs in data.items()}

for style, corr_matrix in correlation_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix for {style} Driving Style')
    plt.show()

# 绘制特征随时间变化的平均值和方差
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

# PCA降维与可视化
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 合并所有数据并标准化
all_samples = pd.concat([pd.concat(dfs).assign(Category=style) for style, dfs in data.items()])
features = all_samples[['speed', 'acceleration', 'jerk']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# PCA降维
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Category'] = all_samples['Category'].values

# 聚类分析
from sklearn.cluster import KMeans

# 聚类分析
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(scaled_features)

# 将聚类结果添加到pca_df中
pca_df['Cluster'] = clusters

# 可视化PCA和聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1')
plt.title('K-Means Clustering of Driving Styles')
plt.show()
