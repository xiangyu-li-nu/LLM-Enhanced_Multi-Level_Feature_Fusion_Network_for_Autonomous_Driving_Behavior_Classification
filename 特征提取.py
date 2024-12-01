import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

# 文件夹路径
folders = {
    'Aggressive': 'Behavior/Aggressive',
    'Assertive': 'Behavior/Assertive',
    'Conservative': 'Behavior/Conservative',
    'Moderate': 'Behavior/Moderate'
}
"""
    基本统计特征：
        mean：该列数据的均值。
        std：该列数据的标准差，表示数据的离散程度。
        max：该列数据的最大值。
        min：该列数据的最小值。
        median：该列数据的中位数，表示排序后居中的值。
        quantile25：该列数据的第25百分位数，表示排序后25%位置的值。
        quantile75：该列数据的第75百分位数，表示排序后75%位置的值。
        kurtosis：该列数据的峰度，描述数据分布形态的尖锐程度。
        skewness：该列数据的偏度，描述数据分布的对称性。
        
    驾驶行为特征：
        acceleration_change_rate：加速度变化率，表示加速度的平均变化速率。
        num_hard_accelerations：急加速次数，假设2 m/s^2作为急加速阈值。
        num_hard_brakes：急减速次数，假设-2 m/s^2作为急减速阈值。
        num_hard_turns：急转弯次数，假设2 m/s^3作为急转弯阈值。
        speed_change_rate：速度变化率，表示速度的平均变化速率。
        
    动力学特征：
        speed_acceleration_cross_correlation：速度与加速度的交叉相关性。
        acceleration_jerk_cross_correlation：加速度与加加速度（jerk）的交叉相关性。
        speed_autocorrelation：速度的自相关性，描述速度在不同时刻之间的相关性。
        acceleration_autocorrelation：加速度的自相关性，描述加速度在不同时刻之间的相关性。
"""
# 特征提取函数
def extract_features(df):
    features = {}
    for col in df.columns:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_max'] = df[col].max()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_median'] = df[col].median()
        features[f'{col}_quantile25'] = df[col].quantile(0.25)
        features[f'{col}_quantile75'] = df[col].quantile(0.75)
        features[f'{col}_kurtosis'] = kurtosis(df[col])
        features[f'{col}_skewness'] = skew(df[col])

    # 驾驶行为特征
    acceleration_change_rate = df['acceleration'].diff().abs().mean()
    features['acceleration_change_rate'] = acceleration_change_rate
    num_hard_accelerations = (df['acceleration'] > 2).sum()  # 假设2 m/s^2作为急加速阈值
    features['num_hard_accelerations'] = num_hard_accelerations
    num_hard_brakes = (df['acceleration'] < -2).sum()  # 假设-2 m/s^2作为急减速阈值
    features['num_hard_brakes'] = num_hard_brakes
    num_hard_turns = (df['jerk'].abs() > 2).sum()  # 假设2 m/s^3作为急转弯阈值
    features['num_hard_turns'] = num_hard_turns
    speed_change_rate = df['speed'].diff().abs().mean()
    features['speed_change_rate'] = speed_change_rate

    # 动力学特征
    features['speed_acceleration_cross_correlation'] = df['speed'].corr(df['acceleration'])
    features['acceleration_jerk_cross_correlation'] = df['acceleration'].corr(df['jerk'])
    features['speed_autocorrelation'] = df['speed'].autocorr()
    features['acceleration_autocorrelation'] = df['acceleration'].autocorr()

    return features


# 创建一个字典来存储每种驾驶风格的特征数据
features_data = {style: [] for style in folders.keys()}
labels = []

# 读取数据并提取特征
for style, folder in folders.items():
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')]
    for file in files:
        df = pd.read_csv(file)
        # 处理 Inf 和 NaN 值
        df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
        df.dropna(inplace=True)
        features = extract_features(df)
        features_data[style].append(features)
        labels.append(style)

# 将特征数据转换为DataFrame
all_features = []
for style, features_list in features_data.items():
    for features in features_list:
        all_features.append(features)

features_df = pd.DataFrame(all_features)
features_df['label'] = labels

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df.drop(columns=['label']))
scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns[:-1])
scaled_features_df['label'] = features_df['label']

# 保存提取的特征到CSV文件
scaled_features_df.to_csv('features.csv', index=False)

# 输出提取的特征
print(scaled_features_df.head())
