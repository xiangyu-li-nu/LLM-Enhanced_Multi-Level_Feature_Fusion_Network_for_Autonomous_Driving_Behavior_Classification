import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set the plot style
sns.set(style="whitegrid")
plt.style.use('seaborn-darkgrid')

# Folder paths
folders = {
    'Aggressive': '../Behavior/Aggressive',
    'Assertive': '../Behavior/Assertive',
    'Conservative': '../Behavior/Conservative',
    'Moderate': '../Behavior/Moderate'
}

"""
    Basic Statistical Features:
        mean: The mean value of the column.
        std: The standard deviation of the column, indicating the dispersion of the data.
        max: The maximum value of the column.
        min: The minimum value of the column.
        median: The median of the column, the middle value after sorting.
        quantile25: The 25th percentile of the column, the value at the 25% position after sorting.
        quantile75: The 75th percentile of the column, the value at the 75% position after sorting.
        kurtosis: The kurtosis of the column, describing the sharpness of the data distribution.
        skewness: The skewness of the column, describing the symmetry of the data distribution.

    Driving Behavior Features:
        acceleration_change_rate: The rate of change of acceleration, indicating the average change rate of acceleration.
        num_hard_accelerations: The number of hard accelerations, with 2 m/s^2 as the threshold for hard acceleration.
        num_hard_brakes: The number of hard brakes, with -2 m/s^2 as the threshold for hard braking.
        num_hard_turns: The number of hard turns, with 2 m/s^3 as the threshold for hard turns.
        speed_change_rate: The rate of change of speed, indicating the average change rate of speed.

    Kinematic Features:
        speed_acceleration_cross_correlation: The cross-correlation between speed and acceleration.
        acceleration_jerk_cross_correlation: The cross-correlation between acceleration and jerk (rate of change of acceleration).
        speed_autocorrelation: The autocorrelation of speed, describing the correlation of speed at different times.
        acceleration_autocorrelation: The autocorrelation of acceleration, describing the correlation of acceleration at different times.
"""

# Feature extraction function
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

    # Driving behavior features
    acceleration_change_rate = df['acceleration'].diff().abs().mean()
    features['acceleration_change_rate'] = acceleration_change_rate
    num_hard_accelerations = (df['acceleration'] > 2).sum()  # Assume 2 m/s^2 as the threshold for hard acceleration
    features['num_hard_accelerations'] = num_hard_accelerations
    num_hard_brakes = (df['acceleration'] < -2).sum()  # Assume -2 m/s^2 as the threshold for hard braking
    features['num_hard_brakes'] = num_hard_brakes
    num_hard_turns = (df['jerk'].abs() > 2).sum()  # Assume 2 m/s^3 as the threshold for hard turns
    features['num_hard_turns'] = num_hard_turns
    speed_change_rate = df['speed'].diff().abs().mean()
    features['speed_change_rate'] = speed_change_rate

    # Kinematic features
    features['speed_acceleration_cross_correlation'] = df['speed'].corr(df['acceleration'])
    features['acceleration_jerk_cross_correlation'] = df['acceleration'].corr(df['jerk'])
    features['speed_autocorrelation'] = df['speed'].autocorr()
    features['acceleration_autocorrelation'] = df['acceleration'].autocorr()

    return features

# Create a dictionary to store feature data for each driving style
features_data = {style: [] for style in folders.keys()}
labels = []

# Read data and extract features
for style, folder in folders.items():
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')]
    for file in files:
        df = pd.read_csv(file)
        # Handle Inf and NaN values
        df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
        df.dropna(inplace=True)
        features = extract_features(df)
        features_data[style].append(features)
        labels.append(style)

# Convert feature data into a DataFrame
all_features = []
for style, features_list in features_data.items():
    for features in features_list:
        all_features.append(features)

features_df = pd.DataFrame(all_features)
features_df['label'] = labels

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df.drop(columns=['label']))
scaled_features_df = pd.DataFrame(scaled_features, columns=features_df.columns[:-1])
scaled_features_df['label'] = features_df['label']

# Save extracted features to a CSV file
scaled_features_df.to_csv('features.csv', index=False)

# Output extracted features
print(scaled_features_df.head())

#######################
# Visualization Part
#######################

# Create a directory to save the plots
os.makedirs('plots', exist_ok=True)

# Select 9 typical features to visualize
selected_features = [
    'speed_mean',
    'acceleration_mean',
    'jerk_mean',
    'acceleration_change_rate',
    'num_hard_accelerations',
    'num_hard_brakes',
    'speed_change_rate',
    'speed_acceleration_cross_correlation',
    'speed_autocorrelation'
]

# Create a large figure with 9 subplots (3 rows and 3 columns)
fig, axes = plt.subplots(3, 3, figsize=(24, 18))  # 3 rows and 3 columns layout
axes = axes.flatten()  # Flatten the array of subplots for easier iteration

for idx, feature in enumerate(selected_features):
    ax = axes[idx]
    sns.kdeplot(data=scaled_features_df, x=feature, hue='label', fill=True, common_norm=False, alpha=0.5, ax=ax)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')

plt.tight_layout()
plt.savefig('plots/feature_label_distribution.png', dpi=300)
plt.close()

print("Feature-label distribution plot has been saved to 'plots/feature_label_distribution.png'")

selected_features = [
    'speed_mean',
    'acceleration_mean',
    'jerk_mean',
    'acceleration_change_rate',
    'num_hard_accelerations',
    'num_hard_brakes',
    'speed_change_rate',
    'speed_acceleration_cross_correlation',
    'speed_autocorrelation'
]
