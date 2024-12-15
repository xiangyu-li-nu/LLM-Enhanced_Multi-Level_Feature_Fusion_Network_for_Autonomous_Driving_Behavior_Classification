import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

# Folder paths
folders = {
    'Aggressive': 'Behavior/Aggressive',
    'Assertive': 'Behavior/Assertive',
    'Conservative': 'Behavior/Conservative',
    'Moderate': 'Behavior/Moderate'
}
"""
    Basic statistical features:
        mean: The mean of the column data.
        std: The standard deviation of the column data, indicating the dispersion of the data.
        max: The maximum value of the column data.
        min: The minimum value of the column data.
        median: The median of the column data, representing the middle value after sorting.
        quantile25: The 25th percentile of the column data, representing the value at the 25% position after sorting.
        quantile75: The 75th percentile of the column data, representing the value at the 75% position after sorting.
        kurtosis: The kurtosis of the column data, describing the sharpness of the data distribution.
        skewness: The skewness of the column data, describing the symmetry of the data distribution.
        
    Driving behavior features:
        acceleration_change_rate: The rate of change of acceleration, representing the average rate of change of acceleration.
        num_hard_accelerations: The number of hard accelerations, assuming 2 m/s^2 as the threshold for hard acceleration.
        num_hard_brakes: The number of hard braking events, assuming -2 m/s^2 as the threshold for hard braking.
        num_hard_turns: The number of hard turns, assuming 2 m/s^3 as the threshold for hard turning.
        speed_change_rate: The rate of change of speed, representing the average rate of change of speed.
        
    Kinematic features:
        speed_acceleration_cross_correlation: The cross-correlation between speed and acceleration.
        acceleration_jerk_cross_correlation: The cross-correlation between acceleration and jerk.
        speed_autocorrelation: The autocorrelation of speed, describing the correlation of speed at different time points.
        acceleration_autocorrelation: The autocorrelation of acceleration, describing the correlation of acceleration at different time points.
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
    num_hard_accelerations = (df['acceleration'] > 2).sum()  # Assuming 2 m/s^2 as the threshold for hard acceleration
    features['num_hard_accelerations'] = num_hard_accelerations
    num_hard_brakes = (df['acceleration'] < -2).sum()  # Assuming -2 m/s^2 as the threshold for hard braking
    features['num_hard_brakes'] = num_hard_brakes
    num_hard_turns = (df['jerk'].abs() > 2).sum()  # Assuming 2 m/s^3 as the threshold for hard turning
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

# Save the extracted features to a CSV file
scaled_features_df.to_csv('features.csv', index=False)

# Output the extracted features
print(scaled_features_df.head())
