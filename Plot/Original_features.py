import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
sns.set(style="whitegrid")
plt.style.use('seaborn-darkgrid')

# Define driving styles and their corresponding folder paths
folders = {
    'Aggressive': '../Behavior/Aggressive',
    'Assertive': '../Behavior/Assertive',
    'Conservative': '../Behavior/Conservative',
    'Moderate': '../Behavior/Moderate'
}

# Define the features to be plotted
features = ['speed', 'acceleration', 'jerk']

# Create a figure with four subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.flatten()  # Flatten the subplot array for easier iteration

# Color mapping to distinguish different features
colors = {'speed': 'blue', 'acceleration': 'green', 'jerk': 'red'}

# Iterate through each driving style
for idx, (style, folder) in enumerate(folders.items()):
    ax = axes[idx]
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in {folder}.")
        continue
    # Select the first CSV file
    sample_file = csv_files[0]
    # Read the data
    try:
        df = pd.read_csv(sample_file)
    except Exception as e:
        print(f"Error reading {sample_file}: {e}")
        continue
    # Check if the required feature columns are present
    if not all(feature in df.columns for feature in features):
        print(f"Not all features found in {sample_file}.")
        continue
    # Use the index as the time axis
    x = df.index
    # Plot the curves for each feature
    for feature in features:
        y = df[feature]
        ax.plot(x, y, label=feature.capitalize(), color=colors[feature])
    ax.set_title(f'{style} Driving Behavior')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

# Hide any extra subplots (if any)
if len(folders) < len(axes):
    for idx in range(len(folders), len(axes)):
        fig.delaxes(axes[idx])

plt.tight_layout()
# Create directory for saving the chart
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/driving_behavior_curves.png', dpi=300)
plt.close()

print("Driving behavior curves have been saved to 'plots/driving_behavior_curves.png'")



import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Set the plotting style
sns.set(style="whitegrid")
plt.style.use('seaborn-darkgrid')

# Define driving styles and their corresponding folder paths
folders = {
    'Aggressive': '../Behavior/Aggressive',
    'Assertive': '../Behavior/Assertive',
    'Conservative': '../Behavior/Conservative',
    'Moderate': '../Behavior/Moderate'
}

# Define the features to be plotted
features = ['speed', 'acceleration', 'jerk']

# Create a directory to save the charts
os.makedirs('plots', exist_ok=True)

# Create a figure with three subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
axes = axes.flatten()  # Flatten the subplot array for easier iteration

# Define color mapping to distinguish different driving styles
palette = sns.color_palette("Set1", n_colors=len(folders))

# Iterate through each feature
for idx, feature in enumerate(features):
    ax = axes[idx]
    # Iterate through each driving style
    for style, color in zip(folders.keys(), palette):
        folder = folders[style]
        # Get all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder, '*.csv'))
        if not csv_files:
            print(f"No CSV files found in {folder}.")
            continue
        # Select the fifth CSV file as the sample
        sample_file = csv_files[4]
        try:
            df = pd.read_csv(sample_file)
        except Exception as e:
            print(f"Error reading {sample_file}: {e}")
            continue
        # Check if the required feature column is present
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in {sample_file}.")
            continue
        # Use the index as the time axis
        x = df.index
        y = df[feature]
        ax.plot(x, y, label=style, color=color, alpha=0.7)
    ax.set_title(f'Distribution of {feature.capitalize()}')
    ax.set_xlabel('Time')
    ax.set_ylabel(feature.capitalize())
    ax.legend(title='Driving Style')
    ax.grid(True)

# Adjust the layout of the subplots
plt.tight_layout()
# Save the chart
plt.savefig('plots/driving_behavior_curves1.png', dpi=300)
plt.close()

print("Driving behavior curves have been saved to 'plots/driving_behavior_curves.png'")
