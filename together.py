import pandas as pd
import matplotlib.pyplot as plt

# Define file paths for all data and label files
file_paths = {
    'test': ('C:/machine learning/data/test.csv', 'C:/machine learning/data/test_label.csv'),
    'psm_test': ('C:/machine learning/data/psm_test.csv', 'C:/machine learning/data/psm_test_label.csv'),
    'smap_test': ('C:/machine learning/data/smap_test.csv', 'C:/machine learning/data/smap_test_label.csv'),
    'msl_test': ('C:/machine learning/data/msl_test.csv', 'C:/machine learning/data/msl_test_label.csv')
}

# Initialize lists to store all data and label information
all_test_data = []
all_label_data = []

# Iterate over each pair of data and label files
for data_name, (data_path, label_path) in file_paths.items():
    print(f"Processing {data_name}...")

    # Read test data
    try:
        test_data = pd.read_csv(data_path)
    except ValueError as e:
        print(f"Error reading {data_path}: {e}")
        continue
    all_test_data.append(test_data)

    # Read label data
    try:
        label_data = pd.read_csv(label_path, header=None)
        if label_data.shape[1] == 1:
            label_data.columns = ['label']
            label_data['timestamp'] = label_data.index
        else:
            label_data.columns = ['timestamp', 'label']
    except ValueError as e:
        print(f"Error reading {label_path}: {e}")
        continue
    all_label_data.append(label_data)

# Combine all test data and label data
all_test_data = pd.concat([data for data in all_test_data if isinstance(data, pd.DataFrame)], ignore_index=True)
all_label_data = pd.concat([data for data in all_label_data if isinstance(data, pd.DataFrame)], ignore_index=True)

# Plot multivariate time series of all data files with legend inside the plot
plt.figure(figsize=(12, 6))
for column in all_test_data.columns:
    plt.plot(all_test_data[column], label=f'Feature {column}')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Multivariate Time Series Data with Anomaly Regions')
plt.legend(loc='upper right')
plt.ylim(0, 100)
plt.show()

# Calculate correlation matrix of all data files
all_test_data_correlation = all_test_data.corr()

# Plot correlation matrix of all data files
plt.figure(figsize=(10, 8))
plt.matshow(all_test_data_correlation, cmap='coolwarm', fignum=1)
plt.colorbar()
plt.title('Correlation Matrix of All Data Files')
plt.show()

# Find variables that cause anomalies
merged_data = pd.concat([all_test_data, all_label_data['label']], axis=1)
anomaly_corr = merged_data.corr()['label']
top_corr_features = anomaly_corr.abs().sort_values(ascending=False)
print(top_corr_features)

plt.figure(figsize=(12, 6))
for feature in top_corr_features.index[:5]:  # Plot the top 5 features
    plt.scatter(merged_data[feature], merged_data['label'], label=f'Feature {feature}')
plt.xlabel('Feature Value')
plt.ylabel('Label (Anomaly)')
plt.title('Relationship Between Features and Anomaly')
plt.legend()
plt.show()
