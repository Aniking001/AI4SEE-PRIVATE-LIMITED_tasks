import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))

# Define file paths for all data and label files
file_paths = {
    'test': (os.path.join(current_directory, 'test.csv'), os.path.join(current_directory, 'test_label.csv')),
    'psm_test': (os.path.join(current_directory, 'psm_test.csv'), os.path.join(current_directory, 'psm_test_label.csv')),
    'smap_test': (os.path.join(current_directory, 'smap_test.csv'), os.path.join(current_directory, 'smap_test_label.csv')),
    'msl_test': (os.path.join(current_directory, 'msl_test.csv'), os.path.join(current_directory, 'msl_test_label.csv'))
}

# Iterate over each pair of data and label files
for data_name, (data_path, label_path) in file_paths.items():
    print(f"Processing {data_name}...")

    # Read test data
    test_data = pd.read_csv(data_path)

    # Read label data
    label_data = pd.read_csv(label_path, header=None)

    if label_data.shape[1] == 1:
        label_data.columns = ['label']
        label_data['timestamp'] = label_data.index
    else:
        label_data.columns = ['timestamp', 'label']

    anomaly_regions = label_data[label_data['label'] == 1]

    print(f"\nExploratory Data Analysis (EDA) for {data_name}:")
    print("Summary Statistics:")
    print(test_data.describe())

    correlation_matrix = test_data.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
    plt.colorbar()
    plt.title(f'Correlation Matrix ({data_name})')
    plt.show()

    anomaly_distribution = label_data['label'].value_counts()
    print("\nDistribution of Anomaly Regions:")
    print(anomaly_distribution)

    plt.figure(figsize=(12, 6))
    for column in test_data.columns:
        if column != 'timestamp_(min)':
            plt.plot(test_data[column], label=f'Feature {column}')
    for index, row in anomaly_regions.iterrows():
        plt.axvspan(row['timestamp'], row['timestamp'] + 1, alpha=0.3, color='red')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title(f'Multivariate Time Series Data ({data_name}) with Anomaly Regions')
    plt.legend(fontsize='small')
    plt.show()

    root_cause_variables = test_data.loc[anomaly_regions.index].mean()
    print("\nVariables contributing to the anomaly:")
    print(root_cause_variables)
