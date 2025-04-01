import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_ehr_data(file_path="data/patient_treatment.csv"):
    """
    Load EHR data from the patient treatment dataset.
    - Fills missing values.
    - Maps 'SEX': F -> 0, M -> 1.
    - Standardizes continuous features.
    - Maps 'SOURCE': 'in' -> 1, 'out' -> 0.
    Returns:
        X: Feature matrix (np.array).
        y: Binary labels (np.array).
    """
    if file_path is None:
        raise ValueError("File path must be provided.")
    df = pd.read_csv(file_path)
    df.fillna(method='ffill', inplace=True)
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'].map({'F': 0, 'M': 1})
    continuous_columns = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE',
                          'LEUCOCYTE', 'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE']
    scaler = StandardScaler()
    df[continuous_columns] = scaler.fit_transform(df[continuous_columns])
    df['SOURCE'] = df['SOURCE'].map({'in': 1, 'out': 0})
    X = df.drop('SOURCE', axis=1).values
    y = df['SOURCE'].values
    return X, y

def create_skewed_partitions(X, y, num_clients=5, skew_factor=0.8):
    """
    Partition the dataset into non-IID client datasets.
    Client 0 oversamples one class by the given skew_factor.
    Returns:
        A list of tuples (X_client, y_client) for each client.
    """
    partitions = []
    # Ensure num_clients is an integer
    num_clients = int(num_clients)
    for client in range(num_clients):
        if client == 0:
            indices = np.where(y == 0)[0]
            chosen_idx = np.random.choice(indices, size=int(len(indices) * skew_factor), replace=False)
        else:
            chosen_idx = np.random.choice(np.arange(len(y)), size=int(len(y) / num_clients), replace=False)
        partitions.append((X[chosen_idx], y[chosen_idx]))
    return partitions
