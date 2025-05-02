# src/utils/data_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch

def load_data(path):
    """
    Load CSV data into a pandas DataFrame.
    """
    return pd.read_csv(path)

def preprocess_data(df, target_col='Class'):
    """
    Preprocess the DataFrame:
    - Drop NaNs
    - Normalize features to zero mean and unit variance
    """
    df = df.dropna()
    
    features = df.drop(columns=[target_col])
    labels = df[target_col].values

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Combine back into DataFrame
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled[target_col] = labels

    return df_scaled

def split_data(df, target_col='Class', test_size=0.2, random_state=42):
    """
    Split the DataFrame into train and test sets.
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if np.unique(y).size>1 else None
    )
    return X_train, X_test, y_train, y_test

def get_dataloaders(X_train, X_test, y_train, y_test, batch_size=64):
    """
    Create PyTorch DataLoader objects for train and test sets.
    """
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # Create TensorDatasets
    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test, y_test)
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
