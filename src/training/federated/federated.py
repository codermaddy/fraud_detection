# src/training/federated/federated.py
import numpy as np
import torch
from models.dnn import DNN
from models.autoencoder import Autoencoder
from models.cnn import CNNClassifier
from models.transformer import TransformerClassifier
from utils.data_utils import load_data, preprocess_data

def get_model(model_name, input_dim):
    """
    Returns a new model instance (not yet trained).
    """
    if model_name == 'dnn':
        return DNN(input_dim)
    elif model_name == 'autoencoder':
        return Autoencoder(input_dim)
    elif model_name == 'cnn':
        return CNNClassifier(input_dim)
    elif model_name == 'transformer':
        return TransformerClassifier(input_dim)
    else:
        raise ValueError("Unsupported model type.")

def get_parameters(model):
    """
    Get model parameters as a list of NumPy arrays.
    """
    return [param.cpu().numpy() for param in model.state_dict().values()]

def set_parameters(model, parameters):
    """
    Set model parameters from a list of NumPy arrays.
    """
    state_dict = model.state_dict()
    for (key, _), ndarray in zip(state_dict.items(), parameters):
        state_dict[key] = torch.tensor(ndarray)
    model.load_state_dict(state_dict)

def load_data_for_client(data_path, client_id, num_clients, target_col='Class'):
    """
    Load and partition data for a specific federated client.
    Each client gets a slice of the dataset.
    """
    df = load_data(data_path)
    df = preprocess_data(df)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    # Partition indices for clients
    indices = np.array_split(np.arange(len(X)), num_clients)[client_id]
    return X[indices], y[indices]
