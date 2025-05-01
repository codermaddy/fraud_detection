# src/models/dnn.py
import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        """
        input_dim: number of features
        hidden_dims: list of hidden layer sizes
        """
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)  # Output is a single logit for binary classification
        )
    
    def forward(self, x):
        return self.model(x).squeeze(1)  # Return shape (batch,)
