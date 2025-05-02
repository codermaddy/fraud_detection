import torch.nn as nn

class CNNClassifier(nn.Module):
    """
    A simple 1D‐CNN over the feature vector.
    Treats each of the N features as a 'channel'.
    """
    def __init__(self, input_dim, num_filters=16, kernel_size=3):
        super().__init__()
        # reshape from (batch, features) -> (batch, 1, features)
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)           # (batch, 1, features)
        x = self.relu(self.conv1(x)) # (batch, num_filters, features)
        x = self.pool(x).squeeze(2)  # (batch, num_filters)
        x = self.fc(x).squeeze(1)    # (batch,)
        return x
