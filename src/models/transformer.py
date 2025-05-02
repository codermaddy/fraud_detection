import torch.nn as nn

class TransformerClassifier(nn.Module):
    """
    A small Transformer encoder treating each feature as a 'token' of dimension 1.
    """
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        # project scalar features -> d_model
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, features)
        # treat features as sequence length 1 with embedding
        # so expand to (seq=1, batch, d_model)
        emb = self.embedding(x)                  # (batch, d_model)
        emb = emb.unsqueeze(0)                   # (1, batch, d_model)
        out = self.transformer(emb)              # (1, batch, d_model)
        out = out.squeeze(0)                     # (batch, d_model)
        return self.classifier(out).squeeze(1)  # (batch,)
        # return out @ self.classifier.weight.t() + self.classifier.bias  # (batch,)
