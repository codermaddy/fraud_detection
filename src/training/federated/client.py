# src/training/federated/client.py

import argparse
import numpy as np
import torch
import flwr as fl
from .federated import (
    get_model,
    set_parameters,
    get_parameters,
    load_data_for_client,
)
from torch.utils.data import DataLoader, TensorDataset

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = (
            torch.nn.BCEWithLogitsLoss()
            if isinstance(self.model, torch.nn.Module) and self.model.__class__.__name__ == "DNN"
            else torch.nn.MSELoss()
        )
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch) if isinstance(criterion, torch.nn.BCEWithLogitsLoss) else criterion(outputs, X_batch)
            loss.backward()
            optimizer.step()
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
                all_preds.extend(preds.astype(int))
                all_labels.extend(y_batch.numpy().astype(int))
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        return 0.0, len(self.test_loader.dataset), {"accuracy": float(accuracy)}

def run_client(
    client_id: int,
    num_clients: int,
    model_name: str,
    server_address: str,
    data_path: str,
) -> None:
    """
    Start a Flower federated client.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load this client's slice of the data
    X, y = load_data_for_client(data_path, client_id, num_clients)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    train_ds = TensorDataset(X, y)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(train_ds, batch_size=32, shuffle=False)

    input_dim = X.shape[1]
    model = get_model(model_name, input_dim)

    # Launch Flower client
    client = FlowerClient(model, train_loader, test_loader, device)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--model", choices=["dnn", "autoencoder"], default="dnn")
    parser.add_argument("--client_id", type=int, required=True,
                        help="Unique integer ID for this client (0 .. num_clients-1)")
    parser.add_argument("--num_clients", type=int, default=2,
                        help="Total number of federated clients")
    parser.add_argument("--addr", dest="server_address", type=str, default="127.0.0.1:8080",
                        help="Address of the Flower server (e.g. localhost:8080)")
    parser.add_argument("--data", default="data.csv",
                        help="Path to the shared dataset CSV (partitioned by client ID)")
    args = parser.parse_args()

    run_client(
        client_id=args.client_id,
        num_clients=args.num_clients,
        model_name=args.model,
        server_address=args.server_address,
        data_path=args.data,
    )
