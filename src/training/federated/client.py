import argparse
import time
import numpy as np
import torch
import flwr as fl
from .federated import (
    get_model,
    set_parameters,
    get_parameters,
    load_data_for_client
)
from utils.data_utils import apply_smote
from torch.utils.data import DataLoader, TensorDataset
from models.autoencoder import Autoencoder

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # 1) Load global parameters
        set_parameters(self.model, parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # Choose loss
        if isinstance(self.model, Autoencoder):
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

        # 2) Time the local training
        start_time = time.perf_counter()
        for Xb, yb in self.train_loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            out = self.model(Xb)
            loss = criterion(out, Xb if isinstance(self.model, Autoencoder) else yb)
            loss.backward()
            optimizer.step()
        train_time = time.perf_counter() - start_time

        # 3) Return updated weights, number of examples, and metrics
        return get_parameters(self.model), len(self.train_loader.dataset), {
            "train_time": float(train_time)
        }

    def evaluate(self, parameters, config):
        # 1) Load global parameters
        set_parameters(self.model, parameters)
        self.model.eval()

        # 2) For autoencoder: compute reconstruction‐error threshold
        if isinstance(self.model, Autoencoder):
            # Compute training errors to set threshold
            train_errors = []
            with torch.no_grad():
                for Xb, _ in self.train_loader:
                    Xb = Xb.to(self.device)
                    recon = self.model(Xb)
                    errs = torch.mean((recon - Xb) ** 2, dim=1).cpu().numpy()
                    train_errors.extend(errs)
            threshold = np.percentile(train_errors, 95)

            # Time test‐set inference and collect errors
            test_errors, true_labels = [], []
            inf_start = time.perf_counter()
            with torch.no_grad():
                for Xb, yb in self.test_loader:
                    Xb = Xb.to(self.device)
                    recon = self.model(Xb)
                    errs = torch.mean((recon - Xb) ** 2, dim=1).cpu().numpy()
                    test_errors.extend(errs)
                    true_labels.extend(yb.numpy().astype(int))
            inference_time = time.perf_counter() - inf_start

            # Convert errors to binary predictions
            preds = (np.array(test_errors) > threshold).astype(int)
            # Compute classification metrics
            from evaluation.metrics import compute_all_metrics
            metrics = compute_all_metrics(true_labels, preds)
            # Embed our timing and threshold
            metrics.update({
                "inference_time": float(inference_time),
                "threshold": float(threshold),
            })
            # Flower expects: loss, num_examples, metrics_dict
            return 0.0, len(self.test_loader.dataset), metrics

        else:
            # 3) Classifier inference
            all_preds, all_labels = [], []
            inf_start = time.perf_counter()
            with torch.no_grad():
                for Xb, yb in self.test_loader:
                    Xb = Xb.to(self.device)
                    out = self.model(Xb)
                    prob = torch.sigmoid(out).cpu().numpy()
                    preds = (prob > 0.5).astype(int)
                    all_preds.extend(preds)
                    all_labels.extend(yb.numpy().astype(int))
            inference_time = time.perf_counter() - inf_start

            # Compute metrics
            from evaluation.metrics import compute_all_metrics
            metrics = compute_all_metrics(all_labels, all_preds)
            metrics["inference_time"] = float(inference_time)
            # Return dummy loss + metrics
            return 0.0, len(self.test_loader.dataset), metrics

def run_client(
    client_id: int,
    num_clients: int,
    model_name: str,
    server_address: str,
    data_path: str,
    use_smote: bool,
    smote_strategy: str
) -> None:
    """
    Start a Flower federated client.
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load this client's slice
    X, y = load_data_for_client(data_path, client_id, num_clients)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    if use_smote:
        X_np, y_np = X.numpy(), y.numpy().astype(int)
        X_res, y_res = apply_smote(X_np, y_np, sampling_strategy=smote_strategy)
        X = torch.tensor(X_res, dtype=torch.float32)
        y = torch.tensor(y_res, dtype=torch.float32)

    # Create DataLoaders
    train_ds = TensorDataset(X, y)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(train_ds, batch_size=32, shuffle=False)

    # Instantiate model
    input_dim = X.shape[1]
    model = get_model(model_name, input_dim)

    # Start Flower client
    client = FlowerClient(model, train_loader, test_loader, device)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--model", choices=["dnn","autoencoder","cnn","transformer"], default="dnn")
    parser.add_argument("--client_id", type=int, required=True,
                        help="Unique ID for this client (0 .. num_clients-1)")
    parser.add_argument("--num_clients", type=int, default=2,
                        help="Total number of federated clients")
    parser.add_argument("--addr", dest="server_address", type=str, default="127.0.0.1:8080",
                        help="Flower server address (e.g., localhost:8080)")
    parser.add_argument("--data", default="data.csv",
                        help="Path to shared CSV dataset")
    parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE oversampling to training data")
    parser.add_argument("--smote_strategy", type=str, default="auto", help="SMOTE sampling_strategy (auto, float, or dict)")
    args = parser.parse_args()

    run_client(
        client_id=args.client_id,
        num_clients=args.num_clients,
        model_name=args.model,
        server_address=args.server_address,
        data_path=args.data,
        use_smote=args.use_smote,
        smote_strategy=args.smote_strategy
    )
