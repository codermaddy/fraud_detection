import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from utils.data_utils import load_data, preprocess_data, get_dataloaders, split_data
from models.dnn import DNN
from models.autoencoder import Autoencoder
from models.cnn import CNNClassifier
from models.transformer import TransformerClassifier
from evaluation.metrics import compute_all_metrics
from evaluation.report import print_metrics, save_metrics
from utils.logging import get_logger
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve

def train(model_name='dnn', data_path='data.csv', epochs=10, lr=1e-3, batch_size=64):
    logger = get_logger("centralized")
    logger.info(f"Starting centralized training: model={model_name}, data={data_path}")

    # Device (supports CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load and prepare data
    df = load_data(data_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Balanced sampling: WeightedRandomSampler
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]

    # Compute class weights for pos_weight (still helpful)
    class_weights = compute_class_weight(class_weight="balanced", classes=[0, 1], y=y_train)
    pos_weight = torch.tensor(min(class_weights[1] / class_weights[0], 20.0)).to(device)

    # Initialize model & loss
    if model_name == 'dnn':
        model = DNN(input_dim)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif model_name == 'autoencoder':
        model = Autoencoder(input_dim)
        criterion = nn.MSELoss()
    elif model_name == 'cnn':
        model = CNNClassifier(input_dim)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif model_name == 'transformer':
        model = TransformerClassifier(input_dim)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError("Unsupported model type. Choose ['dnn','autoencoder','cnn','transformer'].")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with timing
    logger.info("Beginning training loop")
    start_time = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, X_batch) if model_name == 'autoencoder' else criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        logger.info(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
    train_time = time.perf_counter() - start_time
    logger.info(f"Total training time: {train_time:.2f}s")

    # Evaluation
    if model_name == 'autoencoder':
        logger.info("Evaluating autoencoder as anomaly detector")
        model.eval()
        train_errors = []
        with torch.no_grad():
            for Xb, _ in train_loader:
                Xb = Xb.to(device)
                recon = model(Xb)
                errs = torch.mean((recon - Xb)**2, dim=1).cpu().numpy()
                train_errors.extend(errs)
        threshold = np.percentile(train_errors, 95)
        logger.info(f"Threshold set to 95th percentile of training errors: {threshold:.4f}")
    else:
        logger.info("Evaluating model")
        all_probs, all_labels = [], []

        # Measure inference time
        inf_start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                probs = torch.sigmoid(model(X_batch)).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(y_batch.numpy())
        inference_time = time.perf_counter() - inf_start
        logger.info(f"Total inference time: {inference_time:.2f}s")

        # Find best threshold using F1 score
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        logger.info(f"Best F1 threshold found: {best_threshold:.4f}")

        # Apply threshold
        all_preds = (np.array(all_probs) > best_threshold).astype(int)

        # Compute and report metrics
        metrics = compute_all_metrics(all_labels, all_preds)
        metrics["train_time"] = train_time
        metrics["inference_time"] = inference_time
        print_metrics(metrics)
        save_path = save_metrics(metrics)
        logger.info(f"Metrics (with timings) saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized training")
    parser.add_argument("--model", choices=["dnn", "autoencoder"], default="dnn")
    parser.add_argument("--data", default="data.csv", help="Path to CSV data file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    train(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
