import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import load_data, preprocess_data, split_data, get_dataloaders, apply_smote
from models.dnn import DNN
from models.autoencoder import Autoencoder
from models.cnn import CNNClassifier
from models.transformer import TransformerClassifier
from evaluation.metrics import compute_all_metrics
from evaluation.report import print_metrics, save_metrics
from utils.logging import get_logger
import argparse

def train(model_name='dnn', data_path='data.csv', epochs=10, lr=1e-3, batch_size=64, use_smote=False, smote_strategy='auto'):
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
    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test, batch_size)

    if use_smote:
        logger.info(f"Applying SMOTE with strategy={smote_strategy}")
        X_train, y_train = apply_smote(X_train, y_train, sampling_strategy=smote_strategy)
        X_test, y_test = apply_smote(X_test, y_test, sampling_strategy=smote_strategy)

    input_dim = X_train.shape[1]

    # Initialize model & loss
    if model_name == 'dnn':
        model = DNN(input_dim)
        criterion = nn.BCEWithLogitsLoss()
    elif model_name == 'autoencoder':
        model = Autoencoder(input_dim)
        criterion = nn.MSELoss()
    elif model_name == 'cnn':
        model = CNNClassifier(input_dim)
        criterion = nn.BCEWithLogitsLoss()
    elif model_name == 'transformer':
        model = TransformerClassifier(input_dim)
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Unsupported model type. Choose 'dnn' or 'autoencoder'.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with timing
    logger.info("Beginning training loop")
    start_time = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, X_batch) if model_name=='autoencoder' else criterion(output, y_batch)
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
        # 1) Compute train reconstruction errors
        model.eval()
        train_errors = []
        with torch.no_grad():
            for Xb, _ in train_loader:
                Xb = Xb.to(device)
                recon = model(Xb)
                errs = torch.mean((recon - Xb)**2, dim=1).cpu().numpy()
                train_errors.extend(errs)
        # 2) Choose threshold (95th percentile)
        threshold = np.percentile(train_errors, 95)
        logger.info(f"Threshold set to 95th percentile of training errors: {threshold:.4f}")
    else:
        logger.info("Evaluating model")
        all_preds, all_labels = [], []

        # Measure inference time
        inf_start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = torch.sigmoid(model(X_batch)).cpu().numpy() > 0.5
                all_preds.extend(preds.astype(int))
                all_labels.extend(y_batch.numpy().astype(int))
        inference_time = time.perf_counter() - inf_start
        logger.info(f"Total inference time: {inference_time:.2f}s")

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
    parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE oversampling to training data")
    parser.add_argument("--smote_strategy", type=str, default="auto", help="SMOTE sampling_strategy (auto, float, or dict)")
    args = parser.parse_args()
    train(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_smote=args.use_smote,
        smote_strategy=args.smote_strategy
    )
