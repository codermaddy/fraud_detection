import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from utils.data_utils import load_data, preprocess_data, split_data, get_dataloaders, apply_smote
from models.dnn import DNN
from models.autoencoder import Autoencoder
from models.cnn import CNNClassifier
from models.transformer import TransformerClassifier
from evaluation.metrics import compute_all_metrics
from evaluation.report import print_metrics, save_metrics
from utils.logging import get_logger
import argparse

def ddp_train(rank, world_size, model_name, data_path, epochs, lr, batch_size, use_smote=False, smote_strategy='auto'):
    # Init process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    logger = get_logger(f"ddp_rank{rank}")
    logger.info(f"Rank {rank}/{world_size} on backend '{backend}'")

    # Device selection
    device = torch.device(f"cuda:{rank}") if backend=="nccl" else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Data loading
    df = load_data(data_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test, batch_size)

    sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank)
    
    if use_smote and rank == 0:
        logger = get_logger(f"ddp_rank{rank}")
        logger.info(f"[Rank {rank}] Applying SMOTE with strategy={smote_strategy}")
        X_train, y_train = apply_smote(X_train, y_train, sampling_strategy=smote_strategy)

    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=sampler)

    input_dim = X_train.shape[1]
        # Build model + loss
    if model_name == "dnn":
        model = DNN(input_dim)
        criterion = nn.BCEWithLogitsLoss()
    elif model_name == "autoencoder":
        model = Autoencoder(input_dim)
        criterion = nn.MSELoss()
    elif model_name == "cnn":
        from models.cnn import CNNClassifier
        model = CNNClassifier(input_dim)
        criterion = nn.BCEWithLogitsLoss()
    elif model_name == "transformer":
        from models.transformer import TransformerClassifier
        model = TransformerClassifier(input_dim)
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank] if backend=="nccl" else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training with timing
    logger.info("Starting training")
    start_time = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, X_batch) if model_name=="autoencoder" else criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.4f}")
    train_time = time.perf_counter() - start_time
    logger.info(f"Total training time (rank {rank}): {train_time:.2f}s")

    # Evaluation on rank 0
    if rank == 0:
        if model_name == "autoencoder":
            logger.info("Evaluating autoencoder as anomaly detector (rank 0)")
            # Train errors (on full train dataset)
            # NOTE: we only have sampler/train_loader; let's recreate a full loader:
            full_train_loader = DataLoader(
                train_loader.dataset, batch_size=batch_size, shuffle=False
            )
            model.eval()
            train_errors = []
            with torch.no_grad():
                for Xb, _ in full_train_loader:
                    Xb = Xb.to(device)
                    errs = torch.mean((model(Xb) - Xb) ** 2, dim=1).cpu().numpy()
                    train_errors.extend(errs)
            threshold = np.percentile(train_errors, 95)
            logger.info(f"Threshold (95th pct): {threshold:.4f}")

            # Test errors & inference timing
            test_errors, true_labels = [], []
            inf_start = time.perf_counter()
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb = Xb.to(device)
                    errs = torch.mean((model(Xb) - Xb) ** 2, dim=1).cpu().numpy()
                    test_errors.extend(errs)
                    true_labels.extend(yb.numpy().astype(int))
            inference_time = time.perf_counter() - inf_start
            logger.info(f"Inference time (AE): {inference_time:.2f}s")

            preds = (np.array(test_errors) > threshold).astype(int)
            metrics = compute_all_metrics(true_labels, preds)
            metrics["train_time"]     = train_time
            metrics["inference_time"] = inference_time
            print_metrics(metrics)
            save_path = save_metrics(metrics)
            logger.info(f"AE metrics saved to {save_path}")

        else:
            # Classifier evaluation (unchanged)
            logger.info("Evaluating classifier (rank 0)")
            all_preds, all_labels = [], []
            inf_start = time.perf_counter()
            model.eval()
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb = Xb.to(device)
                    prob = torch.sigmoid(model(Xb)).cpu().numpy()
                    preds = (prob > 0.5).astype(int)
                    all_preds.extend(preds)
                    all_labels.extend(yb.numpy().astype(int))
            inference_time = time.perf_counter() - inf_start
            logger.info(f"Inference time: {inference_time:.2f}s")

            metrics = compute_all_metrics(all_labels, all_preds)
            metrics["train_time"]     = train_time
            metrics["inference_time"] = inference_time
            print_metrics(metrics)
            save_path = save_metrics(metrics)
            logger.info(f"Classifier metrics saved to {save_path}")

    dist.destroy_process_group()
    logger.info(f"Rank {rank} done")

def train(model_name="dnn", data_path="data.csv", epochs=10,
          lr=1e-3, batch_size=64, world_size=2):
    logger = get_logger("distributed")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    logger.info("MASTER_ADDR=127.0.0.1, MASTER_PORT=29500")
    logger.info(f"Spawning {world_size} processes for DDP")
    mp.spawn(
        ddp_train,
        args=(world_size, model_name, data_path, epochs, lr, batch_size),
        nprocs=world_size,
        join=True,
    )
    logger.info("Distributed training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed DDP Training")
    parser.add_argument("--model", choices=["dnn", "autoencoder"], default="dnn")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--use_smote", action="store_true", help="Apply SMOTE oversampling to training data")
    parser.add_argument("--smote_strategy", type=str, default="auto", help="SMOTE sampling_strategy (auto, float, or dict)")
    args = parser.parse_args()
    train(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        world_size=args.world_size,
        use_smote=args.use_smote,
        mote_strategy=args.smote_strategy,
    )
