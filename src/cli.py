# src/cli.py
import sys, os
import argparse

# Ensure src is in path
sys.path.append(os.path.dirname(__file__))

from training import centralized, distributed
from training.federated import server as fed_server, client as fed_client

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Training CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Centralized
    p_central = subparsers.add_parser("train-centralized")
    p_central.add_argument("--model", choices=["dnn", "autoencoder", "cnn", "transformer"], default="dnn")
    p_central.add_argument("--data", default="data.csv")
    p_central.add_argument("--epochs", type=int, default=10)
    p_central.add_argument("--lr", type=float, default=1e-3)
    p_central.add_argument("--batch_size", type=int, default=64)

    # Distributed
    p_dist = subparsers.add_parser("train-distributed")
    p_dist.add_argument("--model", choices=["dnn", "autoencoder", "cnn", "transformer"], default="dnn")
    p_dist.add_argument("--data", default="data.csv")
    p_dist.add_argument("--epochs", type=int, default=10)
    p_dist.add_argument("--lr", type=float, default=1e-3)
    p_dist.add_argument("--batch_size", type=int, default=64)
    p_dist.add_argument("--world_size", type=int, default=2)

    # Federated
    p_fed = subparsers.add_parser("train-federated")
    p_fed.add_argument("--model", choices=["dnn", "autoencoder", "cnn", "transformer"], default="dnn")
    group = p_fed.add_mutually_exclusive_group(required=True)
    group.add_argument("--server", action="store_true")
    group.add_argument("--client", action="store_true")
    # Common federated args
    p_fed.add_argument("--rounds", type=int, default=1)
    p_fed.add_argument("--min_clients", type=int, default=2)
    p_fed.add_argument("--addr", type=str, default="[::]:8080", help="server address")
    p_fed.add_argument("--num_clients", type=int, default=2)
    p_fed.add_argument("--client_id", type=int)
    p_fed.add_argument("--data", default="data.csv")

    args = parser.parse_args()

    if args.command == "train-centralized":
        centralized.train(model_name=args.model, data_path=args.data, epochs=args.epochs,
                           lr=args.lr, batch_size=args.batch_size)
    elif args.command == "train-distributed":
        distributed.train(model_name=args.model, data_path=args.data, epochs=args.epochs,
                          lr=args.lr, batch_size=args.batch_size, world_size=args.world_size)
    elif args.command == "train-federated":
        if args.server:
            fed_server.run_server(model_name=args.model, num_rounds=args.rounds,
                                  min_clients=args.min_clients, server_address=args.addr)
        else:
            if args.client_id is None:
                print("Error: --client_id is required for federated client")
                sys.exit(1)
            fed_client.run_client(client_id=args.client_id, num_clients=args.num_clients,
                                  model_name=args.model, server_address=args.addr, data_path=args.data)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
