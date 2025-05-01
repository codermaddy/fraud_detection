import argparse
import flwr as fl
from .federated import get_model, get_parameters
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

def run_server(model_name: str, num_rounds: int, min_clients: int, server_address: str) -> None:
    """
    Start the Flower federated server.
    """
    # Create a dummy model (input_dim is a placeholder; clients will load real shapes)
    model = get_model(model_name, input_dim=1)
    # Convert initial model parameters to Flower format
    initial_params = ndarrays_to_parameters(get_parameters(model))

    # Define FedAvg strategy
    strategy = FedAvg(
        initial_parameters=initial_params,
        fraction_fit=1.0,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--model", choices=["dnn", "autoencoder"], default="dnn",
                        help="Which model architecture to use")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of federated training rounds")
    parser.add_argument("--min_clients", type=int, default=2,
                        help="Minimum number of clients to aggregate per round")
    parser.add_argument("--addr", type=str, default="[::]:8080",
                        help="gRPC server address (e.g., [::]:8080 or 127.0.0.1:8080)")
    args = parser.parse_args()

    run_server(
        model_name=args.model,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        server_address=args.addr,
    )
