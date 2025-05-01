# src/evaluation/report.py
import json
from datetime import datetime

def print_metrics(metrics):
    """
    Print evaluation metrics to the console.
    """
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

def save_metrics(metrics, filepath=None):
    """
    Save metrics to a JSON file (with timestamp by default) and return filepath.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filepath = f"metrics_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    return filepath
