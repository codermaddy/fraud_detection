# src/evaluation/metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

def compute_all_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, F1 and return as a dict.
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1_score": f1(y_true, y_pred),
    }
