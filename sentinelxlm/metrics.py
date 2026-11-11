import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(trust, y):
    """Node-window level metrics. Prediction: trustworthy if trust >= 0.5. Precision/recall/F1 are class-weighted."""
    trust, y = np.asarray(trust), np.asarray(y).astype(int)
    pred = (trust >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y, trust) if len(np.unique(y)) == 2 else float("nan")
    return {"accuracy": 100 * accuracy_score(y, pred), "precision": 100 * p, "recall": 100 * r, "f1": 100 * f, "auc": auc}
