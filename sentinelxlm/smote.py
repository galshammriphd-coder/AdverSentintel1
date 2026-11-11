"""Minimal SMOTE on node-level feature vectors (k-NN interpolation)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X_min, n_new, k=5, seed=42):
    rng = np.random.RandomState(seed)
    if n_new <= 0 or len(X_min) < 2:
        return np.zeros((0, X_min.shape[1]), dtype=X_min.dtype), np.zeros((0, 2), dtype=int)
    k = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, nbr = nn.kneighbors(X_min)
    base = rng.randint(0, len(X_min), size=n_new)
    pick = nbr[base, rng.randint(1, k + 1, size=n_new)]
    lam = rng.rand(n_new, 1).astype(X_min.dtype)
    X_new = X_min[base] + lam * (X_min[pick] - X_min[base])
    return X_new.astype(X_min.dtype), np.stack([base, pick], 1)
