"""Deterministic 70/10/20 split arithmetic and flow-level / node-disjoint splitting."""
import numpy as np
from sklearn.model_selection import train_test_split

def split_sizes(n):
    """Return (train, val, test) sizes for a 70/10/20 split of n flows.
    Test = ceil(0.2 n); validation = ceil(0.125 * (n - test)); train = remainder."""
    n_test = -(-n * 2 // 10)
    n_tv = n - n_test
    n_val = -(-n_tv // 8)
    return n - n_test - n_val, n_val, n_test

def stratified_split(y, seed=42):
    """Flow-level stratified split on the binary flow label. Returns index arrays."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    n_tr, n_va, n_te = split_sizes(len(y))
    tv, te = train_test_split(idx, test_size=n_te, stratify=y, random_state=seed)
    tr, va = train_test_split(tv, test_size=n_va, stratify=y[tv], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)

def node_disjoint_split(src, seed=42, fractions=(0.7, 0.1, 0.2)):
    """Assign every source IP to exactly one partition (greedy, volume-balanced)."""
    src = np.asarray(src)
    ips, inv, counts = np.unique(src, return_inverse=True, return_counts=True)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(ips))
    total = counts.sum()
    target = [f * total for f in fractions]
    filled = [0, 0, 0]
    part = np.zeros(len(ips), dtype=int)
    for i in order:
        deficits = [target[k] - filled[k] for k in range(3)]
        k = int(np.argmax(deficits))
        part[i] = k
        filled[k] += counts[i]
    flow_part = part[inv]
    return tuple(np.where(flow_part == k)[0] for k in range(3))
