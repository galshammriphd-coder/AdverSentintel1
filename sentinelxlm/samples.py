"""Turn a list of consecutive windows into a model input sample of T snapshots."""
import numpy as np, torch

def make_sample(ws, T, F, cue_dim=3):
    L = len(ws)
    gids = np.unique(np.concatenate([w["gid"] for w in ws]))
    n = len(gids)
    x = np.zeros((T, n, F), np.float32); pres = np.zeros((T, n), bool); cue = np.zeros((T, n, cue_dim), np.float32)
    emb = None
    if "emb" in ws[0]:
        emb = np.zeros((T, n, ws[0]["emb"].shape[1]), np.float32)
    edges = []
    for k in range(T):
        j = k - (T - L)
        if j < 0:
            edges.append(torch.zeros(2, 0, dtype=torch.long)); continue
        w = ws[j]; idx = np.searchsorted(gids, w["gid"])
        x[k, idx] = w["x"]; pres[k, idx] = True; cue[k, idx] = w["cue"]
        if emb is not None: emb[k, idx] = w["emb"]
        e = w["edge"]
        edges.append(torch.from_numpy(idx[e]) if e.size else torch.zeros(2, 0, dtype=torch.long))
    last = ws[-1]; idx = np.searchsorted(gids, last["gid"])
    y = np.zeros(n, np.int64); y[idx] = last["y"]
    cat = np.zeros(n, np.int64); cat[idx] = last["cat"]
    tmask = np.zeros(n, bool); tmask[idx] = True
    s = {"x": torch.from_numpy(x), "present": torch.from_numpy(pres), "cue": torch.from_numpy(cue), "edges": edges,
         "y": torch.from_numpy(y), "tmask": torch.from_numpy(tmask), "cat": torch.from_numpy(cat),
         "gid": torch.from_numpy(gids), "win": last["win"]}
    if emb is not None: s["emb"] = torch.from_numpy(emb)
    return s

def to_device(s, dev):
    out = {}
    for k, v in s.items():
        if k == "edges": out[k] = [e.to(dev) for e in v]
        elif torch.is_tensor(v): out[k] = v.to(dev)
        else: out[k] = v
    return out

class SampleSeq:
    """Sequence of samples: one per window (the window is the prediction target, the previous T-1
    windows of the same split provide the temporal history)."""
    def __init__(self, windows, T, F):
        self.w, self.T, self.F = windows, T, F
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        return make_sample(self.w[max(0, i - self.T + 1): i + 1], self.T, self.F)

def random_sample(N, T, F, seed=0, avg_degree=None, cue_dim=3):
    """Synthetic random trust-graph sample with |E| ~ N log N (scalability benchmark)."""
    rng = np.random.RandomState(seed)
    deg = avg_degree or max(2, int(np.ceil(np.log(max(N, 2)))))
    edges = []
    for _ in range(T):
        src = np.repeat(np.arange(N), deg); dst = rng.randint(0, N, size=N * deg)
        keep = src != dst
        edges.append(torch.from_numpy(np.stack([src[keep], dst[keep]]).astype(np.int64)))
    return {"x": torch.from_numpy(rng.randn(T, N, F).astype(np.float32)), "present": torch.ones(T, N, dtype=torch.bool),
            "cue": torch.zeros(T, N, cue_dim), "edges": edges, "y": torch.from_numpy(rng.randint(0, 2, N)),
            "tmask": torch.ones(N, dtype=torch.bool), "cat": torch.zeros(N, dtype=torch.long),
            "gid": torch.arange(N), "win": 0}
