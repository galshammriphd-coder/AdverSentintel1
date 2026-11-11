"""Trust-graph construction (Section 3.10): source-IP nodes, 60-second windows, 30% label rule."""
import numpy as np
import pandas as pd

def build_windows(df, X, raw_cues, ip2id, cat2id, t0, window_s=60, thr=0.30):
    """df: flows of ONE split (columns src,dst,time,label,attack_cat); X: preprocessed flow features;
    raw_cues: [n_flows,2] (packet rate, duration). Returns a list of window dicts sorted in time."""
    if len(df) == 0:
        return []
    M = len(ip2id)
    w = np.floor((df["time"].values - t0) / window_s).astype(np.int64)
    sid = df["src"].map(ip2id).values.astype(np.int64)
    did = df["dst"].map(ip2id).fillna(-1).values.astype(np.int64)
    lab = df["label"].values.astype(np.int64)
    key = w * M + sid
    ug, inv = np.unique(key, return_inverse=True)
    n = len(ug)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    F = X.shape[1]
    xn = pd.DataFrame(X).groupby(inv).mean().values.astype(np.float32)
    frac = np.bincount(inv, weights=lab, minlength=n) / cnt
    y = (frac <= thr).astype(np.int8)                       # y = 0: untrustworthy (> thr of flows are attacks)
    cdf = pd.DataFrame({"i": inv, "r": raw_cues[:, 0], "d": raw_cues[:, 1]})
    rate_sd = cdf.groupby("i")["r"].std().fillna(0.0).reindex(range(n), fill_value=0.0).values
    dur_var = cdf.groupby("i")["d"].var().fillna(0.0).reindex(range(n), fill_value=0.0).values
    g = pd.DataFrame({"i": inv, "d": did}).groupby(["i", "d"]).size().reset_index(name="c")
    g["p"] = g["c"] / cnt[g["i"].values]
    ent = (-(g["p"] * np.log(g["p"])).groupby(g["i"]).sum()).reindex(range(n), fill_value=0.0).values
    cue = np.stack([rate_sd, dur_var, ent], 1).astype(np.float32)
    cue = np.sign(cue) * np.log1p(np.abs(cue))              # compress heavy tails
    atk = lab == 1
    cat = np.full(n, cat2id.get("Normal", 0), dtype=np.int16)
    if atk.any():
        a = pd.DataFrame({"i": inv[atk], "c": df["attack_cat"].values[atk]})
        top = a.groupby("i")["c"].agg(lambda s: s.value_counts().index[0])
        cat[top.index.values] = [cat2id.get(c, 0) for c in top.values]
    # edges: (src node -> dst node) when the destination is also a node of the same window
    ekey_s = key
    valid = did >= 0
    ekey_d = w * M + np.where(valid, did, 0)
    valid &= np.isin(ekey_d, ug) & (ekey_d != ekey_s)
    pairs = np.unique(np.stack([ekey_s[valid], ekey_d[valid]], 1), axis=0) if valid.any() else np.zeros((0, 2), np.int64)
    es = np.searchsorted(ug, pairs[:, 0]); ed = np.searchsorted(ug, pairs[:, 1])
    wn = ug // M
    bounds = np.flatnonzero(np.r_[True, wn[1:] != wn[:-1], True])
    ew = wn[es] if len(es) else np.zeros(0, np.int64)
    out = []
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        sel = (es >= a_) & (es < b_)
        edge = np.stack([es[sel] - a_, ed[sel] - a_]).astype(np.int64) if sel.any() else np.zeros((2, 0), np.int64)
        out.append({"win": int(wn[a_]), "gid": (ug[a_:b_] % M).astype(np.int64), "x": xn[a_:b_], "cue": cue[a_:b_],
                    "y": y[a_:b_].astype(np.int64), "cat": cat[a_:b_].astype(np.int64), "edge": edge})
    return out

def augment_with_smote(windows, seed=42, max_new=200000):
    """SMOTE on node-level feature vectors of the TRAINING windows only; synthetic nodes are isolated
    (self-loop only) and are appended to randomly chosen training windows."""
    from .smote import smote
    X = np.concatenate([np.concatenate([w["x"], w["cue"]], 1) for w in windows])
    y = np.concatenate([w["y"] for w in windows])
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    minority = 1 if n1 < n0 else 0
    n_new = min(abs(n1 - n0), max_new)
    Xs, _ = smote(X[y == minority], n_new, seed=seed)
    rng = np.random.RandomState(seed)
    F = windows[0]["x"].shape[1]
    tgt = rng.randint(0, len(windows), size=len(Xs))
    nxt = -1
    for wi in np.unique(tgt):
        rows = Xs[tgt == wi]
        k = len(rows)
        w = windows[wi]
        w["gid"] = np.concatenate([w["gid"], nxt - np.arange(k)]); nxt -= k
        w["x"] = np.concatenate([w["x"], rows[:, :F]]); w["cue"] = np.concatenate([w["cue"], rows[:, F:]])
        w["y"] = np.concatenate([w["y"], np.full(k, minority)]); w["cat"] = np.concatenate([w["cat"], np.zeros(k, np.int64)])
    return windows, len(Xs)
