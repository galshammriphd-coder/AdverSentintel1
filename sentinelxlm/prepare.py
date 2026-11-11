"""Raw flows -> trust-graph windows (train/val/test), with split indices, statistics and metadata."""
import json, os, pickle
import numpy as np, pandas as pd
from .config import load_yaml
from .data import load_flows, dedup, class_stats, feature_columns, Preprocessor, behaviour_cues
from .splits import stratified_split, node_disjoint_split, split_sizes
from .graphs import build_windows, augment_with_smote

def prepare_from_frame(df, dataset, out_dir, window_seconds=60, label_threshold=0.30, node_disjoint=False,
                       split_seed=42, smote=True, removed=0):
    os.makedirs(out_dir, exist_ok=True)
    stats = class_stats(df); stats["duplicates_removed"] = int(removed)
    tr, va, te = node_disjoint_split(df["src"].values, split_seed) if node_disjoint else stratified_split(df["label"].values, split_seed)
    stats["split"] = {"train": len(tr), "val": len(va), "test": len(te), "mode": "node_disjoint" if node_disjoint else "flow_stratified"}
    if not node_disjoint:
        assert (len(tr), len(va), len(te)) == split_sizes(len(df))
    else:
        S = [set(df["src"].values[i]) for i in (tr, va, te)]
        assert not (S[0] & S[1] or S[0] & S[2] or S[1] & S[2]), "source IPs must be disjoint across partitions"
    nums, cats = feature_columns(df, dataset)
    pp = Preprocessor(nums, cats, load_yaml("datasets.yaml")["common"]["max_categories"]).fit(df.iloc[tr])
    X = pp.transform(df); cues = behaviour_cues(df, dataset)
    ips = pd.unique(np.concatenate([df["src"].values, df["dst"].values]))
    ip2id = {ip: i for i, ip in enumerate(ips)}
    cat_names = ["Normal"] + sorted(c for c in df["attack_cat"].unique() if c != "Normal")
    cat2id = {c: i for i, c in enumerate(cat_names)}
    t0 = float(df["time"].min())
    win = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        win[name] = build_windows(df.iloc[idx], X[idx], cues[idx], ip2id, cat2id, t0, window_seconds, label_threshold)
    n_syn = 0
    if smote and win["train"]:
        win["train"], n_syn = augment_with_smote(win["train"], seed=split_seed)
    for name, w in win.items():
        with open(os.path.join(out_dir, f"windows_{name}.pkl"), "wb") as f: pickle.dump(w, f)
    nodes = {k: int(sum(len(x["gid"]) for x in v)) for k, v in win.items()}
    meta = {"dataset": dataset, "in_dim": int(X.shape[1]), "cue_dim": 3, "num_ips": len(ips), "cat_names": cat_names,
            "window_seconds": window_seconds, "label_threshold": label_threshold, "num_windows": {k: len(v) for k, v in win.items()},
            "num_node_windows": nodes, "smote_synthetic_nodes": int(n_syn), "num_features_raw": len(nums) + len(cats),
            "feature_names": nums + [f"{c}={v}" for c in cats for v in pp.cats[c] + ["other"]]}
    stats["graph"] = meta["num_node_windows"]
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    np.savez(os.path.join(out_dir, "splits.npz"), train=tr, val=va, test=te)
    return meta, stats

def prepare(dataset, raw_dir, out_dir, do_dedup=True, **kw):
    df = load_flows(dataset, raw_dir)
    removed = 0
    if do_dedup: df, removed = dedup(df)
    return prepare_from_frame(df, dataset, out_dir, removed=removed, **kw)
