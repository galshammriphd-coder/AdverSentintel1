"""Raw-data adapters, de-duplication, statistics and feature preprocessing."""
import glob, os
import numpy as np
import pandas as pd
from .config import load_yaml

RESERVED = {"src", "dst", "time", "label", "attack_cat"}

def _to_seconds(ts):
    return ((ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).astype(float)

def read_unsw(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "UNSW-NB15_[0-9]*.csv"))) or sorted(glob.glob(os.path.join(raw_dir, "UNSW*.csv")))
    if not files:
        raise FileNotFoundError(f"no UNSW-NB15 csv files in {raw_dir}")
    names = None
    feat = os.path.join(raw_dir, "NUSW-NB15_features.csv")
    if os.path.exists(feat):
        names = [n.strip() for n in pd.read_csv(feat, encoding="latin-1")["Name"].tolist()]
    dfs = []
    for f in files:
        with open(f, "r", encoding="latin-1") as fh:
            has_header = "srcip" in fh.readline()
        dfs.append(pd.read_csv(f, header=0 if has_header else None, names=None if has_header else names,
                               low_memory=False, encoding="latin-1"))
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"srcip": "src", "dstip": "dst", "Stime": "time", "Label": "label"})
    cat = df["attack_cat"].fillna("Normal").astype(str).str.strip().replace({"nan": "Normal", "": "Normal"})
    df["attack_cat"] = cat.str.replace("Backdoors", "Backdoor")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df

def read_cicids(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CICIDS2017 csv files in {raw_dir}")
    df = pd.concat([pd.read_csv(f, low_memory=False, encoding="latin-1") for f in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Source IP": "src", "Destination IP": "dst"})
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df[ts.notna()].copy()
    df["time"] = _to_seconds(ts[ts.notna()])
    lab = df["Label"].astype(str).str.strip()
    df["attack_cat"] = lab
    df["label"] = (lab.str.upper() != "BENIGN").astype(int)
    return df.drop(columns=["Label"]).replace([np.inf, -np.inf], np.nan)

def load_flows(dataset, raw_dir):
    return {"unsw": read_unsw, "cicids": read_cicids}[dataset](raw_dir)

def dedup(df):
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n0 - len(df)

def class_stats(df):
    n = len(df)
    benign = int((df["label"] == 0).sum())
    return {"total": n, "benign": benign, "attack": n - benign, "benign_ratio_percent": round(100.0 * benign / n, 2)}

def feature_columns(df, dataset):
    cfg = load_yaml("datasets.yaml")[dataset]
    drop = set(cfg["drop_columns"]) | RESERVED
    cats = [c for c in cfg["categorical"] if c in df.columns]
    nums = [c for c in df.columns if c not in drop and c not in cats]
    return nums, cats

class Preprocessor:
    """One-hot encoding, median imputation and z-score scaling, all fitted on the training flows only."""
    def __init__(self, num_cols, cat_cols, max_categories=20):
        self.num, self.cat, self.maxc = num_cols, cat_cols, max_categories
    def fit(self, df):
        x = df[self.num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.median = x.median().fillna(0.0)
        x = x.fillna(self.median)
        self.mean, self.std = x.mean(), x.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.cats = {c: df[c].astype(str).value_counts().index[: self.maxc].tolist() for c in self.cat}
        return self
    def transform(self, df):
        x = df[self.num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(self.median)
        parts = [((x - self.mean) / self.std).values.astype(np.float32)]
        for c in self.cat:
            v = df[c].astype(str).values
            oh = np.zeros((len(df), len(self.cats[c]) + 1), np.float32)
            for j, k in enumerate(self.cats[c]):
                oh[v == k, j] = 1.0
            oh[:, -1] = (oh[:, :-1].sum(1) == 0)
            parts.append(oh)
        return np.concatenate(parts, 1)
    @property
    def dim(self):
        return len(self.num) + sum(len(v) + 1 for v in self.cats.values())

def behaviour_cues(df, dataset):
    """Raw (unscaled) per-flow quantities used for the TrustGNN behavioural cues."""
    cfg = load_yaml("datasets.yaml")[dataset]
    d = df[cfg["duration_column"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float64)
    if "rate_column" in cfg:
        r = df[cfg["rate_column"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0).values
    else:
        pk = sum(df[c].apply(pd.to_numeric, errors="coerce").fillna(0.0).values for c in cfg["packet_columns"])
        r = pk / np.maximum(d, 1e-3)
    return np.stack([r, d], 1).astype(np.float64)
