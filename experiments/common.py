import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from sentinelxlm.config import sentinel_cfg
from sentinelxlm.train import run_one, load_data, build_model, fit, evaluate, set_seed
from sentinelxlm.samples import SampleSeq
from sentinelxlm.baselines import REGISTRY

ALL_MODELS = list(REGISTRY) + ["sentinel_xlm"]

def base_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsw"); ap.add_argument("--cicids"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int); ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--out", required=True); ap.add_argument("--llm-embed", default="gpt2", choices=["gpt2", "hash"])
    return ap

def datasets(a):
    return [(n, d) for n, d in (("unsw", a.unsw), ("cicids", a.cicids)) if d]

def seeds_of(a, cfg=None):
    return a.seeds or (cfg or sentinel_cfg())["training"]["seeds"]

def write_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", path, len(rows), "rows")

def llm_hook(kind):
    from sentinelxlm.llm_embed import attach_embeddings
    def hook(win):
        return {k: attach_embeddings(v, kind) for k, v in win.items()}
    return hook

def train_eval(name, data_dir, seed, a, cfg=None, hook=None):
    if name == "llm_gnn" and hook is None: hook = llm_hook(a.llm_embed)
    r, model, seqs = run_one(name, data_dir, seed, a.device, a.epochs, cfg, windows_hook=hook)
    return r, model, seqs
