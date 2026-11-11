"""Map logged events to 60-s windows (T snapshots, N=5 nodes), train SENTINEL-XLM and EdgeTrust-6G on the first 70% of the
time-ordered snapshots and evaluate on the remaining 30%. Labels are static (pre-configured adversaries -> 0)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from llm_simulation.features import window_features, DIM
from llm_simulation.agents import ADVERSARY, AGENTS
from sentinelxlm.config import sentinel_cfg
from sentinelxlm.samples import SampleSeq
from sentinelxlm.train import build_model, fit, evaluate, set_seed

def build_windows(events, window_s=60):
    T = int(max(e["t"] for e in events) // window_s) + 1
    wins = []
    for w in range(T):
        feats, gids, y = [], [], []
        for a in AGENTS:
            v, ok = window_features(events, a["id"], w * window_s, (w + 1) * window_s)
            feats.append(v); gids.append(a["id"]); y.append(0 if a["id"] in ADVERSARY else 1)
        dst = {(e["src"], e["dst"]) for e in events if w * window_s <= e["t"] < (w + 1) * window_s}
        edge = np.array([[s, d] for s, d in dst], dtype=np.int64).T if dst else np.zeros((2, 0), np.int64)
        wins.append({"win": w, "gid": np.array(gids), "x": np.stack(feats), "cue": np.zeros((5, 3), np.float32), "y": np.array(y), "cat": np.zeros(5, np.int64), "edge": edge})
    return wins

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--logs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456]); ap.add_argument("--epochs", type=int); a = ap.parse_args()
    rows = []
    for s in a.seeds:
        events = [json.loads(l) for l in open(os.path.join(a.logs, f"events_seed{s}.jsonl"))]
        wins = build_windows(events); k = int(0.7 * len(wins))
        cfg = sentinel_cfg(); T = cfg["model"]["temporal_window"]
        tr, te = wins[:k], wins[k - T + 1:]                  # test windows keep their T-1 history windows
        va = tr[-max(1, len(tr) // 7):]
        for name in ("sentinel_xlm", "edgetrust_6g"):
            set_seed(s); m = build_model(name, DIM, cfg)
            fit(m, name, SampleSeq(tr, T, DIM), SampleSeq(va, T, DIM), cfg, s, epochs=a.epochs)
            seq = SampleSeq(te, T, DIM); idx = range(T - 1, len(te))
            from sentinelxlm.metrics import compute_metrics
            import torch
            from sentinelxlm.samples import to_device
            P, Y = [], []
            m.eval()
            with torch.no_grad():
                for i in idx:
                    smp = to_device(seq[i], "cpu"); o = m(smp); P.append(o["trust"][smp["tmask"]].numpy()); Y.append(smp["y"][smp["tmask"]].numpy())
            r = compute_metrics(np.concatenate(P), np.concatenate(Y)); rows.append({"model": name, "seed": s, **r})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); pd.DataFrame(rows).to_csv(a.out, index=False); print(pd.DataFrame(rows))
