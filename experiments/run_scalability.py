"""Table 14b: latency / memory of the full SENTINEL-XLM forward pass on synthetic sparse random graphs
(|E| ~ N log N per snapshot, T snapshots), 10 warm-up + 100 timed calls per seed."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from efficiency_utils import time_sample
from sentinelxlm.samples import random_sample
ap = argparse.ArgumentParser()
ap.add_argument("--nodes", type=int, nargs="*", default=[100, 500, 1000, 2000, 5000]); ap.add_argument("--in-dim", type=int, default=49)
ap.add_argument("--device", default="cpu"); ap.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456, 789, 1024])
ap.add_argument("--calls", type=int, default=100); ap.add_argument("--out", required=True); a = ap.parse_args()
cfg = sentinel_cfg(); T = cfg["model"]["temporal_window"]; rows = []
for N in a.nodes:
    for s in a.seeds:
        set_seed(s); model = build_model("sentinel_xlm", a.in_dim, cfg)
        lat, mem = time_sample(model, random_sample(N, T, a.in_dim, seed=s), a.device, 10, a.calls)
        rows.append({"nodes": N, "seed": s, "latency_ms": lat, "memory_mb": mem}); print(rows[-1], flush=True)
write_rows(a.out, rows)
