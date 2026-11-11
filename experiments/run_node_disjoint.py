"""Node-disjoint evaluation: every source IP belongs to exactly one of train / validation / test."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sentinelxlm.prepare import prepare
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["unsw", "cicids"]); ap.add_argument("--raw-dir", required=True); ap.add_argument("--work", default="data/processed_node_disjoint")
ap.add_argument("--models", nargs="*", default=["sentinel_xlm"]); ap.add_argument("--device", default="cpu"); ap.add_argument("--epochs", type=int)
ap.add_argument("--seeds", type=int, nargs="*"); ap.add_argument("--llm-embed", default="gpt2"); ap.add_argument("--out", required=True); a = ap.parse_args()
out = os.path.join(a.work, a.dataset); meta, stats = prepare(a.dataset, a.raw_dir, out, node_disjoint=True)
print("node-disjoint split:", stats["split"], flush=True); rows = []
for m in a.models:
    for s in (a.seeds or sentinel_cfg()["training"]["seeds"]):
        r, _, _ = train_eval(m, out, s, a); rows.append({"dataset": a.dataset, "split": "node_disjoint", **r, "test_flows": stats["split"]["test"]})
write_rows(a.out, rows)
