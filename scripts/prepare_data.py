"""python scripts/prepare_data.py --dataset unsw --raw-dir /path/to/raw --out data/processed/unsw [--node-disjoint]
[--window-seconds 60] [--label-threshold 0.30] [--no-smote] [--no-dedup]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentinelxlm.prepare import prepare

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["unsw", "cicids"]); ap.add_argument("--raw-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--node-disjoint", action="store_true"); ap.add_argument("--window-seconds", type=int, default=60)
    ap.add_argument("--label-threshold", type=float, default=0.30); ap.add_argument("--no-smote", action="store_true"); ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    meta, stats = prepare(a.dataset, a.raw_dir, a.out, do_dedup=not a.no_dedup, window_seconds=a.window_seconds, label_threshold=a.label_threshold,
                          node_disjoint=a.node_disjoint, split_seed=a.seed, smote=not a.no_smote)
    print(json.dumps(stats, indent=1))
