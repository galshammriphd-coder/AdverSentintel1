"""Table A1: sensitivity to the node trust-label threshold (needs the raw datasets)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sentinelxlm.prepare import prepare
ap = argparse.ArgumentParser()
ap.add_argument("--unsw-raw"); ap.add_argument("--cicids-raw"); ap.add_argument("--work", default="data/processed_threshold")
ap.add_argument("--thresholds", type=float, nargs="*", default=[0.10, 0.20, 0.30, 0.40, 0.50]); ap.add_argument("--device", default="cpu")
ap.add_argument("--epochs", type=int); ap.add_argument("--seeds", type=int, nargs="*"); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = []
for ds, raw in (("unsw", a.unsw_raw), ("cicids", a.cicids_raw)):
    if not raw: continue
    for th in a.thresholds:
        out = os.path.join(a.work, f"{ds}_th{int(th * 100)}"); prepare(ds, raw, out, label_threshold=th)
        for s in (a.seeds or sentinel_cfg()["training"]["seeds"]):
            r, _, _ = run_one("sentinel_xlm", out, s, a.device, a.epochs); rows.append({"dataset": ds, "threshold": th, **r})
write_rows(a.out, rows)
