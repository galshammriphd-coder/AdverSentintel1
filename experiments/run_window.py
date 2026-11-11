"""Table 17: temporal window size T in {5, 10, 15, 20}: accuracy, F1, latency and memory."""
from common import *
from efficiency_utils import time_model
ap = base_parser(); ap.add_argument("--windows", type=int, nargs="*", default=[5, 10, 15, 20]); a = ap.parse_args(); rows = []
for ds, d in datasets(a):
    for T in a.windows:
        for s in seeds_of(a):
            cfg = sentinel_cfg({"model.temporal_window": T})
            r, model, seqs = train_eval("sentinel_xlm", d, s, a, cfg=cfg)
            lat, mem = time_model(model, seqs["test"], a.device)
            rows.append({"dataset": ds, "T": T, "seed": s, "accuracy": r["accuracy"], "f1": r["f1"], "auc": r["auc"], "latency_ms": lat, "memory_mb": mem})
write_rows(a.out, rows)
