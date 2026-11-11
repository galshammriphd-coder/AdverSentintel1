"""Table 16: per-attack-category precision / recall / F1 of SENTINEL-XLM (one-vs-normal at node-window level)."""
from common import *
from sentinelxlm.train import load_data
from sklearn.metrics import precision_recall_fscore_support
ap = base_parser(); a = ap.parse_args(); rows = []
for ds, d in datasets(a):
    meta, _ = load_data(d); names = meta["cat_names"]
    for s in seeds_of(a):
        r, model, seqs = train_eval("sentinel_xlm", d, s, a)
        _, (t, y, c) = evaluate(model, seqs["test"], a.device)
        pred_untrusted = (t < 0.5)
        for ci, cname in enumerate(names):
            if cname == "Normal":
                yt, yp = (y == 1), (~pred_untrusted)
            else:
                sel = (c == ci) | (c == 0)
                yt, yp = (c[sel] == ci) & (y[sel] == 0), pred_untrusted[sel]
            if yt.sum() == 0: continue
            p, rc, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
            rows.append({"dataset": ds, "seed": s, "category": cname, "precision": 100 * p, "recall": 100 * rc, "f1": 100 * f, "support": int(yt.sum())})
write_rows(a.out, rows)
