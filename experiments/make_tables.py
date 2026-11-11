"""Aggregate per-run CSVs into the paper tables: mean, SD, t-based 95% CI, paired t-tests, log-log slopes."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sentinelxlm.stats import mean_sd_ci, paired_ttest, loglog_slope

def aggregate(df, keys, metrics):
    rows = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(keys, k)); r["n_runs"] = len(g)
        for m in metrics:
            mean, sd, lo, hi = mean_sd_ci(g[m].values)
            r.update({f"{m}_mean": mean, f"{m}_sd": sd, f"{m}_ci_low": lo, f"{m}_ci_high": hi})
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--kind", required=True,
        choices=["main", "variants", "window", "rules", "robustness", "scalability", "streaming", "threshold", "attack_types", "efficiency"])
    ap.add_argument("--out", required=True); ap.add_argument("--reference", default="edgetrust_6g"); a = ap.parse_args()
    df = pd.read_csv(a.inp); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); M = ["accuracy", "f1", "auc"]
    if a.kind == "main":
        out = aggregate(df, ["dataset", "model"], M); out.to_csv(a.out, index=False)
        tests = []
        for ds, g in df.groupby("dataset"):
            ref = g[g.model == a.reference].sort_values("seed")
            for m in g.model.unique():
                if m == a.reference: continue
                cur = g[g.model == m].sort_values("seed")
                for met in ("accuracy", "f1"):
                    t, p = paired_ttest(cur[met].values, ref[met].values); tests.append({"dataset": ds, "model": m, "reference": a.reference, "metric": met, "t": t, "p": p})
        pd.DataFrame(tests).to_csv(a.out.replace(".csv", "_paired_ttests.csv"), index=False)
    elif a.kind == "variants": aggregate(df, ["dataset", "variant"], M).to_csv(a.out, index=False)
    elif a.kind == "rules": aggregate(df, ["dataset", "condition"], M).to_csv(a.out, index=False)
    elif a.kind == "window": aggregate(df, ["dataset", "T"], M + ["latency_ms", "memory_mb"]).to_csv(a.out, index=False)
    elif a.kind == "robustness": aggregate(df, ["dataset", "perturbation", "ratio"], ["accuracy", "f1"]).to_csv(a.out, index=False)
    elif a.kind == "threshold": aggregate(df, ["dataset", "threshold"], M).to_csv(a.out, index=False)
    elif a.kind == "attack_types": aggregate(df, ["dataset", "category"], ["precision", "recall", "f1"]).to_csv(a.out, index=False)
    elif a.kind == "streaming":
        agg = aggregate(df, ["dataset", "epoch", "variant"], ["f1"]); agg.to_csv(a.out, index=False)
        piv = agg.pivot_table(index=["dataset", "epoch"], columns="variant", values="f1_mean"); piv["gap"] = piv["SENTINEL-XLM (SEM)"] - piv["No-SEM"]
        piv.to_csv(a.out.replace(".csv", "_gap.csv"))
    elif a.kind == "scalability":
        agg = aggregate(df, ["nodes"], ["latency_ms", "memory_mb"]); agg.to_csv(a.out, index=False)
        sl = {"latency_loglog_slope": loglog_slope(agg.nodes, agg.latency_ms_mean), "memory_loglog_slope": loglog_slope(agg.nodes, agg.memory_mb_mean)}
        pd.DataFrame([sl]).to_csv(a.out.replace(".csv", "_slopes.csv"), index=False); print(sl)
    elif a.kind == "efficiency": df.to_csv(a.out, index=False)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
