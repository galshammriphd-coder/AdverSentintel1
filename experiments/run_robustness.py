"""Table 13: random edge injection / deletion at test time (5%, 10%, 20% of edges, applied to every snapshot)."""
from common import *
def perturb_factory(kind, ratio, seed):
    rng = np.random.RandomState(seed)
    def f(s):
        n = s["x"].size(1); out = []
        for e in s["edges"]:
            E = e.size(1)
            if kind == "injection":
                k = int(round(ratio * E)); extra = torch.from_numpy(rng.randint(0, n, size=(2, k))).long()
                out.append(torch.cat([e, extra], 1))
            else:
                keep = torch.from_numpy(rng.rand(E) >= ratio); out.append(e[:, keep])
        s = dict(s); s["edges"] = out; return s
    return f
ap = base_parser(); a = ap.parse_args(); rows = []
for ds, d in datasets(a):
    for s in seeds_of(a):
        r, model, seqs = train_eval("sentinel_xlm", d, s, a)
        rows.append({"dataset": ds, "perturbation": "None", "ratio": 0.0, "seed": s, **{k: r[k] for k in ("accuracy", "f1", "auc")}})
        for kind in ("injection", "deletion"):
            for ratio in (0.05, 0.10, 0.20):
                m, _ = evaluate(model, seqs["test"], a.device, perturb_factory(kind, ratio, s))
                rows.append({"dataset": ds, "perturbation": kind, "ratio": ratio, "seed": s, "accuracy": m["accuracy"], "f1": m["f1"], "auc": m["auc"]})
write_rows(a.out, rows)
