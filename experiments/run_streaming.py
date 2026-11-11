"""Table 19: streaming slow-poisoning scenario with and without the SEM at fixed training epochs."""
from common import *
import copy
from sentinelxlm.streaming import poison_windows, stream_eval
ap = base_parser(); ap.add_argument("--eval-epochs", type=int, nargs="*", default=[20, 40, 60, 80, 100]); a = ap.parse_args(); rows = []
for ds, d in datasets(a):
    meta, win = load_data(d); cfg = sentinel_cfg(); T, F_ = cfg["model"]["temporal_window"], meta["in_dim"]
    xs = np.concatenate([w["x"][w["y"] == 0] for w in win["train"]]); mu_attack = xs.mean(0)
    for s in seeds_of(a, cfg):
        stream = poison_windows(win["test"], mu_attack, 0.10, seed=s)
        set_seed(s); model = build_model("sentinel_xlm", F_, cfg)
        seqs = {k: SampleSeq(v, T, F_) for k, v in win.items()}
        def cb(ep, m):
            if ep in a.eval_epochs:
                for variant, use in (("SENTINEL-XLM (SEM)", True), ("No-SEM", False)):
                    r = stream_eval(copy.deepcopy(m), stream, T, F_, use, a.device)
                    rows.append({"dataset": ds, "seed": s, "epoch": ep, "variant": variant, "f1": r["f1"], "accuracy": r["accuracy"]})
        fit(model, "sentinel_xlm", seqs["train"], seqs["val"], cfg, s, a.device, a.epochs or max(a.eval_epochs), patience=None, epoch_cb=cb)
write_rows(a.out, rows)
