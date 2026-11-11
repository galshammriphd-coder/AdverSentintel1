"""Grid search behind Table 8b: lr {1e-4,1e-3,1e-2} x d' {64,128,256} x layers {1,2,3} x batch {128,256,512} x dropout {0,0.1,0.3};
one seed per configuration, selection by peak validation AUC (early stopping patience 20)."""
import itertools
from common import *
from sentinelxlm.config import set_baseline_override, clear_baseline_overrides, baseline_cfg
GRID = {"lr": [1e-4, 1e-3, 1e-2], "d": [64, 128, 256], "layers": [1, 2, 3], "batch": [128, 256, 512], "dropout": [0.0, 0.1, 0.3]}
ap = base_parser(); ap.add_argument("--models", nargs="*", default=[m for m in ALL_MODELS if m != "sentinel_xlm"])
ap.add_argument("--grid-json", help="optional json restricting the grid, e.g. '{\"lr\":[0.001],\"d\":[64,128]}'"); a = ap.parse_args()
import json
grid = {**GRID, **(json.loads(a.grid_json) if a.grid_json else {})}
rows = []; cfg = sentinel_cfg(); seed = (a.seeds or [42])[0]
for ds, d in datasets(a):
    meta, win = load_data(d)
    for m in a.models:
        for vals in itertools.product(*grid.values()):
            ov = dict(zip(grid, vals)); set_baseline_override(m, ov)
            hook = llm_hook(a.llm_embed) if m == "llm_gnn" else None
            w = hook(win) if hook else win
            seqs = {k: SampleSeq(v, cfg["model"]["temporal_window"], meta["in_dim"]) for k, v in w.items()}
            set_seed(seed); model = build_model(m, meta["in_dim"], cfg)
            info = fit(model, m, seqs["train"], seqs["val"], cfg, seed, a.device, a.epochs)
            rows.append({"dataset": ds, "model": m, **ov, "best_val_auc": info["best_val"], "best_epoch": info["best_epoch"]}); print(rows[-1], flush=True)
clear_baseline_overrides(); write_rows(a.out, rows)
