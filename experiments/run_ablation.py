"""Tables 10-12 and the design-variant table: module / design ablations of SENTINEL-XLM."""
from common import *
VARIANTS = {
 "Full SENTINEL-XLM": {},
 "w/o Temporal Attention": {"model.use_temporal": False},
 "w/o Spatial Attention": {"model.use_spatial": False},
 "w/o Neuro-symbolic Reasoning (NRE)": {"model.use_nre": False},
 "w/o Self-Evolution (SEM)": {"model.use_sem": False},
 "w/o Logic-consistency loss": {"model.use_logic_loss": False},
 "Gating replaced by simple average": {"model.gating": "average"},
 "Gating replaced by concatenation + MLP": {"model.gating": "concat_mlp"},
 "lambda fixed at 0.5": {"model.lambda_learnable": False, "model.lambda_init": 0.5},
 "lambda fixed at 0.9 (neural-dominant)": {"model.lambda_learnable": False, "model.lambda_init": 0.9},
 "lambda fixed at 0.1 (symbolic-dominant)": {"model.lambda_learnable": False, "model.lambda_init": 0.1},
 "Single-head temporal attention (H=1)": {"model.temporal_heads": 1},
}
ap = base_parser(); ap.add_argument("--variants", nargs="*", default=list(VARIANTS)); a = ap.parse_args()
rows = []
for ds, d in datasets(a):
    for v in a.variants:
        for s in seeds_of(a):
            r, _, _ = train_eval("sentinel_xlm", d, s, a, cfg=sentinel_cfg(VARIANTS[v])); rows.append({"dataset": ds, "variant": v, **r})
            print(ds, v, s, round(r["accuracy"], 2), round(r["f1"], 2), flush=True)
write_rows(a.out, rows)
