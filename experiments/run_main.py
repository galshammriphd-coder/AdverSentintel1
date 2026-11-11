"""Table 9 (and Tables 9b/20 inputs): SENTINEL-XLM vs the ten baselines, five seeds, both datasets."""
from common import *
ap = base_parser(); ap.add_argument("--models", nargs="*", default=ALL_MODELS); a = ap.parse_args()
rows = []
for ds, d in datasets(a):
    for m in a.models:
        for s in seeds_of(a):
            r, _, _ = train_eval(m, d, s, a); rows.append({"dataset": ds, **r}); print(ds, m, s, {k: round(v, 3) for k, v in r.items() if k in ("accuracy", "f1", "auc")}, flush=True)
write_rows(a.out, rows)
