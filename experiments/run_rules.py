"""Table 5: performance under full / 30% noisy / no symbolic rules."""
from common import *
COND = {"Full correct rules": {}, "30% noisy rules": {"model.rule_noise": 0.3}, "No rules (symbolic disabled)": {"model.use_nre": False}}
ap = base_parser(); a = ap.parse_args(); rows = []
for ds, d in datasets(a):
    for c, ov in COND.items():
        for s in seeds_of(a):
            r, _, _ = train_eval("sentinel_xlm", d, s, a, cfg=sentinel_cfg(ov)); rows.append({"dataset": ds, "condition": c, **r})
write_rows(a.out, rows)
