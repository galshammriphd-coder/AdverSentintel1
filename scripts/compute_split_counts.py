"""Derive / validate the dataset counts reported in Table 7 from the raw corpus counts (single source of truth: data/paper_dataset_stats.json)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentinelxlm.splits import split_sizes
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "paper_dataset_stats.json")
S = json.load(open(p))
for name, d in S.items():
    n = d["used"]["total"]; tr, va, te = split_sizes(n)
    assert d["used"]["benign"] + d["used"]["attack"] == n, name
    assert d["split"] == {"train": tr, "val": va, "test": te}, (name, d["split"], (tr, va, te))
    assert round(100 * d["used"]["benign"] / n, 2) == d["used"]["benign_ratio_percent"], name
    print(name, "OK:", d["used"], d["split"])
