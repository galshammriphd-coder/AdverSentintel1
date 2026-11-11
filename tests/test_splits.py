import json, os, numpy as np
from sentinelxlm.splits import split_sizes, stratified_split, node_disjoint_split
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_reported_split_counts():
    assert split_sizes(257673) == (180370, 25768, 51535)
    assert split_sizes(2821876) == (1975312, 282188, 564376)
    for n in (257673, 2821876): assert sum(split_sizes(n)) == n

def test_paper_stats_file_is_self_consistent():
    S = json.load(open(os.path.join(ROOT, "data", "paper_dataset_stats.json")))
    for d in S.values():
        n = d["used"]["total"]; assert d["used"]["benign"] + d["used"]["attack"] == n
        assert tuple(d["split"].values()) == split_sizes(n)
        assert round(100 * d["used"]["benign"] / n, 2) == d["used"]["benign_ratio_percent"]
    c = S["cicids"]; assert c["raw"]["benign"] - c["duplicates_removed_benign"] == c["used"]["benign"]
    assert c["raw"]["attack"] == c["used"]["attack"] and c["raw"]["benign"] + c["raw"]["attack"] == c["raw"]["total"]

def test_stratified_split_is_exact_partition():
    y = np.random.RandomState(0).rand(5000) < 0.3
    tr, va, te = stratified_split(y, 42); n = split_sizes(5000)
    assert (len(tr), len(va), len(te)) == n
    assert len(set(tr) | set(va) | set(te)) == 5000 and not (set(tr) & set(te)) and not (set(tr) & set(va)) and not (set(va) & set(te))
    assert abs(y[te].mean() - y.mean()) < 0.01

def test_node_disjoint_split_has_no_shared_ips():
    src = np.random.RandomState(1).randint(0, 60, 4000).astype(str)
    parts = node_disjoint_split(src, 42)
    sets = [set(src[p]) for p in parts]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(p) for p in parts) == 4000
