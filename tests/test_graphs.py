import numpy as np, pandas as pd
from sentinelxlm.graphs import build_windows
from sentinelxlm.data import feature_columns

def _df():
    rows = []
    def add(src, dst, t, lab): rows.append({"src": src, "dst": dst, "time": t, "label": lab, "attack_cat": "DoS" if lab else "Normal"})
    for i in range(10): add("A", "B", 1 + i, int(i < 3))        # A: 30% attacks -> trustworthy (not > 30%)
    for i in range(10): add("B", "A", 1 + i, int(i < 4))        # B: 40% attacks -> untrustworthy
    add("A", "C", 59.9, 0); add("A", "B", 60.1, 0); add("B", "A", 60.2, 0)
    return pd.DataFrame(rows)

def test_label_rule_windows_and_edges():
    df = _df(); ip2id = {"A": 0, "B": 1, "C": 2}
    X = np.random.RandomState(0).randn(len(df), 4).astype(np.float32); cues = np.ones((len(df), 2))
    W = build_windows(df, X, cues, ip2id, {"Normal": 0, "DoS": 1}, t0=0.0, window_s=60, thr=0.30)
    assert [w["win"] for w in W] == [0, 1]
    w0 = W[0]; y = dict(zip(w0["gid"], w0["y"]))
    assert y[0] == 1 and y[1] == 0                            # exactly 30% -> trustworthy, 40% -> untrustworthy
    assert w0["edge"].shape[1] == 2                           # A<->B only; C is not a source node of the window
    assert len(W[1]["gid"]) == 2 and W[1]["edge"].shape[1] == 2
    assert W[0]["cat"][list(w0["gid"]).index(1)] == 1         # majority attack category of node B

def test_attack_category_is_never_a_feature():
    df = pd.DataFrame({"src": ["a"], "dst": ["b"], "time": [0.0], "label": [0], "attack_cat": ["x"], "proto": ["tcp"], "dur": [1.0], "sport": [1], "Ltime": [1.0]})
    nums, cats = feature_columns(df, "unsw")
    assert "attack_cat" not in nums + cats and "label" not in nums + cats and nums == ["dur"] and cats == ["proto"]
