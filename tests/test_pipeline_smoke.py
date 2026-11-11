import json, os, numpy as np
from sentinelxlm.splits import split_sizes
from sentinelxlm.train import run_one, load_data
from sentinelxlm.prepare import prepare
from sentinelxlm.streaming import poison_windows, stream_eval
from sentinelxlm.config import sentinel_cfg

def test_prepare_statistics_and_splits(smoke):
    st = json.load(open(os.path.join(smoke["unsw"], "stats.json")))
    assert st["duplicates_removed"] == 25 and st["total"] == 7000
    assert tuple(st["split"][k] for k in ("train", "val", "test")) == split_sizes(7000)
    meta, win = load_data(smoke["unsw"])
    assert meta["smote_synthetic_nodes"] > 0 and all(len(win[k]) > 0 for k in win)
    ys = np.concatenate([w["y"] for w in win["train"]]); assert abs((ys == 1).mean() - 0.5) < 0.05   # SMOTE balances the training nodes

def test_one_epoch_training_for_sentinel_and_a_baseline(smoke):
    for name in ("sentinel_xlm", "gcn_trust"):
        r, _, _ = run_one(name, smoke["unsw"], 42, epochs=1)
        assert all(np.isfinite(r[k]) for k in ("accuracy", "f1", "auc"))

def test_cicids_adapter_runs(smoke):
    r, _, _ = run_one("gat_security", smoke["cicids"], 42, epochs=1); assert np.isfinite(r["auc"])

def test_node_disjoint_pipeline(smoke, tmp_path):
    meta, stats = prepare("unsw", smoke["raw_unsw"], str(tmp_path / "nd"), node_disjoint=True)
    assert stats["split"]["mode"] == "node_disjoint" and sum(stats["split"][k] for k in ("train", "val", "test")) == stats["total"]

def test_streaming_with_and_without_sem(smoke):
    _, model, seqs = run_one("sentinel_xlm", smoke["unsw"], 42, epochs=1)
    meta, win = load_data(smoke["unsw"]); mu = np.concatenate([w["x"][w["y"] == 0] for w in win["train"]]).mean(0)
    stream = poison_windows(win["test"], mu, 0.10, seed=1); T, F = sentinel_cfg()["model"]["temporal_window"], meta["in_dim"]
    w0 = model.nre.w.clone()
    a = stream_eval(model, stream, T, F, True); b = stream_eval(model, stream, T, F, False)
    assert np.isfinite(a["f1"]) and np.isfinite(b["f1"]) and bool((model.nre.w == w0).all())   # rule weights restored after streaming
