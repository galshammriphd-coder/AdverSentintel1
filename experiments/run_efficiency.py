"""Table 14 (measured FP32 latency / memory / parameters) and Table 15 (analytical edge projection)."""
from common import *
from efficiency_utils import time_model
from sentinelxlm.config import load_yaml
ap = base_parser(); a = ap.parse_args(); rows = []
d = a.unsw or a.cicids
meta, win = load_data(d); T = sentinel_cfg()["model"]["temporal_window"]
hook = llm_hook(a.llm_embed)
for m in ALL_MODELS:
    w = hook(win)["test"] if m == "llm_gnn" else win["test"]
    seq = SampleSeq(w, T, meta["in_dim"]); set_seed(42); model = build_model(m, meta["in_dim"])
    lat, mem = time_model(model, seq, a.device)
    if m == "llm_gnn":                                   # include the text-embedding pass of the target window
        from sentinelxlm.llm_embed import attach_embeddings
        import copy, time
        wc = [copy.deepcopy({k: v for k, v in w[len(w) // 2].items() if k != "emb"})]
        t0 = time.perf_counter(); attach_embeddings(wc, a.llm_embed, a.device); lat += (time.perf_counter() - t0) * 1000
    rows.append({"model": m, "latency_ms": lat, "memory_mb": mem, "parameters_M": sum(p.numel() for p in model.parameters()) / 1e6, "kind": "measured"})
    print(rows[-1], flush=True)
base = next(r for r in rows if r["model"] == "sentinel_xlm")
for name, f in load_yaml("edge_projection.yaml").items():
    rows.append({"model": "sentinel_xlm", "platform": name, "latency_ms": base["latency_ms"] * f["latency_factor"],
                 "memory_mb": base["memory_mb"] * f["memory_factor"], "kind": "projected", "note": f["description"]})
write_rows(a.out, rows)
