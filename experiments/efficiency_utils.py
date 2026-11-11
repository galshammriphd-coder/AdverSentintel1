import os, time
import numpy as np, torch
from sentinelxlm.samples import to_device

def _rss_mb():
    with open("/proc/self/statm") as f: return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20

@torch.no_grad()
def time_sample(model, sample, device="cpu", warmup=10, calls=100):
    model.eval().to(device); s = to_device(sample, device)
    if device != "cpu": torch.cuda.reset_peak_memory_stats()
    base = _rss_mb()
    for _ in range(warmup): model(s)
    ts = []
    for _ in range(calls):
        if device != "cpu": torch.cuda.synchronize()
        t0 = time.perf_counter(); model(s)
        if device != "cpu": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device != "cpu" else max(0.0, _rss_mb() - base)
    return float(np.mean(ts)), float(mem)

def time_model(model, seq, device="cpu", n_samples=20):
    """Mean latency (ms) and peak memory (MB) over the largest n_samples windows of a sequence."""
    idx = np.linspace(0, len(seq) - 1, min(n_samples, len(seq))).astype(int)
    out = [time_sample(model, seq[i], device, warmup=3, calls=10) for i in idx]
    return float(np.mean([o[0] for o in out])), float(np.max([o[1] for o in out]))
