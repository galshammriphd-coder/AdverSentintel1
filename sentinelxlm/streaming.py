"""Streaming evaluation (Section 4.14): slow-poisoning stream with / without online SEM updates."""
import copy
import numpy as np, torch
from .samples import make_sample, to_device
from .metrics import compute_metrics

def poison_windows(windows, mu_attack, frac=0.10, seed=0):
    """A fraction of nodes gradually turns adversarial: at stream position r in [0,1] a poisoned node-window
    is shifted towards the mean attack feature vector with probability r and labelled untrustworthy."""
    rng = np.random.RandomState(seed)
    W = copy.deepcopy(windows)
    gids = np.unique(np.concatenate([w["gid"] for w in W]))
    chosen = set(rng.choice(gids, max(1, int(frac * len(gids))), replace=False).tolist())
    for i, w in enumerate(W):
        r = i / max(1, len(W) - 1)
        for k, g in enumerate(w["gid"]):
            if g in chosen and rng.rand() < r:
                w["x"][k] = (1 - r) * w["x"][k] + r * mu_attack
                w["y"][k] = 0
    return W

@torch.no_grad()
def stream_eval(model, windows, T, F, use_sem, device="cpu"):
    """Process windows in time order. With use_sem the SEM keeps a persistent Beta belief per node, updated from the
    observed outcome (label) of each scored window, and evolves rule weights every `evolve_every` windows (Eq. 17, 19, 20).
    Without it the trained trust scores stay frozen."""
    model.eval()
    q = model.cfg_s; M = 1 + max(int(w["gid"].max()) for w in windows)
    alpha = np.ones(M); beta = np.ones(M); cnt = np.zeros(M); ref = None; var = np.zeros(M)
    w0 = model.nre.w.clone() if model.use_nre else None
    P, Y, buf_b, buf_y = [], [], [], []
    for i in range(len(windows)):
        s = to_device(make_sample(windows[max(0, i - T + 1): i + 1], T, F), device)
        z = model.encode(s); o = model.score(z)
        tm = s["tmask"]; g = s["gid"][tm].cpu().numpy(); t_last = o["t"][-1][tm].cpu().numpy(); zl = z[-1][tm].cpu().numpy(); y = s["y"][tm].cpu().numpy()
        if ref is None: ref = np.zeros((M, zl.shape[1]))
        trust = t_last.copy()
        if use_sem:
            t0 = 0.5 * t_last + 0.5 * (alpha[g] / (alpha[g] + beta[g]))
            dist = np.linalg.norm(zl - ref[g], axis=1); rel = dist / (np.linalg.norm(ref[g], axis=1) + 1e-6)
            anomaly = (cnt[g] >= 2) & (dist > q["eta"] * np.sqrt(var[g]) + 1e-9)
            trust = np.where(anomaly, (1 - q["rho"]) * t0, t0 * np.exp(-q["kappa"] * np.minimum(rel, 5.0)))
            alpha[g] = q["gamma"] * alpha[g] + y; beta[g] = q["gamma"] * beta[g] + (1 - y)
            step = 1.0 / np.minimum(cnt[g] + 1, q["tau_ref"])[:, None]
            ref[g] = ref[g] + step * (zl - ref[g]); var[g] = (1 - 1.0 / q["tau_ref"]) * var[g] + (dist ** 2) / q["tau_ref"]; cnt[g] += 1
            if o["body"] is not None:
                buf_b.append(o["body"][-1][tm]); buf_y.append(s["y"][tm])
                if (i + 1) % q["evolve_every"] == 0:
                    model.nre.evolve(torch.cat(buf_b), torch.cat(buf_y), q["mu"]); buf_b, buf_y = [], []
        P.append(trust); Y.append(y)
    if w0 is not None: model.nre.w.copy_(w0)
    return compute_metrics(np.concatenate(P), np.concatenate(Y))
