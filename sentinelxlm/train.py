"""Training / evaluation entry points shared by SENTINEL-XLM and all baselines."""
import argparse, copy, json, os, pickle, random, time
import numpy as np, torch
from .config import sentinel_cfg, baseline_cfg
from .samples import SampleSeq, to_device
from .metrics import compute_metrics
from .sentinel import SentinelXLM, sentinel_loss
from .baselines import REGISTRY
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_data(data_dir):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    win = {s: pickle.load(open(os.path.join(data_dir, f"windows_{s}.pkl"), "rb")) for s in ("train", "val", "test")}
    return meta, win

def build_model(name, in_dim, cfg=None):
    if name == "sentinel_xlm":
        return SentinelXLM(in_dim, cfg or sentinel_cfg())
    return REGISTRY[name](in_dim, baseline_cfg(name))

def n_params(model): return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def evaluate(model, seq, device="cpu", perturb=None):
    model.eval(); T_, Y, C = [], [], []
    for i in range(len(seq)):
        s = seq[i]
        if perturb is not None: s = perturb(s)
        s = to_device(s, device); o = model(s); m = s["tmask"]
        T_.append(o["trust"][m].cpu().numpy()); Y.append(s["y"][m].cpu().numpy()); C.append(s["cat"][m].cpu().numpy())
    t, y, c = np.concatenate(T_), np.concatenate(Y), np.concatenate(C)
    return compute_metrics(t, y), (t, y, c)

def fit(model, name, train_seq, val_seq, cfg, seed, device="cpu", epochs=None, patience="cfg", epoch_cb=None, verbose=False):
    set_seed(seed)
    tcfg = cfg["training"]
    bcfg = None if name == "sentinel_xlm" else baseline_cfg(name)
    lr = tcfg["learning_rate"] if bcfg is None else bcfg["lr"]
    bs = tcfg["batch_size"] if bcfg is None else bcfg["batch"]
    epochs = epochs or tcfg["epochs"]
    patience = tcfg["early_stopping_patience"] if patience == "cfg" else patience
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    best, best_state, best_epoch, hist, step = -1.0, None, 0, [], 0
    use_evolve = name == "sentinel_xlm" and model.use_sem and model.use_nre
    for ep in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(train_seq)); opt.zero_grad(); k = 0; bb, by = [], []; tot = 0.0
        for idx in order:
            s = to_device(train_seq[idx], device); o = model(s); m = s["tmask"]
            if name == "sentinel_xlm":
                loss, _ = sentinel_loss(o, s, model, tcfg)
                if use_evolve and o["body"] is not None: bb.append(o["body"][m].detach()); by.append(s["y"][m])
            else:
                loss = F.binary_cross_entropy(o["trust"][m], s["y"][m].float())
            (loss / bs).backward(); tot += float(loss.detach()); k += 1
            if k % bs == 0 or k == len(order):
                opt.step(); opt.zero_grad(); step += 1
                if use_evolve and bb and step % model.cfg_s["evolve_every"] == 0:
                    model.nre.evolve(torch.cat(bb), torch.cat(by), model.cfg_s["mu"]); bb, by = [], []
        vm, _ = evaluate(model, val_seq, device)
        score = vm["auc"] if tcfg["early_stopping_metric"] == "auc" else vm["f1"]
        score = -1.0 if score != score else score
        hist.append({"epoch": ep, "loss": tot / max(1, k), **{f"val_{a}": b for a, b in vm.items()}})
        if verbose: print(f"{name} seed={seed} ep={ep} loss={tot/max(1,k):.4f} val_auc={vm['auc']:.4f} val_f1={vm['f1']:.2f}", flush=True)
        if epoch_cb is not None: epoch_cb(ep, model)
        if score > best: best, best_epoch, best_state = score, ep, copy.deepcopy(model.state_dict())
        elif patience is not None and ep - best_epoch >= patience: break
    if best_state is not None and patience is not None: model.load_state_dict(best_state)
    return {"history": hist, "best_epoch": best_epoch, "best_val": best}

def run_one(name, data_dir, seed, device="cpu", epochs=None, cfg=None, verbose=False, windows_hook=None):
    cfg = cfg or sentinel_cfg()
    meta, win = load_data(data_dir)
    if windows_hook is not None: win = windows_hook(win)
    T, F_ = cfg["model"]["temporal_window"], meta["in_dim"]
    seqs = {k: SampleSeq(v, T, F_) for k, v in win.items()}
    set_seed(seed)
    model = build_model(name, F_, cfg)
    info = fit(model, name, seqs["train"], seqs["val"], cfg, seed, device, epochs, verbose=verbose)
    test, _ = evaluate(model, seqs["test"], device)
    return {"model": name, "seed": seed, "best_epoch": info["best_epoch"], **test}, model, seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", default="sentinel_xlm"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int); ap.add_argument("--device", default="cpu"); ap.add_argument("--out")
    a = ap.parse_args()
    r, _, _ = run_one(a.model, a.data, a.seed, a.device, a.epochs, verbose=True)
    print(json.dumps(r))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(r, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
