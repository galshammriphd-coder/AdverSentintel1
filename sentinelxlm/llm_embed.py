"""Text embeddings for the LLM-GNN baseline: each node-window feature vector is serialised as text and
embedded with GPT-2 small (mean-pooled last hidden state, 768-d). 'hash' is a deterministic projection
used ONLY for unit tests / smoke runs."""
import numpy as np

def serialise(row, k=8):
    return "node behaviour: " + ", ".join(f"f{i}={v:.2f}" for i, v in enumerate(row[:k]))

def attach_embeddings(windows, kind="gpt2", device="cpu", batch=64):
    if kind == "hash":
        rng = np.random.RandomState(0); Wp = rng.randn(windows[0]["x"].shape[1], 768).astype(np.float32) / 8
        for w in windows: w["emb"] = (w["x"] @ Wp).astype(np.float32)
        return windows
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained("gpt2").to(device).eval()
    for w in windows:
        texts = [serialise(r) for r in w["x"]]; out = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True).to(device)
            with torch.no_grad(): hs = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((hs * m).sum(1) / m.sum(1)).cpu().numpy())
        w["emb"] = np.concatenate(out).astype(np.float32)
    return windows
