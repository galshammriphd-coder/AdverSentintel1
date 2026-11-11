"""The ten baselines (Section 4.1.3), implemented in plain PyTorch with a shared protocol:
static GNNs run on every snapshot of the window and their node embeddings are averaged over the
snapshots in which the node is present; temporal models use their own temporal mechanism."""
import torch, torch.nn as nn, torch.nn.functional as F
from .layers import GATLayer, GCNLayer, SAGELayer, TemporalAttention, gcn_propagate
from .sentinel import NRE

def _masked_mean(H, present):
    p = present.float().unsqueeze(-1)
    return (H * p).sum(0) / p.sum(0).clamp_min(1)

class _Snapshot(nn.Module):
    """Per-snapshot encoder + masked temporal mean + linear trust head."""
    def __init__(self, in_dim, c):
        super().__init__()
        self.in_dim, self.c = in_dim, c
        self.drop = nn.Dropout(c["dropout"])
    def inputs(self, s, t): return s["x"][t]
    def encode(self, h, ei): raise NotImplementedError
    def embed(self, s):
        T = s["x"].size(0)
        H = torch.stack([self.encode(self.inputs(s, t), s["edges"][t]) for t in range(T)])
        return _masked_mean(H, s["present"])
    def forward(self, s):
        h = self.embed(s)
        return {"trust": torch.sigmoid(self.head(h)).squeeze(-1).clamp(1e-6, 1 - 1e-6)}

class GCNTrust(_Snapshot):
    def __init__(self, in_dim, c):
        super().__init__(in_dim, c)
        dims = [in_dim] + [c["d"]] * c["layers"]
        self.l = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(c["layers"])]); self.head = nn.Linear(c["d"], 1)
    def encode(self, h, ei):
        for l in self.l: h = self.drop(l(h, ei))
        return h

class GATSecurity(_Snapshot):
    def __init__(self, in_dim, c):
        super().__init__(in_dim, c)
        dims = [in_dim] + [c["d"]] * c["layers"]
        self.l = nn.ModuleList([GATLayer(dims[i], dims[i + 1], c["heads"]) for i in range(c["layers"])]); self.head = nn.Linear(c["d"], 1)
    def encode(self, h, ei):
        for l in self.l: h = self.drop(l(h, ei))
        return h

class GraphSAGEEdge(_Snapshot):
    def __init__(self, in_dim, c):
        super().__init__(in_dim, c)
        dims = [in_dim] + [c["d"]] * c["layers"]
        self.l = nn.ModuleList([SAGELayer(dims[i], dims[i + 1]) for i in range(c["layers"])]); self.head = nn.Linear(c["d"], 1)
    def encode(self, h, ei):
        for l in self.l: h = self.drop(l(h, ei))
        return h

class TrustGNN(GCNTrust):
    """Two-layer GCN backbone with behavioural-cue features (rate deviation, duration variance, connection entropy) appended."""
    def __init__(self, in_dim, c):
        super().__init__(in_dim + 3, c)
    def inputs(self, s, t): return torch.cat([s["x"][t], s["cue"][t]], -1)

class NeSyTrust(GCNTrust):
    """GCN backbone + differentiable symbolic rule layer (same predicate vocabulary as the NRE; fixed rule confidences, no SEM, no logic loss)."""
    def __init__(self, in_dim, c, rules_file="rules/trust_rules.yaml", m=8):
        super().__init__(in_dim, c)
        self.nre = NRE(c["d"], m, rules_file, 0.6, True)
    def forward(self, s):
        o = self.nre(self.embed(s))
        return {"trust": o["t"].clamp(1e-6, 1 - 1e-6)}

class LLMGNN(GCNTrust):
    """Node text embeddings (GPT-2 small, 768-d; see llm_embed.py) projected to d' feed a two-layer GCN."""
    def __init__(self, in_dim, c, emb_dim=768):
        super().__init__(c["d"], c)
        self.proj = nn.Linear(emb_dim, c["d"])
    def inputs(self, s, t):
        if "emb" not in s: raise KeyError("LLM-GNN needs text embeddings: run attach_embeddings() first")
        return F.relu(self.proj(s["emb"][t]))

class CausalNet(_Snapshot):
    """Learned DAG mask over node indices applied to the observed graph (edge i->j kept with weight sigmoid(u_i.v_j) only if i<j),
    two weighted message-passing layers followed by an MLP head (classical replacement of the quantum components)."""
    def __init__(self, in_dim, c, max_nodes=8192, rank=16):
        super().__init__(in_dim, c)
        self.U = nn.Embedding(max_nodes, rank); self.V = nn.Embedding(max_nodes, rank); self.max_nodes = max_nodes
        dims = [in_dim] + [c["d"]] * c["layers"]
        self.l = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(c["layers"])])
        self.head = nn.Sequential(nn.Linear(c["d"], c["d"]), nn.ReLU(), nn.Linear(c["d"], 1))
    def encode(self, h, ei):
        i, j = ei[0], ei[1]
        w = torch.sigmoid((self.U(i % self.max_nodes) * self.V(j % self.max_nodes)).sum(-1)) * (i < j).float()
        for l in self.l: h = self.drop(l(h, ei, w))
        return h

class EvolveGCNEdge(nn.Module):
    """EvolveGCN-O: the GCN weight matrices are evolved over the snapshots by GRUs."""
    def __init__(self, in_dim, c):
        super().__init__()
        dims = [in_dim] + [c["d"]] * c["layers"]
        self.W0 = nn.ParameterList([nn.Parameter(torch.randn(dims[i], dims[i + 1]) * 0.1) for i in range(c["layers"])])
        self.gru = nn.ModuleList([nn.GRUCell(dims[i], dims[i]) for i in range(c["layers"])])
        self.head = nn.Linear(c["d"], 1); self.drop = nn.Dropout(c["dropout"])
    def forward(self, s):
        Ws = list(self.W0); T = s["x"].size(0); h = None
        for t in range(T):
            h = s["x"][t]
            for l in range(len(Ws)):
                Ws[l] = self.gru[l](Ws[l].t(), Ws[l].t()).t()
                h = self.drop(F.relu(gcn_propagate(h, s["edges"][t]) @ Ws[l]))
        return {"trust": torch.sigmoid(self.head(h)).squeeze(-1).clamp(1e-6, 1 - 1e-6)}

class DySATTrust(nn.Module):
    """Structural self-attention per snapshot followed by temporal self-attention with positional embeddings."""
    def __init__(self, in_dim, c):
        super().__init__()
        dims = [in_dim] + [c["d"]] * c["layers"]
        self.l = nn.ModuleList([GATLayer(dims[i], dims[i + 1], c["heads"]) for i in range(c["layers"])])
        self.temp = TemporalAttention(c["d"], c["heads"], positional=64)
        self.head = nn.Linear(c["d"], 1); self.drop = nn.Dropout(c["dropout"])
    def forward(self, s):
        H = []
        for t in range(s["x"].size(0)):
            h = s["x"][t]
            for l in self.l: h = self.drop(l(h, s["edges"][t]))
            H.append(h)
        H = self.temp(torch.stack(H), s["present"])
        return {"trust": torch.sigmoid(self.head(H[-1])).squeeze(-1).clamp(1e-6, 1 - 1e-6)}

class EdgeTrust6G(nn.Module):
    """Temporal GAT whose edge weights evolve through an LSTM over node states (no published code; see DEVIATIONS.md)."""
    def __init__(self, in_dim, c):
        super().__init__()
        dims = [in_dim] + [c["d"]] * c["layers"]
        self.l = nn.ModuleList([GATLayer(dims[i], dims[i + 1], c["heads"]) for i in range(c["layers"])])
        self.lstm = nn.LSTMCell(c["d"], c["d"]); self.ew = nn.Sequential(nn.Linear(2 * c["d"], c["d"]), nn.ReLU(), nn.Linear(c["d"], 1))
        self.head = nn.Linear(c["d"], 1); self.drop = nn.Dropout(c["dropout"]); self.d = c["d"]
    def forward(self, s):
        n = s["x"].size(1); hx = torch.zeros(n, self.d, device=s["x"].device); cx = torch.zeros_like(hx)
        for t in range(s["x"].size(0)):
            ei = s["edges"][t]
            ew = None if (t == 0 or ei.numel() == 0) else torch.sigmoid(self.ew(torch.cat([hx[ei[0]], hx[ei[1]]], -1))).squeeze(-1)
            h = s["x"][t]
            for l in self.l: h = self.drop(l(h, ei, ew))
            hx, cx = self.lstm(h, (hx, cx))
        return {"trust": torch.sigmoid(self.head(hx)).squeeze(-1).clamp(1e-6, 1 - 1e-6)}

REGISTRY = {"gcn_trust": GCNTrust, "gat_security": GATSecurity, "graphsage_edge": GraphSAGEEdge, "evolvegcn_edge": EvolveGCNEdge,
            "dysat_trust": DySATTrust, "trustgnn": TrustGNN, "nesy_trust": NeSyTrust, "llm_gnn": LLMGNN, "causalnet": CausalNet,
            "edgetrust_6g": EdgeTrust6G}
DISPLAY = {"gcn_trust": "GCN-Trust", "gat_security": "GAT-Security", "graphsage_edge": "GraphSAGE-Edge", "evolvegcn_edge": "EvolveGCN-Edge",
           "dysat_trust": "DySAT-Trust", "trustgnn": "TrustGNN", "nesy_trust": "NeSy-Trust", "llm_gnn": "LLM-GNN",
           "causalnet": "CausalNet", "edgetrust_6g": "EdgeTrust-6G", "sentinel_xlm": "SENTINEL-XLM"}
