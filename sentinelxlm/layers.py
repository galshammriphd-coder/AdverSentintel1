"""Graph / attention primitives implemented with plain PyTorch tensor ops (no PyG dependency).
Convention: edge (i -> j) means node i communicates with node j; node i aggregates from its
communication partners j (its neighbourhood N_i)."""
import math
import torch, torch.nn as nn, torch.nn.functional as F

def segment_softmax(e, index, n):
    H = e.size(1)
    idx = index.unsqueeze(1).expand(-1, H)
    mx = torch.full((n, H), -1e30, device=e.device, dtype=e.dtype).scatter_reduce(0, idx, e, reduce="amax", include_self=True)
    ex = torch.exp(e - mx[index])
    den = torch.zeros(n, H, device=e.device, dtype=e.dtype).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)

def with_self_loops(ei, n, w=None):
    loop = torch.arange(n, device=ei.device)
    ei2 = torch.cat([ei, torch.stack([loop, loop])], 1)
    if w is not None:
        w = torch.cat([w, torch.ones(n, device=w.device, dtype=w.dtype)])
    return ei2, w

class GATLayer(nn.Module):
    """Spatial attention, Eq. (1)-(2): alpha_ij = softmax_j LeakyReLU(a^T [W h_i || W h_j]); h_i = ELU(sum_j alpha_ij W h_j)."""
    def __init__(self, din, dout, heads=4):
        super().__init__()
        assert dout % heads == 0
        self.H, self.D = heads, dout // heads
        self.W = nn.Linear(din, dout, bias=False)
        self.a_i = nn.Parameter(torch.empty(heads, self.D)); self.a_j = nn.Parameter(torch.empty(heads, self.D))
        nn.init.xavier_uniform_(self.a_i); nn.init.xavier_uniform_(self.a_j)
    def forward(self, h, ei, ew=None):
        n = h.size(0)
        ei, ew = with_self_loops(ei, n, ew)
        i, j = ei[0], ei[1]
        wh = self.W(h).view(n, self.H, self.D)
        e = F.leaky_relu((wh * self.a_i).sum(-1)[i] + (wh * self.a_j).sum(-1)[j], 0.2)
        if ew is not None:
            e = e + torch.log(ew.clamp_min(1e-6)).unsqueeze(1)
        a = segment_softmax(e, i, n)
        out = torch.zeros(n, self.H, self.D, device=h.device, dtype=h.dtype).index_add_(0, i, a.unsqueeze(-1) * wh[j])
        return F.elu(out.reshape(n, -1))

def gcn_propagate(h, ei, ew=None):
    n = h.size(0)
    ei, ew = with_self_loops(ei, n, ew)
    i, j = ei[0], ei[1]
    w = torch.ones(ei.size(1), device=h.device, dtype=h.dtype) if ew is None else ew
    deg = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(0, i, w)
    norm = w / torch.sqrt(deg[i] * deg[j] + 1e-12)
    return torch.zeros_like(h).index_add_(0, i, norm.unsqueeze(1) * h[j])

class GCNLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__(); self.lin = nn.Linear(din, dout)
    def forward(self, h, ei, ew=None):
        return F.relu(self.lin(gcn_propagate(h, ei, ew)))

class SAGELayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__(); self.ls = nn.Linear(din, dout); self.ln = nn.Linear(din, dout, bias=False)
    def forward(self, h, ei):
        n = h.size(0)
        deg = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(0, ei[0], torch.ones(ei.size(1), device=h.device, dtype=h.dtype))
        nb = torch.zeros_like(h).index_add_(0, ei[0], h[ei[1]]) / deg.clamp_min(1).unsqueeze(1)
        return F.relu(self.ls(h) + self.ln(nb))

class TemporalAttention(nn.Module):
    """Temporal attention over the node's history, Eq. (3)-(4) (causal, masked by presence)."""
    def __init__(self, d, heads=4, positional=0):
        super().__init__()
        assert d % heads == 0
        self.H, self.dh = heads, d // heads
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.pos = nn.Parameter(torch.zeros(positional, d)) if positional else None
    def forward(self, h, present):
        T, N, d = h.shape
        if self.pos is not None:
            h = h + self.pos[:T].unsqueeze(1)
        q = self.q(h).view(T, N, self.H, self.dh); k = self.k(h).view(T, N, self.H, self.dh); v = self.v(h).view(T, N, self.H, self.dh)
        sc = torch.einsum("tnhd,snhd->nhts", q, k) / math.sqrt(self.dh)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device))
        key_ok = present.t().unsqueeze(1).unsqueeze(1) | torch.eye(T, dtype=torch.bool, device=h.device)
        sc = sc.masked_fill(~(causal & key_ok), -1e30)
        a = torch.softmax(sc, -1)
        return torch.einsum("nhts,snhd->tnhd", a, v).reshape(T, N, d)

class TTGATLayer(nn.Module):
    """One TT-GAT layer: spatial attention per snapshot + temporal attention + gated fusion, Eq. (5)-(6)."""
    def __init__(self, din, d, heads, t_heads, gating, use_spatial, use_temporal, dropout):
        super().__init__()
        self.use_spatial, self.use_temporal, self.gating = use_spatial, use_temporal, gating
        self.spatial = GATLayer(din, d, heads) if use_spatial else nn.Linear(din, d)
        if use_temporal:
            self.temporal = TemporalAttention(d, t_heads)
            if gating == "learned": self.gate = nn.Linear(2 * d, d)
            elif gating == "concat_mlp": self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.drop = nn.Dropout(dropout)
    def forward(self, h, edges, present):
        hs = torch.stack([self.spatial(h[t], edges[t]) if self.use_spatial else F.elu(self.spatial(h[t])) for t in range(h.size(0))])
        if not self.use_temporal:
            return self.drop(hs)
        ht = self.temporal(hs, present)
        if self.gating == "learned":
            g = torch.sigmoid(self.gate(torch.cat([hs, ht], -1)))
            out = g * hs + (1 - g) * ht
        elif self.gating == "average":
            out = 0.5 * (hs + ht)
        else:
            out = self.mlp(torch.cat([hs, ht], -1))
        return self.drop(out)
