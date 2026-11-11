"""SENTINEL-XLM: TT-GAT + Neuro-symbolic Reasoning Engine (NRE) + Self-Evolution Module (SEM)."""
import math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from .config import ROOT
from .layers import TTGATLayer

class NRE(nn.Module):
    """Differentiable first-order rules over learnable predicates (Eq. 7-14).
    body_r = prod_k p_k^pos * (1-p_k)^neg (product t-norm); the rule-implied trust is body_r for a
    'Trustworthy' head and 1-body_r for a 'not Trustworthy' head; the symbolic trust is the
    confidence-weighted average over rules; T = lambda*T_neural + (1-lambda)*T_symbolic (Eq. 14).
    Rule satisfaction for the logic loss uses the Lukasiewicz implication min(1, 1-body+head) (Eq. 11/24)."""
    def __init__(self, d, m, rules_file, lam_init, lam_learnable, rule_noise=0.0):
        super().__init__()
        kb = yaml.safe_load(open(rules_file if os.path.isabs(rules_file) else os.path.join(ROOT, rules_file)))
        R = len(kb["rules"])
        pos = torch.zeros(R, m); neg = torch.zeros(R, m); head = torch.zeros(R)
        for r, rule in enumerate(kb["rules"]):
            for k in rule["pos"]: pos[r, k] = 1
            for k in rule["neg"]: neg[r, k] = 1
            head[r] = rule["head"]
        if rule_noise > 0:                                   # corrupt a fixed fraction of rules (Table 5)
            rs = np.random.RandomState(0)
            flip = rs.choice(R, int(np.floor(rule_noise * R + 0.5)), replace=False)   # 30% of 15 rules -> 5
            head[torch.as_tensor(flip)] *= -1
        self.R = R
        self.register_buffer("pos", pos); self.register_buffer("neg", neg); self.register_buffer("head", head)
        self.register_buffer("w", torch.ones(R))             # rule confidences, evolved by the SEM (Eq. 20)
        self.pred = nn.Linear(d, m)                          # predicate grounding, Eq. (7)
        self.neural = nn.Linear(d, 1)                        # neural trust head, Eq. (12)
        logit = math.log(lam_init / (1 - lam_init))
        if lam_learnable: self.lam_logit = nn.Parameter(torch.tensor(logit))
        else: self.register_buffer("lam_logit", torch.tensor(logit))
    @property
    def lam(self): return torch.sigmoid(self.lam_logit)
    def body(self, p):
        lp, lq = torch.log(p.clamp(1e-6, 1)), torch.log((1 - p).clamp(1e-6, 1))
        return torch.exp(lp @ self.pos.t() + lq @ self.neg.t())
    def forward(self, z):
        p = torch.sigmoid(self.pred(z))
        b = self.body(p)
        t_nn = torch.sigmoid(self.neural(z)).squeeze(-1)
        s_r = torch.where(self.head > 0, b, 1 - b)
        w = self.w.clamp_min(1e-3)
        t_sym = (s_r * w).sum(-1) / w.sum()
        t = self.lam * t_nn + (1 - self.lam) * t_sym
        h_r = torch.where(self.head > 0, t_nn.unsqueeze(-1), 1 - t_nn.unsqueeze(-1))
        sat = torch.clamp(1 - b + h_r, max=1.0)
        return {"t": t, "t_nn": t_nn, "t_sym": t_sym, "sat": sat, "body": b}
    @torch.no_grad()
    def evolve(self, body, y, mu):
        """Eq. (20): w_r <- w_r + mu * (Acc(r, D) - mean(w)). Accuracy is measured on the nodes whose body fires."""
        fires = body > 0.5
        pred = (self.head > 0).long().unsqueeze(0).expand_as(body)       # rule predicts trustworthy (1) / not (0)
        correct = (pred == y.long().unsqueeze(1)) & fires
        n = fires.sum(0).float()
        acc = torch.where(n > 0, correct.sum(0).float() / n.clamp_min(1), self.w)
        self.w.copy_((self.w + mu * (acc - self.w.mean())).clamp(0.05, 1.0))

def sem_window_trust(t, z, pres, gamma, rho, kappa, eta, tau_ref):
    """SEM trust over the T-step window: Beta posterior (Eq. 16-17) followed by anomaly-triggered
    adaptation (Eq. 19). t: [T,n] per-step trust used as soft outcome, z: [T,n,d], pres: [T,n]."""
    T, n = t.shape
    a = torch.ones(n, device=t.device); b = torch.ones(n, device=t.device)
    for k in range(T):
        pk = pres[k].float()
        a = gamma * a + pk * t[k]
        b = gamma * b + pk * (1 - t[k])
    m = a / (a + b)
    if T < 3:
        return m
    prev = z[:-1]; pp = pres[:-1].float().unsqueeze(-1)
    cnt = pp.sum(0)
    mean = (prev * pp).sum(0) / cnt.clamp_min(1)
    sigma = ((((prev - mean) ** 2) * pp).sum(0) / cnt.clamp_min(1)).sum(-1).sqrt()
    ref = prev[-tau_ref:]; rp = pres[:-1][-tau_ref:].float().unsqueeze(-1)
    ostar = (ref * rp).sum(0) / rp.sum(0).clamp_min(1)                 # reference o* (Section 3.5.2)
    dist = (z[-1] - ostar).norm(dim=-1) / (ostar.norm(dim=-1) + 1e-6)
    anomaly = (cnt.squeeze(-1) >= 2) & ((z[-1] - mean).norm(dim=-1) > eta * sigma)
    return torch.where(anomaly, (1 - rho) * m, m * torch.exp(-kappa * dist.clamp(max=5.0)))

class SentinelXLM(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        m, s = cfg["model"], cfg["sem"]
        d = m["embedding_dim"]
        self.cfg_m, self.cfg_s = m, s
        dims = [in_dim] + [d] * m["num_layers"]
        self.layers = nn.ModuleList([TTGATLayer(dims[i], d, m["num_heads"], m["temporal_heads"], m["gating"],
                                                m["use_spatial"], m["use_temporal"], m["dropout"]) for i in range(m["num_layers"])])
        self.use_nre, self.use_sem = m["use_nre"], m["use_sem"]
        if self.use_nre:
            self.nre = NRE(d, m["num_predicates"], m["rules_file"], m["lambda_init"], m["lambda_learnable"], m["rule_noise"])
        else:
            self.head = nn.Linear(d, 1)
    def encode(self, s):
        h = s["x"]
        for layer in self.layers:
            h = layer(h, s["edges"], s["present"])
        return h
    def score(self, z):
        if self.use_nre:
            return self.nre(z)
        t = torch.sigmoid(self.head(z)).squeeze(-1)
        return {"t": t, "t_nn": t, "t_sym": None, "sat": None, "body": None}
    def forward(self, s):
        z = self.encode(s)
        o = self.score(z)
        if self.use_sem:
            q = self.cfg_s
            trust = sem_window_trust(o["t"], z, s["present"], q["gamma"], q["rho"], q["kappa"], q["eta"], q["tau_ref"])
        else:
            trust = o["t"][-1]
        return {"trust": trust.clamp(1e-6, 1 - 1e-6), "t_nn": o["t_nn"][-1], "sat": None if o["sat"] is None else o["sat"][-1],
                "body": None if o["body"] is None else o["body"][-1], "z": z}

def sentinel_loss(out, s, model, tcfg, edges_for_sym=None):
    """Eq. (22): L = BCE + alpha_L * L_logic + beta_R * (||theta||^2 + tau_s ||A - A^T||_F^2)."""
    m = s["tmask"]
    y = s["y"][m].float()
    loss = F.binary_cross_entropy(out["trust"][m], y)
    logs = {"bce": float(loss.detach())}
    if out["sat"] is not None and model.cfg_m["use_logic_loss"]:
        logic = torch.clamp(1 - out["sat"][m], min=0).mean()
        loss = loss + tcfg["logic_loss_weight"] * logic; logs["logic"] = float(logic.detach())
    params = [p for p in model.parameters() if p.requires_grad]
    reg = sum((p ** 2).sum() for p in params)
    if tcfg.get("regularization_normalize", True):          # ||theta||^2 averaged over the number of parameters
        reg = reg / sum(p.numel() for p in params)
    sym = 0.0
    ei = s["edges"][-1]
    if ei.numel():
        n = s["x"].size(1)
        k1 = ei[0] * n + ei[1]; k2 = ei[1] * n + ei[0]
        both = torch.isin(k1, k2).sum().item()
        sym = float(2 * (ei.size(1) - both))                  # ||A - A^T||_F^2 for a binary adjacency
    loss = loss + tcfg["regularization_weight"] * (reg + tcfg["symmetry_weight"] * sym)
    return loss, logs
