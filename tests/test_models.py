import torch, pytest
from sentinelxlm.config import sentinel_cfg
from sentinelxlm.train import load_data, build_model
from sentinelxlm.samples import SampleSeq
from sentinelxlm.baselines import REGISTRY
from sentinelxlm.llm_embed import attach_embeddings
from sentinelxlm.sentinel import NRE, sem_window_trust

@pytest.fixture(scope="module")
def sample(smoke):
    meta, win = load_data(smoke["unsw"]); win["test"] = attach_embeddings(win["test"], "hash")
    return SampleSeq(win["test"], 10, meta["in_dim"])[20], meta["in_dim"]

@pytest.mark.parametrize("name", list(REGISTRY) + ["sentinel_xlm"])
def test_every_model_forward_backward(sample, name):
    s, d = sample; m = build_model(name, d); t = m(s)["trust"]
    assert t.shape == (s["x"].size(1),) and bool(((t > 0) & (t < 1)).all())
    torch.nn.functional.binary_cross_entropy(t[s["tmask"]], s["y"][s["tmask"]].float()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())

def test_rule_base_and_lambda(sample):
    s, d = sample; m = build_model("sentinel_xlm", d)
    assert m.nre.R == 15 and abs(float(m.nre.lam.detach()) - 0.6) < 1e-6 and m.nre.pos.size(1) == 8

def test_ablation_flags_change_the_architecture(sample):
    s, d = sample
    assert not hasattr(build_model("sentinel_xlm", d, sentinel_cfg({"model.use_nre": False})), "nre")
    n_full = sum(p.numel() for p in build_model("sentinel_xlm", d).parameters())
    assert sum(p.numel() for p in build_model("sentinel_xlm", d, sentinel_cfg({"model.use_temporal": False})).parameters()) < n_full
    noisy = build_model("sentinel_xlm", d, sentinel_cfg({"model.rule_noise": 0.3}))
    assert int((noisy.nre.head != build_model("sentinel_xlm", d).nre.head).sum()) == 5       # 30% of 15 rules, rounded half up
    fixed = build_model("sentinel_xlm", d, sentinel_cfg({"model.lambda_learnable": False}))
    assert all(n != "nre.lam_logit" for n, _ in fixed.named_parameters())

def test_rule_evolution_moves_weights(sample):
    s, d = sample; m = build_model("sentinel_xlm", d); body = torch.rand(40, 15); y = torch.randint(0, 2, (40,))
    w0 = m.nre.w.clone(); m.nre.evolve(body, y, 0.5); assert not torch.equal(w0, m.nre.w) and float(m.nre.w.min()) >= 0.05

def test_sem_posterior_is_probability():
    T, n, dd = 10, 7, 16; t = torch.rand(T, n); z = torch.randn(T, n, dd); pres = torch.ones(T, n, dtype=torch.bool)
    out = sem_window_trust(t, z, pres, 0.95, 0.3, 0.5, 3.0, 10)
    assert out.shape == (n,) and bool(((out >= 0) & (out <= 1)).all())
