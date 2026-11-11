import os, copy, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_yaml(name):
    path = name if os.path.isabs(name) else os.path.join(ROOT, "configs", name)
    with open(path) as f:
        return yaml.safe_load(f)
def sentinel_cfg(overrides=None):
    cfg = load_yaml("sentinel_xlm.yaml")
    for k, v in (overrides or {}).items():
        sec, key = k.split(".")
        cfg[sec][key] = v
    return copy.deepcopy(cfg)
_BASELINE_OVERRIDES = {}
def set_baseline_override(name, values):
    """Used by experiments/run_baseline_search.py to evaluate candidate hyperparameters."""
    _BASELINE_OVERRIDES[name] = dict(values)
def clear_baseline_overrides():
    _BASELINE_OVERRIDES.clear()
def baseline_cfg(name):
    cfg = dict(load_yaml("baselines.yaml")[name]); cfg.update(_BASELINE_OVERRIDES.get(name, {})); return cfg
