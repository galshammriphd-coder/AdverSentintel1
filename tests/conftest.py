import os, sys, importlib.util
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))

def _load_smoke():
    spec = importlib.util.spec_from_file_location("make_smoke_data", os.path.join(ROOT, "scripts", "make_smoke_data.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

@pytest.fixture(scope="session")
def smoke(tmp_path_factory):
    from sentinelxlm.prepare import prepare
    base = tmp_path_factory.mktemp("smoke"); mk = _load_smoke()
    mk.make_unsw(str(base / "raw_unsw")); mk.make_cicids(str(base / "raw_cicids"))
    out = {"raw_unsw": str(base / "raw_unsw"), "raw_cicids": str(base / "raw_cicids")}
    out["unsw"] = str(base / "unsw"); out["cicids"] = str(base / "cicids")
    prepare("unsw", out["raw_unsw"], out["unsw"]); prepare("cicids", out["raw_cicids"], out["cicids"])
    return out
