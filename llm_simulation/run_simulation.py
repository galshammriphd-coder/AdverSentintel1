"""python llm_simulation/run_simulation.py --mode live --seeds 42 123 456 --out results/simulation/logs"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_simulation.agents import simulate
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="live", choices=["live", "offline"])
ap.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456]); ap.add_argument("--minutes", type=int, default=60); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for s in a.seeds:
    ev = simulate(s, a.minutes, a.mode)
    with open(os.path.join(a.out, f"events_seed{s}.jsonl"), "w") as f:
        for e in ev: f.write(json.dumps(e) + "\n")
    print(f"seed {s}: {len(ev)} events")
