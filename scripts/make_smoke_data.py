"""Generate small synthetic raw-format files (UNSW-NB15-like and CICIDS2017-like) for tests and the smoke run.
The data are random and carry no scientific meaning; they only exercise the pipeline."""
import argparse, os
import numpy as np, pandas as pd

def _flows(n, n_ips, seed, t_span=3000):
    rng = np.random.RandomState(seed)
    ips = [f"10.0.0.{i}" for i in range(1, n_ips + 1)]
    kind = np.array(["benign"] * n_ips, dtype=object); kind[: n_ips // 4] = "attacker"; kind[n_ips // 4: n_ips // 3] = "mixed"
    src = rng.randint(0, n_ips, n)
    p_att = np.where(kind[src] == "attacker", 0.85, np.where(kind[src] == "mixed", 0.30, 0.02))
    atk = (rng.rand(n) < p_att).astype(int)
    dst = rng.randint(0, n_ips, n)
    t = np.sort(rng.rand(n) * t_span)
    X = rng.randn(n, 6) + atk[:, None] * np.array([1.5, -1.0, 1.2, 0.8, -0.6, 1.0])
    return ips, src, dst, atk, t, X, rng

def make_unsw(path, n=7000, seed=1):
    ips, src, dst, atk, t, X, rng = _flows(n, 24, seed)
    cats = np.array(["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance"])
    df = pd.DataFrame({"srcip": [ips[i] for i in src], "sport": rng.randint(1024, 65535, n), "dstip": [ips[i] for i in dst],
                       "dsport": rng.choice([80, 443, 53, 22], n), "proto": rng.choice(["tcp", "udp", "icmp"], n),
                       "state": rng.choice(["FIN", "CON", "INT"], n), "dur": np.abs(X[:, 0]) + rng.rand(n),
                       "sbytes": np.abs(X[:, 1] * 100).astype(int), "dbytes": np.abs(X[:, 2] * 100).astype(int),
                       "sttl": rng.choice([31, 62, 254], n), "Spkts": np.abs(X[:, 3] * 5).astype(int) + 1,
                       "Dpkts": np.abs(X[:, 4] * 5).astype(int) + 1, "service": rng.choice(["-", "http", "dns"], n),
                       "Sload": np.abs(X[:, 5]) * 1000, "smeansz": 100 + 30 * X[:, 0], "ct_srv_src": rng.randint(1, 30, n),
                       "Stime": 1421927000 + t, "Ltime": 1421927001 + t,
                       "attack_cat": np.where(atk == 1, rng.choice(cats, n), ""), "Label": atk})
    df = pd.concat([df, df.iloc[:25]], ignore_index=True)           # exact duplicates for the de-duplication step
    os.makedirs(path, exist_ok=True); df.to_csv(os.path.join(path, "UNSW-NB15_1.csv"), index=False)

def make_cicids(path, n=8000, seed=2):
    ips, src, dst, atk, t, X, rng = _flows(n, 30, seed)
    ts = pd.to_datetime("2017-07-07 08:00:00") + pd.to_timedelta(t, unit="s")
    lab = np.where(atk == 1, rng.choice(["DoS Hulk", "PortScan", "DDoS"], n), "BENIGN")
    df = pd.DataFrame({"Flow ID": [f"f{i}" for i in range(n)], "Source IP": [ips[i] for i in src], "Source Port": rng.randint(1024, 65535, n),
                       "Destination IP": [ips[i] for i in dst], "Destination Port": rng.choice([80, 443, 53], n), "Protocol": 6,
                       "Timestamp": ts.strftime("%m/%d/%Y %H:%M:%S"), "Flow Duration": np.abs(X[:, 0] * 1e5) + 1,
                       "Total Fwd Packets": np.abs(X[:, 1] * 5).astype(int) + 1, "Flow Packets/s": np.abs(X[:, 2] * 100) + 1,
                       "Packet Length Std": np.abs(X[:, 3]) * 10, "Fwd Packet Length Max": np.abs(X[:, 4]) * 100 + 40,
                       "ACK Flag Count": rng.randint(0, 2, n), "Average Packet Size": 100 + 20 * X[:, 5], "Label": lab})
    df = pd.concat([df, df.iloc[:30]], ignore_index=True)
    os.makedirs(path, exist_ok=True); df.to_csv(os.path.join(path, "smoke_cicids.csv"), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/smoke_raw"); a = ap.parse_args()
    make_unsw(os.path.join(a.out, "unsw")); make_cicids(os.path.join(a.out, "cicids"))
    print("smoke raw data written to", a.out)
