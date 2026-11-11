"""Twelve agent-communication behaviour features per (agent, 60-s window) -> 49-d vector (Appendix B, Table B1).
The remaining dimensions are zero-padded."""
import numpy as np
FEATURES = ["message_rate", "mean_response_latency_ms", "output_token_entropy", "context_utilization", "injection_prefix_present",
            "corrupt_response_ratio", "agent_error_rate", "ack_flag", "syn_flag", "fin_flag", "payload_size_mean", "connection_entropy"]
DIM = 49

def window_features(events, agent, w0, w1):
    """events: list of dicts with t, src, dst, latency_ms, tokens, entropy, ctx, injected, corrupt, error, flag, size."""
    ev = [e for e in events if e["src"] == agent and w0 <= e["t"] < w1]
    v = np.zeros(DIM, np.float32)
    if not ev: return v, False
    n = len(ev)
    dst = np.array([e["dst"] for e in ev]); _, c = np.unique(dst, return_counts=True); p = c / c.sum()
    f = [n, np.mean([e["latency_ms"] for e in ev]), np.mean([e["entropy"] for e in ev]), np.mean([e["ctx"] for e in ev]),
         float(any(e["injected"] for e in ev)), np.mean([e["corrupt"] for e in ev]), np.mean([e["error"] for e in ev]),
         np.mean([e["flag"] == "ack" for e in ev]), np.mean([e["flag"] == "syn" for e in ev]), np.mean([e["flag"] == "fin" for e in ev]),
         np.mean([e["size"] for e in ev]), float(-(p * np.log(p)).sum())]
    v[:12] = f
    return v, True
