"""Mean / SD / t-based confidence intervals, paired tests, log-log slopes, Krippendorff alpha."""
import numpy as np
from scipy import stats as st

def mean_sd_ci(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, 0.0, m, m
    sd = float(x.std(ddof=1))
    h = float(st.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n))
    return m, sd, m - h, m + h

def paired_ttest(a, b):
    t, p = st.ttest_rel(np.asarray(a, float), np.asarray(b, float))
    return float(t), float(p)

def loglog_slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])

def krippendorff_alpha_ordinal(ratings):
    """ratings: array [raters, units] with np.nan for missing. Ordinal metric."""
    R = np.asarray(ratings, float)
    vals = np.unique(R[~np.isnan(R)])
    idx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    o = np.zeros((k, k))
    for u in range(R.shape[1]):
        col = R[:, u]
        col = col[~np.isnan(col)]
        m = len(col)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[col[a]], idx[col[b]]] += 1.0 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    delta = np.zeros((k, k))
    for c in range(k):
        for d in range(k):
            lo, hi = min(c, d), max(c, d)
            delta[c, d] = (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[d]) / 2.0) ** 2
    do = (o * delta).sum() / n
    de = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1.0 - do / de) if de > 0 else 1.0

def ci_half_width(sd, n, conf=0.95):
    """t-based half width of the confidence interval of a mean: t_{(1+conf)/2, n-1} * sd / sqrt(n)."""
    return float(st.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n))
