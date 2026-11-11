import numpy as np
from sentinelxlm.stats import mean_sd_ci, ci_half_width, loglog_slope, krippendorff_alpha_ordinal, paired_ttest

def test_reported_confidence_intervals():
    assert round(ci_half_width(0.14, 5), 2) == 0.17           # SENTINEL-XLM UNSW accuracy: [97.66, 98.00]
    assert (round(97.83 - ci_half_width(0.14, 5), 2), round(97.83 + ci_half_width(0.14, 5), 2)) == (97.66, 98.00)
    assert (round(90.56 - ci_half_width(0.19, 5), 2), round(90.56 + ci_half_width(0.19, 5), 2)) == (90.32, 90.80)   # GAT-Security
    assert (round(87.78 - ci_half_width(0.24, 5), 2), round(87.78 + ci_half_width(0.24, 5), 2)) == (87.48, 88.08)   # GCN-Trust F1

def test_cicids_per_run_statistics():
    m, sd, lo, hi = mean_sd_ci([97.99, 98.07, 98.13, 98.19, 98.22])
    assert (round(m, 2), round(sd, 2), round(lo, 2), round(hi, 2)) == (98.12, 0.09, 98.00, 98.24)
    m, sd, lo, hi = mean_sd_ci([96.87, 97.01, 97.09, 97.17, 97.16])
    assert (round(m, 2), round(sd, 2), round(lo, 2), round(hi, 2)) == (97.06, 0.12, 96.91, 97.21)

def test_scalability_slopes_of_table_14b():
    N = [100, 500, 1000, 2000, 5000]
    assert round(loglog_slope(N, [1.23, 3.03, 4.17, 6.23, 9.63]), 2) == 0.53
    assert round(loglog_slope(N, [20, 89, 203, 312, 498]), 2) == 0.84

def test_krippendorff_and_paired_test():
    R = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], float)
    assert abs(krippendorff_alpha_ordinal(R) - 1.0) < 1e-9
    t, p = paired_ttest([1, 2, 3, 4, 5.5], [1, 2, 3, 4, 5]); assert p < 1.0
