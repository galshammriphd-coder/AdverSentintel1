"""Table 18: explainability ratings. Input csv columns: rater, sample_id, method, clarity, completeness, actionability (1-5)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats as st
from sentinelxlm.stats import krippendorff_alpha_ordinal
ap = argparse.ArgumentParser(); ap.add_argument("--ratings", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
df = pd.read_csv(a.ratings); dims = ["clarity", "completeness", "actionability"]; os.makedirs(a.out, exist_ok=True)
df.groupby("method")[dims].mean().round(2).to_csv(os.path.join(a.out, "table18_means.csv"))
rows = []
for d in dims:
    piv = df.assign(unit=df["method"] + ":" + df["sample_id"].astype(str)).pivot(index="rater", columns="unit", values=d)
    groups = [g[d].values for _, g in df.groupby("method")]
    F, p = st.f_oneway(*groups); tk = st.tukey_hsd(*groups)
    rows.append({"dimension": d, "krippendorff_alpha_ordinal": krippendorff_alpha_ordinal(piv.values), "anova_F": F, "anova_p": p})
pd.DataFrame(rows).to_csv(os.path.join(a.out, "table18_reliability.csv"), index=False); print(pd.DataFrame(rows))
