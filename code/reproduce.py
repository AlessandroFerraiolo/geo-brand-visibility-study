"""Reproduce the Chapter 4 conditional-logit results from data/data.csv (no Stata required).
Run from the code/ directory:  python reproduce.py
Requires: pandas, numpy, scipy, statsmodels."""
import numpy as np, pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy import stats
from math import comb, log

df = pd.read_csv("../data/data.csv", dtype=str)
df["avgrating_b_g2"] = df["avgrating_b_g2"].str.replace(",", ".", regex=False).astype(float)
for c in ["reviewcount_b_g2","lighthouse_seo_b","listicle_topic_hits_bt","reddit_topic_hits_bt",
          "youtube_topic_hits_bt","linkedin_topic_hits_bt","domain_topic_hits_bt","mention","query_id"]:
    df[c] = pd.to_numeric(df[c])

z  = lambda x: (x - x.mean()) / x.std(ddof=1)   # Stata egen std = sample SD
ln = lambda v: np.log(1 + v)
df["z_social_buzz"]  = z(ln(df.reddit_topic_hits_bt + df.youtube_topic_hits_bt + df.linkedin_topic_hits_bt))
df["z_articles"]     = z(ln(df.listicle_topic_hits_bt + df.domain_topic_hits_bt))
df["z_review_index"] = z(df.avgrating_b_g2 * ln(df.reviewcount_b_g2))
df["z_seo"]          = z(df.lighthouse_seo_b.astype(float))
df["gemini"]         = (df.source == "gemini").astype(float)   # chatgpt = base

terms = ["z_social_buzz", "z_articles", "z_review_index", "z_seo", "gemini"]
res = ConditionalLogit(df.mention, df[terms], groups=df.query_id).fit(disp=0)
b = res.params.values

# cluster-robust SEs by query (clusters = groups), Stata G/(G-1) correction
A = -res.model.hessian(b); B = np.zeros((len(b),) * 2)
for q, idx in df.groupby("query_id").groups.items():
    sub = df.loc[idx]; k = sub.mention.sum(); n = len(sub)
    if 0 < k < n:
        sg = ConditionalLogit(sub.mention, sub[terms], groups=sub.query_id).score(b)
        B += np.outer(sg, sg)
G = df.query_id.nunique()
V = np.linalg.inv(A) @ B @ np.linalg.inv(A) * G / (G - 1.0)
se = np.sqrt(np.diag(V)); zt = b / se; pv = 2 * stats.norm.sf(abs(zt))
tab = pd.DataFrame({"coef": b, "clusterSE": se, "z": zt, "p": pv, "OR": np.exp(b),
                    "CI_lo": np.exp(b - 1.96 * se), "CI_hi": np.exp(b + 1.96 * se)},
                   index=terms).round(4)
ll0 = sum(-log(comb(len(g), int(g.mention.sum()))) for _, g in df.groupby("query_id"))
print(tab.to_string())
print(f"\nMcFadden pseudo-R2 = {1 - res.llf / ll0:.4f}  (llf={res.llf:.3f}, ll0={ll0:.3f})")

# LR test: are the four marketing slopes equal across engines?
for t in ["z_social_buzz", "z_articles", "z_review_index", "z_seo"]:
    df[t + "_X"] = df[t] * df.gemini
ru = ConditionalLogit(df.mention,
        df[terms + [t + "_X" for t in ["z_social_buzz","z_articles","z_review_index","z_seo"]]],
        groups=df.query_id).fit(disp=0)
LR = 2 * (ru.llf - res.llf)
print(f"LR test (equal slopes across engines): LR={LR:.3f}, df=4, p={stats.chi2.sf(LR, 4):.3f}")
