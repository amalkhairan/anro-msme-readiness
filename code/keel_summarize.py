"""Summarise keel_results.jsonl (written by keel_bench.py) into keel_summary.csv:
mean macro-F1 / balanced accuracy over the 10 folds per dataset and method, and the
MC-NRO / A-NRO empty-region and distinct-row diagnostics. Also prints mean ranks and
Wilcoxon signed-rank tests across datasets, Holm-adjusted over all 15 pairs per metric (Table 7 of the manuscript)."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
M = ["none", "threshold", "ros", "smote", "mcnro", "anro"]
rows = [json.loads(l) for l in open(os.path.join(RES, "keel_results.jsonl"))]
recs = []
for r in rows:
    d = {"ds": r["ds"]}
    for m in M:
        d[f"f1_{m}"], d[f"ba_{m}"] = r["res"][m]["f1"], r["res"][m]["ba"]
    for m in ("mcnro", "anro"):
        d[f"empty_{m}"], d[f"distinct_{m}"] = r["diag"][m]["empty_rate"], r["diag"][m]["distinct"]
    recs.append(d)
S = pd.DataFrame(recs).groupby("ds").mean().reset_index()
S.to_csv(os.path.join(RES, "keel_summary.csv"), index=False)
for metric in ("f1", "ba"):
    V = S[[f"{metric}_{m}" for m in M]].values
    ranks = np.apply_along_axis(lambda v: rankdata(-v), 1, V).mean(0)
    print(metric, "mean ranks:", dict(zip(M, ranks.round(2))))
# Holm adjustment over all 15 pairwise comparisons of the six methods, separately per metric
import itertools
for mt in ("f1", "ba"):
    pairs = list(itertools.combinations(M, 2))
    p = [wilcoxon(S[f"{mt}_{a}"], S[f"{mt}_{b}"]).pvalue for a, b in pairs]
    o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(1, run)
    print(f"\n{mt}: Wilcoxon signed-rank, Holm over 15 pairs")
    for (a, b), pi, ai in zip(pairs, p, adj):
        wins = int((S[f"{mt}_{a}"] > S[f"{mt}_{b}"]).sum())
        print(f"  {a:9s} vs {b:9s}: {a} better on {wins:2d}/20, p = {pi:.4f}, Holm p = {ai:.4f}")
