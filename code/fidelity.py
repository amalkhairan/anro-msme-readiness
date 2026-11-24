import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ANRO_DATA", os.path.join(HERE, "..", "data", "msme_digital_readiness_anonymized.csv"))
RES = os.path.join(HERE, "..", "results")
# Re-implementation of the sampling rule in the public MC-NRO code (algorithm.py, NRO.fit_sample,
# 'proportional' branch, region 'S'): L2 distance, potential over ALL minority points of the class,
# radius = distance to the (k1+1)-th majority neighbour - 0.01, candidates = random cube direction
# normalised * U(0,1) * r, safe threshold = seed + 0.33 (max(scores + [seed]) - seed).
import os, sys, numpy as np, pandas as pd, pickle
import anro_exp as E
from scipy.spatial.distance import cdist
rng = np.random.default_rng(0)
def pot(P, M, g): return np.exp(-(cdist(np.atleast_2d(P), M) / g) ** 2).sum(1)
rows = []
for fi in range(0, 50, 5):
    tr, te = E.CV[fi]; prep = E.StdPrep().fit(E.D.iloc[tr]); X = prep.transform(E.D.iloc[tr]); y = E.Y[tr]
    lo, hi = X.min(0), X.max(0); Z = (X - lo) / np.where(hi - lo > 0, hi - lo, 1)
    for c, name in [(1, "Medium"), (2, "High")]:
        Mn = Z[y == c]; Mj = Z[y < c]
        D = cdist(Mn, Mj)
        for g in [0.1, 0.5, 2.5, "median"]:
            gg = float(np.median(np.sort(D, 1)[:, 1])) if g == "median" else g
            for i in range(len(Mn)):
                r = max(np.sort(D[i])[1] - 0.01, 1e-6)          # k1 = 1 (code default), argsort()[k1]
                dirs = 2 * rng.random((500, Z.shape[1])) - 1
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                C = Mn[i] + dirs * rng.random((500, 1)) * r
                s0 = pot(Mn[i], Mn, gg)[0]; sc = pot(C, Mn, gg)
                rows.append(dict(fold=fi, cls=name, gamma=str(g), empty=bool(sc.max() < s0)))
r = pd.DataFrame(rows)
print(r.groupby(["gamma", "cls"]).empty.agg(["size", "mean"]))
r.to_csv(os.path.join(RES, "fidelity_reference_rule.csv"), index=False)
d = pd.DataFrame(pickle.load(open(os.path.join(RES, "degen.pkl"), "rb")))
print("share of seed visits with only the seed in M_i:", d.groupby("cls").nloc.apply(lambda s: (s == 1).mean()).round(3).to_dict())
