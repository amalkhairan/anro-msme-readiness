import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ANRO_DATA", os.path.join(HERE, "..", "data", "msme_digital_readiness_anonymized.csv"))
RES = os.path.join(HERE, "..", "results")
import os, sys, numpy as np, pandas as pd
import mcnro, anro_exp as E
from scipy.spatial.distance import cdist
rec = []; cur = {}
orig = mcnro._potential
def wrap(points, ref, eps, p):
    out = orig(points, ref, eps, p)
    if np.ndim(points) == 1 or len(points) == 1: cur["phi0"] = float(out[0])
    else: rec.append(dict(cur, umax=float(out.max())))
    return out
mcnro._potential = wrap
def scaled(fi):
    tr, te = E.CV[fi]; prep = E.StdPrep().fit(E.D.iloc[tr]); X = prep.transform(E.D.iloc[tr])
    lo, hi = X.min(0), X.max(0); return (X - lo) / np.where(hi - lo > 0, hi - lo, 1), E.Y[tr], X
for fi in range(0, 50, 5):           # 10 folds (one per repetition) for speed
    Z, y, Xraw = scaled(fi)
    D = cdist(Z, Z, "minkowski", p=1)
    radii = [np.sort(D[i, y != y[i]])[4] for i in np.where(y > 0)[0]]
    med_r = float(np.median(radii))
    for mult in [0.25, 0.5, 1.0, 2.0]:
        cur.update(fold=fi, scale="median radius", mult=mult, eps=mult * med_r)
        par = dict(E.NRO_PAR); par["eps"] = mult * med_r
        mcnro.MCNRO(**par, random_state=1000 + fi).fit_resample(Xraw, y)
r = pd.DataFrame(rec); r["empty"] = r.phi0 > r.umax
print(r.groupby("mult").agg(eps=("eps", "mean"), visits=("empty", "size"), empty_rate=("empty", "mean")))
r.to_csv(os.path.join(RES, "degen_adaptive_eps.csv"), index=False)
