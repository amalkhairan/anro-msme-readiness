import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ANRO_DATA", os.path.join(HERE, "..", "data", "msme_digital_readiness_anonymized.csv"))
RES = os.path.join(HERE, "..", "results")
import os, sys, numpy as np, pickle
import mcnro, anro_exp as E
rec = []; cur = {}
orig = mcnro._potential
def wrap(points, ref, eps, p):
    out = orig(points, ref, eps, p)
    if np.ndim(points) == 1 or len(points) == 1:
        cur["phi0"] = float(out[0]); cur["nloc"] = len(ref)
    else:
        rec.append(dict(cls=cur["cls"], phi0=cur["phi0"], umax=float(out.max()),
                        umed=float(np.median(out)), nloc=cur["nloc"], fold=cur["fold"]))
    return out
mcnro._potential = wrap
orun = mcnro.BNRO.run
def run(self, X_maj, X_min, n):
    cur["cls"] = "High" if len(X_min) < 15 else "Medium"
    return orun(self, X_maj, X_min, n)
mcnro.BNRO.run = run
for fi in range(len(E.CV)):
    cur["fold"] = fi
    tr, te = E.CV[fi]
    prep = E.StdPrep().fit(E.D.iloc[tr]); Xt = prep.transform(E.D.iloc[tr])
    mcnro.MCNRO(**E.NRO_PAR, random_state=1000 + fi).fit_resample(Xt, E.Y[tr])
pickle.dump(rec, open(os.path.join(RES, "degen.pkl"), "wb"))
import pandas as pd
r = pd.DataFrame(rec); r["empty"] = r.phi0 > r.umax
print(r.groupby("cls").agg(n=("empty","size"), empty=("empty","mean"), nloc=("nloc","mean")))
print(r.describe())
