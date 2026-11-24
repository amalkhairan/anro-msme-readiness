import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ANRO_DATA", os.path.join(HERE, "..", "data", "msme_digital_readiness_anonymized.csv"))
RES = os.path.join(HERE, "..", "results")
import sys, os, time, numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from mcnro import MCNRO, ANRO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from samplers import resample as own_resample
ROOT = os.environ.get("MCNRO_FOLDS", os.path.join(HERE, "..", "MC-NRO-master", "data", "folds"))  # folds from the MC-NRO reference repository
DS = sys.argv[1:] or sorted(os.listdir(ROOT))
RF = dict(n_estimators=500, criterion="gini", max_features="sqrt", bootstrap=True, random_state=42, n_jobs=2)
NRO = dict(k1=5, k2=5, eps=0.5, n_partitions=3, n_candidates=1000)
out = open(os.path.join(RES, "keel_results.jsonl"), "a")
for ds in DS:
    t0 = time.time()
    for pt in range(1, 6):
        for f in (1, 2):
            tr = pd.read_csv(f"{ROOT}/{ds}/{ds}.{pt}.{f}.train.csv").values
            te = pd.read_csv(f"{ROOT}/{ds}/{ds}.{pt}.{f}.test.csv").values
            Xt, yt = tr[:, :-1].astype(float), tr[:, -1].astype(int)
            Xs, ys = te[:, :-1].astype(float), te[:, -1].astype(int)
            seed = pt * 10 + f
            prior = np.bincount(yt, minlength=yt.max() + 1) / len(yt)
            sets, diag = {}, {}
            sets["none"] = (Xt, yt)
            sets["ros"] = own_resample(Xt, yt, "ros", random_state=seed)
            kmin = np.bincount(yt)[np.bincount(yt) > 0].min()
            try:
                sets["smote"] = own_resample(Xt, yt, "smote", random_state=seed)
            except Exception as e:
                sets["smote"] = sets["ros"]; diag["smote_fallback"] = 1
            for name, cls in [("mcnro", MCNRO), ("anro", ANRO)]:
                s = cls(**NRO, random_state=seed)
                Xr, yr = s.fit_resample(Xt, yt)
                nmin = sum(d["n_min"] for d in s.log_)
                syn = Xr[s.synthetic_mask_]
                diag[name] = dict(empty_rate=sum(d["n_short"] for d in s.log_) / max(nmin, 1),
                                  distinct=len(np.unique(syn.round(8), axis=0)) / max(len(syn), 1),
                                  removed=s.n_removed_)
                sets[name] = (Xr, yr)
            res = {}
            for name, (Xr, yr) in sets.items():
                m = RandomForestClassifier(**RF).fit(Xr, yr)
                P = m.predict_proba(Xs); cl = m.classes_
                preds = {name: cl[P.argmax(1)]}
                if name == "none":
                    adj = P / np.maximum(prior[cl], 1e-12)
                    preds["threshold"] = cl[adj.argmax(1)]
                for k, p in preds.items():
                    res[k] = dict(f1=f1_score(ys, p, average="macro"), ba=balanced_accuracy_score(ys, p))
            out.write(json.dumps(dict(ds=ds, part=pt, fold=f, res=res, diag=diag)) + "\n"); out.flush()
    print(ds, round(time.time() - t0, 1), "s", flush=True)
