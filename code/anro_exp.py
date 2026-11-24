"""Experiment core for 'Beyond the Baseline' (paper 1).

Protocol = IKADO baseline paper:
  19 non-label ordinal items + 7 respondent/business characteristics (26 predictors);
  imputation + one-hot fitted on the training fold; repeated stratified 5-fold x 10
  (random_state=42); RF 500 trees, gini, sqrt, bootstrap, random_state=42.
  Every resampler is applied to the TRAINING fold only.
The data path is read from the environment variable ANRO_DATA so that joblib
worker processes can load the same data.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import f1_score

from encode import load, NONLABEL, DEMO
from mcnro import MCNRO, ANRO

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, BorderlineSMOTE, ADASYN
    HAVE_IMBLEARN = True
except ImportError:                          # local fallback (not for the manuscript)
    HAVE_IMBLEARN = False
    from samplers import resample as _own_resample

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "msme_digital_readiness_anonymized.csv")
X, demo, drs, y = load(os.environ.get("ANRO_DATA", _DEFAULT))
NL, PRED = NONLABEL, NONLABEL + DEMO
D = pd.concat([X[NL], demo], axis=1)
Y = y.values
CV = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).split(D, Y))
RF = dict(n_estimators=500, criterion="gini", max_features="sqrt", bootstrap=True, random_state=42)
NRO_PAR = dict(k1=5, k2=5, eps=0.5, n_partitions=3, n_candidates=1000)
PERM_REPEATS = int(os.environ.get("ANRO_PERM_REPEATS", 10))

#            name: (sampler, extra RF kwargs, decision rule, tune, OOF permutation)
METHODS = {
    "baseline":        (None, {}, "argmax", False, True),
    "rf_tuned":        (None, {}, "argmax", True, False),
    "class_weight":    (None, {"class_weight": "balanced"}, "argmax", False, True),
    "threshold_prior": (None, {}, "prior", False, True),
    "ros":             ("ros", {}, "argmax", False, False),
    "smote":           ("smote", {}, "argmax", False, True),
    "smotenc":         ("smotenc", {}, "argmax", False, True),
    "borderline":      ("borderline", {}, "argmax", False, False),
    "adasyn":          ("adasyn", {}, "argmax", False, False),
    "mcnro":           ("mcnro", {}, "argmax", False, True),
    "mcnro+quantile":  ("mcnro+quantile", {}, "argmax", False, False),
    "mcnro+majclean":  ("mcnro+majclean", {}, "argmax", False, False),
    "mcnro+ordinal":   ("mcnro+ordinal", {}, "argmax", False, False),
    "anro-noproj":     ("anro-noproj", {}, "argmax", False, False),
    "anro":            ("anro", {}, "argmax", False, True),
}
LABEL = {"baseline": "RF baseline (IKADO config)", "rf_tuned": "RF tuned (nested CV)",
         "class_weight": "RF class_weight=balanced", "threshold_prior": "RF + prior-corrected threshold",
         "ros": "ROS", "smote": "SMOTE", "smotenc": "SMOTE-NC", "borderline": "Borderline-SMOTE-1",
         "adasyn": "ADASYN", "mcnro": "MC-NRO (paper-faithful)",
         "mcnro+quantile": "MC-NRO + quantile repartition",
         "mcnro+majclean": "MC-NRO + majority-vote cleaning",
         "mcnro+ordinal": "MC-NRO + ordinal projection",
         "anro-noproj": "A-NRO without ordinal projection", "anro": "A-NRO (full)"}


# ---------------------------------------------------------------- preprocessing
class StdPrep:
    """Median / mode imputation + one-hot (same as the IKADO pipeline)."""
    def fit(self, Dtr):
        self.ct = ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NL),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), DEMO)],
            remainder="drop").fit(Dtr)
        cats = self.ct.named_transformers_["categorical"].named_steps["onehot"].categories_
        self.owner, self.groups, start = list(range(len(NL))), [], len(NL)
        for j, cs in enumerate(cats):
            self.groups.append(list(range(start, start + len(cs))))
            self.owner += [len(NL) + j] * len(cs)
            start += len(cs)
        self.owner = np.array(self.owner)
        return self

    def transform(self, Dx):
        M = self.ct.transform(Dx)
        return M.toarray() if hasattr(M, "toarray") else np.asarray(M, float)


class NCPrep:
    """For SMOTE-NC: impute, ordinal-code categories, oversample, then one-hot."""
    def fit(self, Dtr):
        self.num_imp = SimpleImputer(strategy="median").fit(Dtr[NL])
        self.cat_imp = SimpleImputer(strategy="most_frequent").fit(Dtr[DEMO])
        self.ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
            self.cat_imp.transform(Dtr[DEMO]))
        n_levels = [len(c) for c in self.ord.categories_]
        self.oh = OneHotEncoder(categories=[list(range(n)) for n in n_levels],
                                handle_unknown="ignore").fit(np.zeros((1, len(DEMO))))
        self.owner, self.groups, start = list(range(len(NL))), [], len(NL)
        for j, n in enumerate(n_levels):
            self.groups.append(list(range(start, start + n)))
            self.owner += [len(NL) + j] * n
            start += n
        self.owner = np.array(self.owner)
        return self

    def raw(self, Dx):            # numeric + ordinal codes (for SMOTE-NC)
        return np.hstack([self.num_imp.transform(Dx[NL]),
                          self.ord.transform(self.cat_imp.transform(Dx[DEMO]))])

    def encode(self, R):          # raw -> model matrix
        oh = self.oh.transform(np.rint(R[:, len(NL):]).astype(int))
        oh = oh.toarray() if hasattr(oh, "toarray") else oh
        return np.hstack([R[:, :len(NL)], oh])

    def transform(self, Dx):
        return self.encode(self.raw(Dx))


# ---------------------------------------------------------------- samplers
def _imb(method, Xt, yt, seed):
    kmin = int(np.bincount(yt).min())
    k = max(1, min(5, kmin - 1))
    if not HAVE_IMBLEARN:
        return _own_resample(Xt, yt, method, random_state=seed)
    s = {"ros": lambda: RandomOverSampler(random_state=seed),
         "smote": lambda: SMOTE(k_neighbors=k, random_state=seed),
         "borderline": lambda: BorderlineSMOTE(kind="borderline-1", k_neighbors=k,
                                               m_neighbors=10, random_state=seed),
         "adasyn": lambda: ADASYN(n_neighbors=k, random_state=seed)}[method]()
    return s.fit_resample(Xt, yt)


def resample_fold(method, Xt, yt, groups, seed, prep=None, Dtr=None):
    diag = {}
    if method in ("ros", "smote", "borderline", "adasyn"):
        try:
            Xr, yr = _imb(method, Xt, yt, seed)
        except (RuntimeError, ValueError) as e:          # e.g. ADASYN with no hard samples
            Xr, yr = Xt, yt
            diag["sampler_failed"] = str(e)[:120]
        diag["train_counts"] = np.bincount(yr, minlength=3).tolist()
        return Xr, yr, diag
    if method == "smotenc":
        R = prep.raw(Dtr)
        k = max(1, min(5, int(np.bincount(yt).min()) - 1))
        cat_idx = list(range(len(NL), len(NL) + len(DEMO)))
        if HAVE_IMBLEARN:
            Rr, yr = SMOTENC(categorical_features=cat_idx, k_neighbors=k,
                             random_state=seed).fit_resample(R, yt)
        else:
            Rr, yr = R, yt
            diag["sampler_failed"] = "imbalanced-learn not installed"
        diag["train_counts"] = np.bincount(yr, minlength=3).tolist()
        return prep.encode(Rr), yr, diag
    ordc = list(range(len(NL)))
    s = {
        "mcnro":          lambda: MCNRO(**NRO_PAR, random_state=seed),
        "mcnro+quantile": lambda: MCNRO(**NRO_PAR, repartition="quantile", random_state=seed),
        "mcnro+majclean": lambda: MCNRO(**NRO_PAR, clean_rule="majority", random_state=seed),
        "mcnro+ordinal":  lambda: MCNRO(**NRO_PAR, ordinal_cols=ordc, onehot_groups=groups,
                                        project_moved=True, random_state=seed),
        "anro-noproj":    lambda: ANRO(**NRO_PAR, random_state=seed),
        "anro":           lambda: ANRO(**NRO_PAR, ordinal_cols=ordc, onehot_groups=groups,
                                       random_state=seed),
    }[method]()
    Xr, yr = s.fit_resample(Xt, yt)
    syn = Xr[s.synthetic_mask_]
    diag = dict(n_moved=s.n_moved_, n_removed=s.n_removed_,
                n_short=[d["n_short"] for d in s.log_],
                uniq_syn_high=len(np.unique(np.round(syn[yr[s.synthetic_mask_] == 2], 8), axis=0)),
                n_syn_high=int((yr[s.synthetic_mask_] == 2).sum()),
                train_counts=np.bincount(yr, minlength=3).tolist())
    return Xr, yr, diag


# ---------------------------------------------------------------- model
def tune_rf(Xt, yt, seed):
    grid = [dict(max_features=mf, min_samples_leaf=ml, max_depth=md)
            for mf in ("sqrt", 0.5) for ml in (1, 3, 5) for md in (None, 8)]
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, best_s = None, -1.0
    for g in grid:
        s = []
        for a, b in inner.split(Xt, yt):
            m = RandomForestClassifier(n_estimators=200, criterion="gini", bootstrap=True,
                                       random_state=42, n_jobs=1, **g).fit(Xt[a], yt[a])
            s.append(f1_score(yt[b], m.predict(Xt[b]), average="macro"))
        if np.mean(s) > best_s:
            best, best_s = g, float(np.mean(s))
    return best


def decide(proba, rule, prior):
    return (proba / prior).argmax(1) if rule == "prior" else proba.argmax(1)


def fold_job(fi, name, perm=None):
    samp, rfkw, rule, tune, do_perm = METHODS[name]
    do_perm = do_perm if perm is None else perm
    tr, te = CV[fi]
    prep = (NCPrep() if samp == "smotenc" else StdPrep()).fit(D.iloc[tr])
    Xt, Xs = prep.transform(D.iloc[tr]), prep.transform(D.iloc[te])
    yt = Y[tr]
    prior = np.bincount(yt, minlength=3) / len(yt)
    diag = {}
    if samp is not None:
        Xt, yt, diag = resample_fold(samp, Xt, yt, prep.groups, seed=1000 + fi,
                                     prep=prep, Dtr=D.iloc[tr])
    kw = dict(RF); kw.update(rfkw)
    if tune:
        best = tune_rf(Xt, yt, seed=fi); kw.update(best); diag["best"] = best
    m = RandomForestClassifier(n_jobs=1, **kw).fit(Xt, yt)
    proba = m.predict_proba(Xs)
    pred = decide(proba, rule, prior)
    gini26 = np.bincount(prep.owner, weights=m.feature_importances_, minlength=len(PRED))
    perm26 = None
    if do_perm:
        rng = np.random.default_rng(fi)
        base = f1_score(Y[te], pred, average="macro")
        perm26 = np.zeros(len(PRED))
        for j, col in enumerate(PRED):
            drops = []
            for _ in range(PERM_REPEATS):
                Dp = D.iloc[te].copy()
                Dp[col] = rng.permutation(Dp[col].values)
                pp = decide(m.predict_proba(prep.transform(Dp)), rule, prior)
                drops.append(base - f1_score(Y[te], pp, average="macro"))
            perm26[j] = np.mean(drops)
    return dict(fold=fi, rep=fi // 5, te=te, pred=pred, proba=proba,
                gini26=gini26, perm26=perm26, diag=diag)


def sens_job(fi, algo, k1, eps, n):
    tr, te = CV[fi]
    prep = StdPrep().fit(D.iloc[tr])
    Xt, Xs = prep.transform(D.iloc[tr]), prep.transform(D.iloc[te])
    par = dict(k1=k1, k2=5, eps=eps, n_partitions=n, n_candidates=1000, random_state=1000 + fi)
    s = MCNRO(**par) if algo == "MC-NRO" else ANRO(**par, ordinal_cols=list(range(len(NL))),
                                                    onehot_groups=prep.groups)
    Xr, yr = s.fit_resample(Xt, Y[tr])
    p = RandomForestClassifier(n_jobs=1, **RF).fit(Xr, yr).predict(Xs)
    return Y[te], p
