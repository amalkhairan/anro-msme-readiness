"""Tables for the manuscript, computed from results/<method>.pkl."""
import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
                             mean_absolute_error, confusion_matrix)
from anro_exp import METHODS, LABEL, PRED, Y

IKADO_SET = {"Q13", "Q15", "Q16", "Q7"}
IKADO_PUBLISHED = dict(Accuracy=0.7384, BalAcc=0.4555, MacroF1=0.4470, Kappa=0.3739, MCC=0.3825)


def load(name, folder="results"):
    return pickle.load(open(f"{folder}/{name}.pkl", "rb"))


def stack(res):
    t = np.concatenate([Y[r["te"]] for r in res])
    p = np.concatenate([r["pred"] for r in res])
    pr = np.vstack([r["proba"] for r in res])
    idx = np.concatenate([r["te"] for r in res])
    return t, p, pr, idx


def brier(t, pr):
    onehot = np.eye(pr.shape[1])[t]
    return float(np.mean(np.sum((pr - onehot) ** 2, axis=1)))


def pooled(res):
    t, p, pr, _ = stack(res)
    P, R, F, _ = precision_recall_fscore_support(t, p, labels=[0, 1, 2], zero_division=0)
    return dict(Accuracy=accuracy_score(t, p), BalAcc=balanced_accuracy_score(t, p),
                MacroF1=f1_score(t, p, average="macro"), WeightedF1=f1_score(t, p, average="weighted"),
                Kappa=cohen_kappa_score(t, p), MCC=matthews_corrcoef(t, p),
                QWK=cohen_kappa_score(t, p, weights="quadratic"), MAE=mean_absolute_error(t, p),
                MacroAUC=roc_auc_score(t, pr, multi_class="ovr", average="macro"),
                AUC_High=roc_auc_score((t == 2).astype(int), pr[:, 2]), Brier=brier(t, pr),
                P_Low=P[0], R_Low=R[0], F1_Low=F[0], P_Med=P[1], R_Med=R[1], F1_Med=F[1],
                P_High=P[2], R_High=R[2], F1_High=F[2],
                Extreme_errors=int(np.sum(np.abs(t - p) == 2)))


def per_fold(res, metric):
    f = (lambda t, p: f1_score(t, p, average="macro")) if metric == "f1" else balanced_accuracy_score
    return np.array([f(Y[r["te"]], r["pred"]) for r in res])


def per_rep(res):
    rows = []
    for k in range(10):
        rr = [r for r in res if r["rep"] == k]
        t = np.concatenate([Y[r["te"]] for r in rr]); p = np.concatenate([r["pred"] for r in rr])
        rows.append([f1_score(t, p, average="macro"), balanced_accuracy_score(t, p), np.mean(p[t == 2] == 2)])
    return np.array(rows)


def corrected_t(d, test_frac=0.2):
    """Nadeau & Bengio (2003) corrected resampled t-test on per-fold differences."""
    n, m, v = len(d), d.mean(), d.var(ddof=1)
    if v == 0:
        return 0.0, 1.0
    t = m / np.sqrt((1 / n + test_frac / (1 - test_frac)) * v)
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def respondent_votes(res):
    votes = np.zeros((len(Y), 3), int)
    for r in res:
        votes[r["te"], r["pred"]] += 1
    return votes


def respondent(res):
    v = respondent_votes(res); maj = v.argmax(1)
    out = {f"{n} correct (majority vote)": f"{int(np.sum((maj == c) & (Y == c)))}/{int(np.sum(Y == c))}"
           for c, n in [(0, "Low"), (1, "Medium"), (2, "High")]}
    out["High hit at least once (of 10 reps)"] = f"{int(np.sum(v[Y == 2, 2] > 0))}/9"
    return out


def bootstrap(res, base_res=None, B=2000, seed=0):
    """Respondent-level stratified bootstrap of pooled OOF predictions (all 10 reps per respondent)."""
    rng = np.random.default_rng(seed)
    t, p, _, idx = stack(res)
    if base_res is not None:
        tb, pb, _, idxb = stack(base_res)
        order_b = np.argsort(idxb, kind="stable")
    order = np.argsort(idx, kind="stable")
    t, p = t[order].reshape(len(Y), -1), p[order].reshape(len(Y), -1)     # respondent x reps
    if base_res is not None:
        pb = pb[order_b].reshape(len(Y), -1)
    cls_idx = [np.where(Y == c)[0] for c in range(3)]
    f1s, rh, d = [], [], []
    for _ in range(B):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in cls_idx])
        tt, pp = t[s].ravel(), p[s].ravel()
        f1s.append(f1_score(tt, pp, average="macro")); rh.append(np.mean(pp[tt == 2] == 2))
        if base_res is not None:
            d.append(f1s[-1] - f1_score(tt, pb[s].ravel(), average="macro"))
    q = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
    out = {"MacroF1 95% CI": q(f1s), "Recall High 95% CI": q(rh)}
    if base_res is not None:
        out["dMacroF1 vs baseline 95% CI"] = q(d)
        out["P(dMacroF1 > 0)"] = float(np.mean(np.array(d) > 0))
    return out


def stability(res, key):
    arr = [r[key] for r in res if r[key] is not None]
    if not arr:
        return None
    imp = np.vstack(arr)
    ranks = (-imp).argsort(1).argsort(1) + 1
    mean_imp = imp.mean(0)
    top4 = [PRED[i] for i in np.argsort(-mean_imp)[:4]]
    return dict(top4=top4, jaccard=len(set(top4) & IKADO_SET) / len(set(top4) | IKADO_SET),
                top10=({q: float(np.mean(ranks[:, PRED.index(q)] <= 10)) for q in ["Q13", "Q15", "Q16", "Q7", "Q2"]}),
                mean_imp=mean_imp)


def build_tables(folder="results", B=2000):
    avail = [n for n in METHODS if os.path.exists(f"{folder}/{n}.pkl")]
    R = {n: load(n, folder) for n in avail}
    base = R["baseline"]
    bf1, bba = per_fold(base, "f1"), per_fold(base, "ba")
    thr = R.get("threshold_prior")
    tables = {k: [] for k in ["performance", "per_repetition", "significance", "bootstrap_CI",
                              "respondent_level", "confusion_matrices", "nro_diagnostics",
                              "indicator_stability", "sampler_notes"]}
    for n in avail:
        res, L = R[n], LABEL[n]
        tables["performance"].append({"Method": L, **pooled(res)})
        pr = per_rep(res)
        tables["per_repetition"].append({"Method": L,
            "MacroF1 mean±sd": f"{pr[:,0].mean():.3f} ± {pr[:,0].std(ddof=1):.3f}",
            "BalAcc mean±sd": f"{pr[:,1].mean():.3f} ± {pr[:,1].std(ddof=1):.3f}",
            "Recall High mean±sd": f"{pr[:,2].mean():.3f} ± {pr[:,2].std(ddof=1):.3f}"})
        f1, ba = per_fold(res, "f1"), per_fold(res, "ba")
        row = {"Method": L, "dMacroF1 vs baseline": (f1 - bf1).mean(), "p corrected t (F1)": corrected_t(f1 - bf1)[1],
               "dBalAcc vs baseline": (ba - bba).mean(), "p corrected t (BA)": corrected_t(ba - bba)[1]}
        if thr is not None:
            tf1 = per_fold(thr, "f1")
            row.update({"dMacroF1 vs threshold": (f1 - tf1).mean(), "p vs threshold": corrected_t(f1 - tf1)[1]})
        tables["significance"].append(row)
        tables["bootstrap_CI"].append({"Method": L, **bootstrap(res, None if n == "baseline" else base, B=B)})
        tables["respondent_level"].append({"Method": L, **respondent(res)})
        t, p, _, _ = stack(res); cm = confusion_matrix(t, p, labels=[0, 1, 2])
        tables["confusion_matrices"].append({"Method": L, **{f"{a}->{b}": int(cm[i, j])
            for i, a in enumerate(["Low", "Med", "High"]) for j, b in enumerate(["Low", "Med", "High"])}})
        ds = [r["diag"] for r in res if r["diag"]]
        if ds and "n_moved" in ds[0]:
            tc = np.array([d["train_counts"] for d in ds])
            tables["nro_diagnostics"].append({"Method": L,
                "moved/fold": np.mean([d["n_moved"] for d in ds]),
                "removed/fold": np.mean([d["n_removed"] for d in ds]),
                "short seeds Medium": np.mean([d["n_short"][0] for d in ds]),
                "short seeds High": np.mean([d["n_short"][1] for d in ds]),
                "unique synthetic High / generated": f"{np.mean([d['uniq_syn_high'] for d in ds]):.1f} / {np.mean([d['n_syn_high'] for d in ds]):.1f}",
                "train Low/Med/High after": "/".join(f"{v:.1f}" for v in tc.mean(0))})
        fails = [d.get("sampler_failed") for d in ds if d.get("sampler_failed")]
        if fails or n == "rf_tuned":
            note = f"{len(fails)}/50 folds sampler failed (no resampling): {fails[0]}" if fails else ""
            if n == "rf_tuned":
                note = "best params: " + "; ".join(f"{k} x{v}" for k, v in
                                                     pd.Series([str(d.get('best')) for d in ds]).value_counts().items())
            tables["sampler_notes"].append({"Method": L, "Note": note})
        for key, lab in [("gini26", "Gini"), ("perm26", "OOF permutation")]:
            s = stability(res, key)
            if s:
                tables["indicator_stability"].append({"Method": L, "Importance": lab,
                    "Top-4": ", ".join(s["top4"]), "Jaccard top-4 vs IKADO set": s["jaccard"],
                    **{f"Top-10 freq {q}": v for q, v in s["top10"].items()}})
    tables = {k: pd.DataFrame(v) for k, v in tables.items()}
    b = tables["performance"].iloc[0]
    tables["baseline_check"] = pd.DataFrame([{"Metric": k, "IKADO published": v, "This run": b[k],
                                              "abs diff": abs(b[k] - v)} for k, v in IKADO_PUBLISHED.items()])
    return tables
