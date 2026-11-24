"""Discrimination (macro one-vs-rest ROC-AUC, macro PR-AUC, AP of High) and indicator-ranking
similarity (Spearman rho of mean importances against the baseline), computed from the raw
per-fold results in results/raw/<method>.pkl written by the notebook."""
import os, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
RAW = os.path.join(RES, "raw")
def load(m): return pickle.load(open(os.path.join(RAW, f"{m}.pkl"), "rb"))
# readiness class (0 = Low, 1 = Medium, 2 = High) of each survey row, so no survey data are needed
Y = pd.read_csv(os.path.join(RAW, "labels.csv")).sort_values("row").label.values
LABEL = {"baseline": "RF baseline (IKADO config)", "rf_tuned": "RF tuned (nested CV)",
         "class_weight": "RF class_weight=balanced", "threshold_prior": "RF + prior-corrected threshold",
         "ros": "ROS", "smote": "SMOTE", "smotenc": "SMOTE-NC", "borderline": "Borderline-SMOTE-1",
         "adasyn": "ADASYN", "mcnro": "MC-NRO (paper-faithful)",
         "mcnro+quantile": "MC-NRO + quantile repartition",
         "mcnro+majclean": "MC-NRO + majority-vote cleaning",
         "mcnro+ordinal": "MC-NRO + ordinal projection",
         "anro-noproj": "A-NRO without ordinal projection", "anro": "A-NRO (full)"}
auc, rho = [], []
base = load("baseline")
for m in LABEL:
    if not os.path.exists(os.path.join(RAW, f"{m}.pkl")):
        continue
    R = load(m)
    y = np.concatenate([Y[r["te"]] for r in R]); P = np.vstack([r["proba"] for r in R])
    Yb = np.eye(3)[y]
    auc.append({"Method": LABEL[m], "AUC": roc_auc_score(y, P, multi_class="ovr", average="macro"),
                "PRAUC": average_precision_score(Yb, P, average="macro"),
                "AP_High": average_precision_score(Yb[:, 2], P[:, 2])})
    for key, lab in (("gini26", "Gini"), ("perm26", "OOF permutation")):
        if all(r.get(key) is not None for r in R) and all(r.get(key) is not None for r in base):
            a = np.mean([r[key] for r in R], 0); b = np.mean([r[key] for r in base], 0)
            rho.append({"Method": LABEL[m], "Importance": lab, "rho": spearmanr(a, b).correlation})
pd.DataFrame(auc).to_csv(os.path.join(RES, "extra_auc.csv"), index=False)
pd.DataFrame(rho).to_csv(os.path.join(RES, "extra_rho.csv"), index=False)
print(pd.DataFrame(auc).round(3)); print(pd.DataFrame(rho).round(2))
