"""Questionnaire encoding, identical to the final IKADO notebook (RF-MSME_final_revisi).

Every ordinal item is scored 1..5 following the option order of the Google Form.
Q20 follows the questionnaire ("Selalu Dibantu" = 1 ... "Tidak pernah dibantu" = 5),
Q24 is coded so that higher = greater concern (as in the IKADO paper).
Label items Q8, Q9, Q10, Q12, Q14 -> DRS (5-25) -> Low 5-11 / Medium 12-18 / High 19-25.
"""
import pandas as pd

LABEL_ITEMS = ["Q8", "Q9", "Q10", "Q12", "Q14"]
# Feature order reproduces the IKADO notebook exactly: Q2 sits at the END of the
# numeric block (it was dropped as "PII" because of the word "Smartphone" and
# re-appended later). Random Forest results depend on feature order, so this
# order is required to reproduce the published baseline (macro-F1 0.4470).
NONLABEL = [f"Q{i}" for i in range(1, 25) if f"Q{i}" not in LABEL_ITEMS + ["Q2"]] + ["Q2"]
DEMO = ["gender", "edu", "age", "btype", "market", "tenure", "turnover"]
DEMO_HEADERS = ["Jenis Kelamin", "Pendidikan Terakhir", "Usia", "Jenis Usaha/dagangan",
                "Lokasi Pasar", "Lama Jualan", "RataRata Omzet per BUlan"]

_FREQ = ["tidak pernah", "jarang", "kadang", "sering", "sangat sering"]
_MAMPU = ["sangat tidak mampu", "tidak mampu", "cukup mampu", "mampu", "sangat mampu"]
OPTIONS = {
    1: ["sangat tidak baik", "tidak baik", "cukup", "baik", "sangat baik"],
    2: ["sangat tidak memadai", "tidak memadai", "cukup memadai", "memadai", "sangat memadai"],
    3: ["tidak ada", "ada, tapi rusak", "ada, milik pribadi", "ada, milik usaha", "ada, lebih dari satu"],
    4: ["tidak pernah", "1- 3", "4 - 6", "7 -", "> 10"],
    5: _MAMPU, 6: _MAMPU, 8: _MAMPU,
    7: ["sangat tidak paham", "tidak paham", "cukup paham", "paham", "sangat paham"],
    9: ["hampir tidak ada", "hanya melayani 1-2", "melayani sebagian kecil",
        "melayani sebagian besar", "hampir seluruh"],
    10: ["tidak mempromosikan", "jarang", "kadang", "sering", "sangat sering"],
    11: _FREQ,
    12: ["tidak pernah", "sangat jarang", "kadang", "sering", "sangat sering"],
    13: ["sangat tidak paham", "kurang paham", "cukup paham", "paham", "sangat paham"],
    14: ["tidak pernah", "jarang", "kadang", "rutin", "sangat rutin"],
    15: ["sangat tidak penting", "kurang penting", "cukup penting", "penting", "sangat penting"],
    16: ["sama sekali", "manfaat kecil", "cukup bermanfaat", "bermanfaat", "sangat bermanfaat"],
    17: _FREQ, 18: _FREQ,
    19: ["sangat tidak mendukung", "kurang mendukung", "cukup mendukung", "mendukung", "sangat mendukung"],
    20: ["selalu dibantu", "sering dibantu", "kadang", "jarang dibantu", "tidak pernah dibantu"],
    21: ["sangat tidak percaya", "tidak percaya", "cukup percaya", "percaya", "sangat percaya"],
    22: ["sangat tidak aman", "tidak aman", "cukup aman", "aman", "sangat aman"],
    23: ["sangat tidak percaya", "kurang percaya", "cukup percaya", "percaya", "sangat percaya"],
    24: ["sangat tidak khawatir", "tidak khawatir", "cukup khawatir", "khawatir", "sangat khawatir"],
}


def _score(value, opts):
    """Longest matching option prefix -> score 1..5 (raises if unmatched)."""
    s = " ".join(str(value).replace(" ", " ").split()).lower()
    best = None
    for i, o in enumerate(opts):
        if s.startswith(o) and (best is None or len(o) > len(opts[best])):
            best = i
    if best is None:
        raise ValueError(f"Unmapped answer: {value!r}")
    return best + 1


def load(path):
    """Returns X (Q1..Q24 scores), demo (7 categorical columns), drs, y (0/1/2)."""
    d = pd.read_excel(path) if str(path).lower().endswith(".xlsx") else pd.read_csv(path)
    import re
    qcols = [c for c in d.columns if re.match(r"^\s*\d+\.", str(c))][:24]
    assert len(qcols) == 24, "Expected 24 questionnaire columns (1. ... 24.)"
    X = pd.DataFrame({f"Q{i+1}": d[c].map(lambda v, i=i: _score(v, OPTIONS[i + 1]))
                      for i, c in enumerate(qcols)})
    if all(h in d.columns for h in DEMO_HEADERS):      # raw export or anonymized CSV
        demo = d[DEMO_HEADERS].copy()
    else:                                               # fallback: original column positions
        demo = d.iloc[:, 4:11].copy()
    demo.columns = DEMO
    demo = demo.apply(lambda s: s.astype(str).str.strip())
    drs = X[LABEL_ITEMS].sum(axis=1)
    y = pd.cut(drs, [4, 11, 18, 25], labels=[0, 1, 2]).astype(int)
    return X, demo, drs, y
