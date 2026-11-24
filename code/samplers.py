"""Plain implementations of random oversampling and SMOTE (multiclass, one-vs-rest,
oversampling each class to the size of the largest). Used in the KEEL benchmark
(keel_bench.py); the survey experiments use imbalanced-learn 0.13.0."""
import numpy as np
from scipy.spatial.distance import cdist


def _interp(Z, seeds, k, n_per_seed, rng):
    """SMOTE interpolation from seeds (indices into Z) toward minority NNs."""
    if len(Z) < 2:
        return np.repeat(Z[seeds], n_per_seed, axis=0)
    kk = min(k, len(Z) - 1)
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1, kind="stable")[:, :kk]
    a = np.repeat(seeds, n_per_seed)
    b = nn[a, rng.integers(0, kk, len(a))]
    g = rng.random((len(a), 1))
    return Z[a] + g * (Z[b] - Z[a])


def _alloc_even(n_seeds, n, rng):
    base = np.full(n_seeds, n // n_seeds)
    base[rng.choice(n_seeds, n % n_seeds, replace=False)] += 1
    return base


def resample(X, y, method, k=5, m=10, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float); y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_max = counts.max()
    Xs, ys, info = [X], [y], {}
    for c, nc in zip(classes, counts):
        n = n_max - nc
        if n <= 0:
            continue
        idx = np.where(y == c)[0]
        Z = X[idx]
        if method == "ros":
            new = Z[rng.choice(len(Z), n)]
        elif method == "smote":
            per = _alloc_even(len(Z), n, rng)
            new = np.vstack([_interp(Z, np.array([i]), k, per[i], rng) for i in range(len(Z)) if per[i]])
        elif method in ("borderline", "adasyn"):
            K = m if method == "borderline" else k
            Dall = cdist(Z, X)
            Dall[np.arange(len(Z)), idx] = np.inf
            nnall = np.argsort(Dall, axis=1, kind="stable")[:, :min(K, len(X) - 1)]
            n_foreign = (y[nnall] != c).sum(1)
            if method == "borderline":
                danger = np.where((n_foreign >= K / 2) & (n_foreign < K))[0]
                info[c] = dict(n_danger=len(danger))
                if len(danger) == 0:           # same behaviour as imbalanced-learn: nothing to do
                    continue
                per = _alloc_even(len(danger), n, rng)
                new = np.vstack([_interp(Z, np.array([danger[i]]), k, per[i], rng)
                                 for i in range(len(danger)) if per[i]])
            else:
                r = n_foreign / K
                info[c] = dict(n_hard=int((r > 0).sum()))
                if r.sum() == 0:
                    continue
                g = np.rint(r / r.sum() * n).astype(int)
                new = np.vstack([_interp(Z, np.array([i]), k, g[i], rng) for i in range(len(Z)) if g[i]])
        else:
            raise ValueError(method)
        Xs.append(new); ys.append(np.full(len(new), c, dtype=y.dtype))
    return np.vstack(Xs), np.concatenate(ys)
