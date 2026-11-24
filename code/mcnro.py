"""
mcnro.py - Clean re-implementation of MC-NRO
(Multiclass Neighborhood Repartition-based Oversampling)

Reference
---------
S. Shen, Z. Li, Z. Huan, F. Shang, Y. Wang, Y. Chen (2024).
"Neighborhood repartition-based oversampling algorithm for multiclass
imbalanced data with label noise." Neurocomputing 600, 128090.
doi:10.1016/j.neucom.2024.128090

This module follows the PAPER (Algorithm 1 = B-NRO, Algorithm 2 = MC-NRO,
Eqs. 1-3), not the authors' public code, because the public code
(github.com/sshiyi77/MC-NRO) departs from the paper in several places.
Every place where the paper is ambiguous is marked "CHOICE" below, and the
differences from the public code are marked "CODE-DIFF".

Summary of CODE-DIFF (paper vs. public code)
  1. Distance: paper uses the L1 norm; code uses L2 everywhere
     (scipy distance_matrix is always L2).
  2. Radius: paper uses the distance to the k1-th majority neighbour;
     code uses argsort()[k1] - 0.01, i.e. the (k1+1)-th neighbour.
  3. First cleaning: with the default 'translate and remove' the code never
     applies the translations to the majority points (neither branch of the
     if/elif fires), so majority points stay where they were.
  4. Second cleaning: code matches moved points with np.in1d on flattened
     arrays and then deletes those *flat* indices as *row* indices, so it
     removes essentially arbitrary rows. It also uses majority_labels[0] as
     the label of every moved point, which is wrong in the multiclass case,
     and runs the cleaning inside every binary step instead of once after
     all classes have been oversampled (paper, Algorithm 2 lines 17-22).
  5. Potential: paper computes it w.r.t. the minority points inside the
     current neighbourhood; code uses all minority points.
  6. Repartition: paper uses R = phi0 + (max(U) - phi0)/n (Eq. 3); code uses
     fixed 0.33 thresholds borrowed from RBO/CCR ('D','S','E' regions),
     which corresponds to n = 3 with a slightly different max term.
  7. Candidate sampling: paper says uniform in the neighbourhood; code draws
     a normalised random cube direction times U(0,1)*r, which is NOT uniform
     in the ball (in 19 dimensions ~50% of candidates fall inside r/2,
     versus ~2e-6 for a uniform ball).
  8. IDF-weighted one-hot (datasets.py) is undone by the MinMaxScaler that is
     applied afterwards, so the weighting has no effect in their experiments.

Optional extension (NOT part of MC-NRO, off by default)
  `ordinal_cols` / `onehot_groups` project synthetic samples back to valid
  questionnaire values (round + clip ordinal items, arg-max for one-hot
  groups). This is the kind of "adaptive" step proposed for the MSME paper
  and must be reported as the authors' own modification.

Usage
-----
    from mcnro import MCNRO
    X_res, y_res = MCNRO(k1=5, k2=5, eps=0.5, n_partitions=3,
                         random_state=0).fit_resample(X_train, y_train)

Apply it to the TRAINING fold only (never before splitting).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

__all__ = ["MCNRO", "ANRO", "BNRO"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _cdist(A, B, p):
    if p == 1:
        return cdist(A, B, metric="cityblock")
    if p == 2:
        return cdist(A, B, metric="euclidean")
    return cdist(A, B, metric="minkowski", p=p)


def _uniform_in_ball(center, radius, size, p, rng):
    """Draw `size` points uniformly inside the Lp ball (p in {1, 2})."""
    m = center.shape[0]
    if p == 1:
        z = rng.laplace(size=(size, m))           # normalised -> uniform on L1 sphere
        norms = np.abs(z).sum(1, keepdims=True)
    elif p == 2:
        z = rng.standard_normal((size, m))        # normalised -> uniform on L2 sphere
        norms = np.sqrt((z ** 2).sum(1, keepdims=True))
    else:
        raise ValueError("uniform sampling implemented for p in {1, 2}")
    norms[norms == 0] = 1.0
    u = rng.random((size, 1)) ** (1.0 / m)        # radial part for uniform volume
    return center + z / norms * u * radius


def _potential(points, ref, eps, p):
    """Eq. 2: phi(x, C, eps) = sum_c exp(-(||c - x||_p / eps)^2)."""
    d = _cdist(np.atleast_2d(points), ref, p)
    return np.exp(-(d / eps) ** 2).sum(1)


def _log_potential(points, ref, eps, p):
    """log of Eq. 2 computed with log-sum-exp (used by A-NRO for ranking)."""
    from scipy.special import logsumexp
    d = _cdist(np.atleast_2d(points), ref, p)
    return logsumexp(-(d / eps) ** 2, axis=1)


def _allocate(radii, n_total):
    """Eq. 1: num_i proportional to 1/r_i.
    CHOICE: the paper floors each share, which may leave the class slightly
    below balance; the remainder is handed out by largest fractional part so
    the total equals n_total exactly (set exact=False in BNRO for pure floor)."""
    w = 1.0 / radii
    raw = w / w.sum() * n_total
    base = np.floor(raw).astype(int)
    return raw, base


# ----------------------------------------------------------------------------
# Algorithm 1 (Steps 1-2): binary NRO, used inside MC-NRO
# ----------------------------------------------------------------------------
class BNRO:
    """Binary NRO, Steps 1-2 of Algorithm 1 (Step 3 is done by MCNRO)."""

    def __init__(self, k1=5, eps=0.5, n_partitions=3, n_candidates=1000,
                 region="high", p=1, translate=True, exact=True,
                 fill="replace", repartition="paper", rng=None):
        assert region in ("high", "low")
        assert fill in ("replace", "strict")
        assert repartition in ("paper", "quantile")
        self.repartition = repartition
        self.k1, self.eps, self.n_partitions = k1, eps, n_partitions
        self.n_candidates, self.region, self.p = n_candidates, region, p
        self.translate, self.exact, self.fill = translate, exact, fill
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(self, X_maj, X_min, n_generate):
        """Returns (X_maj_after, moved_mask, X_syn, info)."""
        rng, p = self.rng, self.p
        X_maj = X_maj.astype(float).copy()
        n_min, n_maj = len(X_min), len(X_maj)
        k1 = int(min(self.k1, n_maj))                  # guard for tiny folds
        D = _cdist(X_min, X_maj, p)

        # ---- Step 1: neighbourhood radius = distance to k1-th majority NN
        radii = np.sort(D, axis=1)[:, k1 - 1].copy()
        tiny = 1e-9
        radii[radii < tiny] = tiny                     # CHOICE: duplicates across classes

        # ---- Step 1: move majority points that lie inside each neighbourhood
        #      (Alg. 1 lines 12-15):  t_j += (r_i - d_ij)/d_ij * (x_j - x_i)
        T = np.zeros_like(X_maj)
        for i in range(n_min):
            inside = np.where(D[i] < radii[i])[0]
            for j in inside:
                diff = X_maj[j] - X_min[i]
                d = D[i, j]
                if d < tiny:                           # coincident point: random direction
                    diff = rng.choice([-1.0, 1.0], X_maj.shape[1]) * 1e-6
                    d = np.abs(diff).sum() if p == 1 else np.linalg.norm(diff)
                T[j] += (radii[i] - d) / d * diff
        moved = np.any(T != 0, axis=1)
        if self.translate:
            X_maj = X_maj + T

        # ---- Step 2: allocation (Eq. 1)
        raw, num = _allocate(radii, n_generate)
        if self.exact:
            rest = int(n_generate - num.sum())
            if rest > 0:
                num[np.argsort(-(raw - num))[:rest]] += 1

        # ---- Step 2: repartition with the class potential (Eqs. 2-3)
        D_mm = _cdist(X_min, X_min, p)
        syn, n_short = [], 0
        for i in range(n_min):
            if num[i] <= 0:
                continue
            local = X_min[D_mm[i] <= radii[i]]          # min^i (contains x_i itself)
            phi0 = _potential(X_min[i], local, self.eps, p)[0]
            C = _uniform_in_ball(X_min[i], radii[i], self.n_candidates, p, rng)
            if self.repartition == "paper":
                U = _potential(C, local, self.eps, p)
                R = phi0 + (U.max() - phi0) / self.n_partitions      # Eq. 3
                keep = C[U >= R] if self.region == "high" else C[U < R]
            else:
                # A-NRO: rank candidates by log-potential (no underflow) and keep
                # the top (or bottom) 1/n fraction; the seed is not the reference,
                # so the region can never be empty
                logU = _log_potential(C, local, self.eps, p)
                q = 1.0 / self.n_partitions
                if self.region == "high":
                    keep = C[logU >= np.quantile(logU, 1 - q)]
                else:
                    keep = C[logU <= np.quantile(logU, q)]
            if len(keep) >= num[i]:
                syn.append(keep[rng.choice(len(keep), num[i], replace=False)])
            else:
                n_short += 1
                if self.fill == "replace":
                    # CHOICE (same as public code): pool = retained candidates + seed,
                    # drawn with replacement so the requested count is met
                    pool = np.vstack([keep, X_min[i][None]]) if len(keep) else X_min[i][None]
                    syn.append(pool[rng.choice(len(pool), num[i], replace=True)])
                elif len(keep):                        # 'strict': keep what exists
                    syn.append(keep)
        X_syn = np.vstack(syn) if syn else np.empty((0, X_min.shape[1]))
        info = dict(radii=radii, num=num, n_moved=int(moved.sum()),
                    n_short=n_short)
        return X_maj, moved, X_syn, info


# ----------------------------------------------------------------------------
# Algorithm 2: MC-NRO with OVS decomposition + final cleaning
# ----------------------------------------------------------------------------
class MCNRO:
    """Multiclass Neighbourhood Repartition-based Oversampling.

    Parameters (paper names in brackets)
    ----------
    k1 : neighbours that define the neighbourhood radius [k1]
    k2 : neighbours used in the final overlap cleaning [k2]
    eps : spread of the Gaussian RBF [epsilon]
    n_partitions : number of target regions in Eq. 3 [n]
    n_candidates : candidate samples per seed [iterations]
    region : 'high' (near the seed, Fig. 3a) or 'low' (near moved majority, Fig. 3b)
    p : Lp norm for distances (paper: 1)
    cleaning : 'translate_remove' (paper default), 'translate', or 'none'
    scale : min-max scale internally (as in the authors' experiments) and map
            results back to the original units
    ordinal_cols, onehot_groups : OPTIONAL extension (not MC-NRO), see module doc
    """

    def __init__(self, k1=5, k2=5, eps=0.5, n_partitions=3, n_candidates=1000,
                 region="high", p=1, cleaning="translate_remove", scale=True,
                 exact=True, fill="replace", ordinal_cols=None,
                 onehot_groups=None, repartition="paper", clean_rule="any",
                 project_moved=False, random_state=None):
        assert cleaning in ("translate_remove", "translate", "none")
        assert clean_rule in ("any", "majority")
        self.repartition, self.clean_rule = repartition, clean_rule
        self.project_moved = project_moved
        self.k1, self.k2, self.eps = k1, k2, eps
        self.n_partitions, self.n_candidates = n_partitions, n_candidates
        self.region, self.p, self.cleaning, self.scale = region, p, cleaning, scale
        self.exact, self.fill = exact, fill
        self.ordinal_cols, self.onehot_groups = ordinal_cols, onehot_groups
        self.random_state = random_state

    # sklearn / imbalanced-learn style
    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        lo, hi = X.min(0), X.max(0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        Z = (X - lo) / span if self.scale else X.copy()

        classes, counts = np.unique(y, return_counts=True)
        order = classes[np.argsort(-counts, kind="stable")]      # descending size
        n_max = counts.max()

        # pool of all observations; each row keeps label / synthetic / moved flags
        P = Z.copy()
        L = y.copy()
        syn_flag = np.zeros(len(P), bool)
        moved_flag = np.zeros(len(P), bool)
        self.log_ = []

        bnro = BNRO(k1=self.k1, eps=self.eps, n_partitions=self.n_partitions,
                    n_candidates=self.n_candidates, region=self.region, p=self.p,
                    translate=self.cleaning != "none", exact=self.exact,
                    fill=self.fill, repartition=self.repartition, rng=rng)

        for i in range(1, len(order)):
            c = order[i]
            min_idx = np.where(L == c)[0]
            n_gen = int(n_max - len(min_idx))
            if n_gen <= 0:
                continue
            # combined majority: floor(|C1| / n_c) random rows from each larger class
            # (Alg. 2 line 12), drawn from the CURRENT pool incl. earlier synthetics
            per = int(np.floor(n_max / i))
            maj_idx = []
            for j in range(i):
                cand = np.where(L == order[j])[0]
                maj_idx.append(rng.choice(cand, min(per, len(cand)), replace=False))
            maj_idx = np.concatenate(maj_idx)

            X_maj_new, moved, X_syn, info = bnro.run(P[maj_idx], P[min_idx], n_gen)
            P[maj_idx] = X_maj_new                     # moved points re-enter the pool
            moved_flag[maj_idx[moved]] = True
            if len(X_syn):
                P = np.vstack([P, X_syn])
                L = np.concatenate([L, np.full(len(X_syn), c, dtype=L.dtype)])
                syn_flag = np.concatenate([syn_flag, np.ones(len(X_syn), bool)])
                moved_flag = np.concatenate([moved_flag, np.zeros(len(X_syn), bool)])
            self.log_.append(dict(cls=c, n_min=len(min_idx), n_maj=len(maj_idx),
                                  n_generated=len(X_syn), **{k: v for k, v in info.items()
                                                            if k not in ("radii", "num")}))

        # ---- final cleaning (Alg. 2 lines 17-22): a moved majority point is removed
        #      if its k2 nearest neighbours in the whole sample space are not all
        #      of its own class
        removed = np.zeros(len(P), bool)
        if self.cleaning == "translate_remove" and moved_flag.any():
            mv = np.where(moved_flag)[0]
            D = _cdist(P[mv], P, self.p)
            D[np.arange(len(mv)), mv] = np.inf         # exclude the point itself
            k2 = int(min(self.k2, len(P) - 1))
            nn = np.argsort(D, axis=1, kind="stable")[:, :k2]
            disagree = (L[nn] != L[mv][:, None])
            if self.clean_rule == "any":                   # paper: any foreign neighbour
                bad = disagree.any(axis=1)
            else:                                          # A-NRO: majority vote (ENN-like)
                bad = disagree.sum(axis=1) > k2 / 2.0
            removed[mv[bad]] = True

        keep = ~removed
        P, L, syn_flag = P[keep], L[keep], syn_flag[keep]
        moved_kept = moved_flag[keep]
        Xo = P * span + lo if self.scale else P

        # ---- OPTIONAL extension: project synthetic rows to valid questionnaire values
        if self.ordinal_cols is not None or self.onehot_groups is not None:
            sel = syn_flag | (moved_kept if self.project_moved else False)
            S = Xo[sel]
            for col in (self.ordinal_cols or []):
                S[:, col] = np.clip(np.rint(S[:, col]), lo[col], hi[col])
            for grp in (self.onehot_groups or []):
                g = np.asarray(grp)
                hot = np.argmax(S[:, g], axis=1)
                S[:, g] = 0.0
                S[np.arange(len(S)), g[hot]] = 1.0
            Xo[sel] = S

        self.n_removed_ = int(removed.sum())
        self.n_moved_ = int(moved_flag.sum())
        self.synthetic_mask_ = syn_flag
        return Xo, L


class ANRO(MCNRO):
    """A-NRO: adaptive variant of MC-NRO proposed for small ordinal survey data
    (authors' own modification, NOT part of Shen et al., 2024).

    Differences from MCNRO (each can be switched off for ablation):
      repartition='quantile' : rank candidates by log-potential and keep the
                               top 1/n fraction, instead of Eq. 3 (which is
                               anchored at the seed and is empty whenever the
                               seed has the highest potential)
      clean_rule='majority'  : remove a moved point only if most of its k2
                               neighbours belong to other classes (paper: any)
      ordinal projection     : synthetic AND moved rows are mapped back to valid
                               questionnaire values (pass ordinal_cols/onehot_groups)
    """

    def __init__(self, repartition="quantile", clean_rule="majority",
                 project_moved=True, **kw):
        super().__init__(repartition=repartition, clean_rule=clean_rule,
                         project_moved=project_moved, **kw)
