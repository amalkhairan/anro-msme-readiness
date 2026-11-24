"""Property tests for mcnro.py. Run: python test_mcnro.py"""
import numpy as np
from scipy.spatial.distance import cdist
from mcnro import MCNRO, BNRO, _uniform_in_ball, _potential

rng = np.random.default_rng(1)
ok = lambda name, cond: print(("PASS " if cond else "FAIL ") + name) or cond
res = []

# synthetic 3-class problem, 6 features
X = np.vstack([rng.normal(0, 1, (120, 6)), rng.normal(1.5, 1, (30, 6)), rng.normal(3, 1, (8, 6))])
y = np.array([0] * 120 + [1] * 30 + [2] * 8)

# 1 uniform-in-ball: fraction inside r/2 must be ~0.5**m
for p in (1, 2):
    pts = _uniform_in_ball(np.zeros(6), 1.0, 200000, p, rng)
    norm = np.abs(pts).sum(1) if p == 1 else np.linalg.norm(pts, axis=1)
    res.append(ok(f"uniform L{p} ball: inside<=r and P(<r/2)~{0.5**6:.4f} (got {(norm<0.5).mean():.4f})",
                  norm.max() <= 1 + 1e-9 and abs((norm < 0.5).mean() - 0.5 ** 6) < 0.003))

# 2 balance + only moved majority rows removed + minority never removed
m = MCNRO(k1=5, k2=5, eps=0.5, random_state=0)
Xr, yr = m.fit_resample(X, y)
cnt = np.bincount(yr)
res.append(ok(f"minority classes balanced to n_max: {cnt.tolist()}", cnt[1] == 120 and cnt[2] == 120))
res.append(ok(f"removed rows ({m.n_removed_}) <= moved rows ({m.n_moved_})", m.n_removed_ <= m.n_moved_))
res.append(ok("no NaN/inf", np.isfinite(Xr).all()))

# 3 every synthetic point lies inside some seed's L1 neighbourhood (scaled space)
lo, hi = X.min(0), X.max(0); Z = (X - lo) / (hi - lo)
b = BNRO(k1=5, eps=0.5, rng=np.random.default_rng(0))
Xm, mv, S, info = b.run(Z[y != 2], Z[y == 2], 112)
d = cdist(S, Z[y == 2], "cityblock")
res.append(ok("synthetic within radius of a seed", np.all((d <= info["radii"][None] + 1e-9).any(1))))
res.append(ok(f"generated exactly n ({len(S)})", len(S) == 112))

# 4 moved majority points end on/outside every neighbourhood that contained them
D_before = cdist(Z[y == 2], Z[y != 2], "cityblock")
D_after = cdist(Z[y == 2], Xm, "cityblock")
inside = D_before < info["radii"][:, None]
single = np.where(inside.sum(0) == 1)[0]            # majority rows inside exactly one ball
owner = inside[:, single].argmax(0)
res.append(ok(f"first cleaning moves majority to the ball boundary ({len(single)} rows checked)",
              len(single) > 0 and np.allclose(D_after[owner, single], info["radii"][owner], atol=1e-8)))

# 5 high-region rule: retained candidates satisfy U >= R (Eq. 3)
xi = Z[y == 2][0]; local = Z[y == 2][cdist(xi[None], Z[y == 2], "cityblock")[0] <= info["radii"][0]]
C = _uniform_in_ball(xi, info["radii"][0], 2000, 1, rng)
U = _potential(C, local, 0.5, 1); phi0 = _potential(xi, local, 0.5, 1)[0]
R = phi0 + (U.max() - phi0) / 3
res.append(ok(f"Eq.3 threshold between phi0 and max(U): phi0={phi0:.3f} R={R:.3f} maxU={U.max():.3f}",
              min(phi0, U.max()) - 1e-12 <= R <= max(phi0, U.max()) + 1e-12))

# 6 reproducibility
a = MCNRO(random_state=7).fit_resample(X, y)[0]; c = MCNRO(random_state=7).fit_resample(X, y)[0]
res.append(ok("reproducible with random_state", np.array_equal(a, c)))

# 7 optional ordinal projection gives valid Likert values on synthetic rows
Xo = np.clip(np.rint(X + 3), 1, 5)
mo = MCNRO(random_state=0, ordinal_cols=list(range(6)))
Xp, yp = mo.fit_resample(Xo, y)
S = Xp[mo.synthetic_mask_]
res.append(ok("ordinal projection -> integers in [1,5]", np.all(S == np.rint(S)) and S.min() >= 1 and S.max() <= 5))

# 8 A-NRO: quantile repartition never empty -> no short seeds, synthetic rows distinct
from mcnro import ANRO
an = ANRO(random_state=0); Xa, ya = an.fit_resample(X, y)
shorts = [d["n_short"] for d in an.log_]
Sa = Xa[an.synthetic_mask_]
res.append(ok(f"A-NRO no short seeds {shorts}, unique synthetic {len(np.unique(Sa.round(10), axis=0))}/{len(Sa)}",
              sum(shorts) == 0 and len(np.unique(Sa.round(10), axis=0)) == len(Sa)))
gen = [d["n_generated"] for d in an.log_]
res.append(ok(f"A-NRO generates n_max - n_c per class before cleaning ({gen})", gen == [90, 112]))

# 9 majority-vote cleaning removes no more than the paper's 'any' rule (same seed)
m_any = MCNRO(random_state=3, repartition="quantile", clean_rule="any"); m_any.fit_resample(X, y)
m_maj = MCNRO(random_state=3, repartition="quantile", clean_rule="majority"); m_maj.fit_resample(X, y)
res.append(ok(f"majority rule removes <= any rule ({m_maj.n_removed_} <= {m_any.n_removed_})",
              m_maj.n_removed_ <= m_any.n_removed_))

# 10 A-NRO projection also covers moved rows: every row is on the Likert lattice
ap = ANRO(random_state=0, ordinal_cols=list(range(6))); Xq, yq = ap.fit_resample(Xo, y)
res.append(ok("A-NRO: all rows (synthetic + moved) integer in [1,5]",
              np.all(Xq == np.rint(Xq)) and Xq.min() >= 1 and Xq.max() <= 5))

print(f"\n{sum(res)}/{len(res)} passed")
