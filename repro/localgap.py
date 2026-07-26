"""The exact quantity appearing in Theorem 3.3.

    G_k := L(x^k, z^k, w^k) - min_{(x,z) in B(x^k,z^k;r)} L(x, z, w^k)

Note this is *not* the Lagrangian gap L^k - L* of Theorems 3.1/3.2: the dual
variable is frozen at w^k and the comparison is against the best point in a ball
of fixed radius r around the current iterate.  Points outside prod_i X_i have
L = +inf (the indicators are part of F), so the inner problem is

    min  f(x) + phi(z) + <w^k, A(x)+Qz> + (rho/2)||A(x)+Qz||^2
    s.t. ||(x,z) - (x^k,z^k)|| <= r,   x_i in X_i for every block i.

The inner problem is non-convex (A is quadratic), so it is solved numerically by
SLSQP from several starting points inside the ball.  Finding a *lower* inner
minimum makes G_k *larger*, i.e. makes a claim of geometric decay harder to
satisfy -- so multi-start is the conservative direction here, and the reported
G_k is a lower bound on the true value of the theorem's quantity.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from repro.core import MAQEP, FreeSet, Polyhedron


def _pack(x, z):
    return np.concatenate([x, z])


def local_gap(prob: MAQEP, xk, zk, wk, rho: float, r: float,
              n_starts: int = 3, seed: int = 0, maxiter: int = 200) -> dict:
    """Evaluate G_k at one iterate."""
    rng = np.random.default_rng(seed)
    nx, nz = prob.n_x, prob.n_z
    v0 = _pack(np.asarray(xk, float), np.asarray(zk, float))
    L_at_iterate = prob.lagrangian(xk, zk, wk, rho)

    def fun(v):
        x, z = v[:nx], v[nx:]
        return prob.lagrangian(x, z, wk, rho)

    def jac(v):
        x, z = v[:nx], v[nx:]
        gx, gz, _ = prob.grad_L(x, z, wk, rho)
        return np.concatenate([gx, gz])

    cons = [dict(type="ineq",
                 fun=lambda v, v0=v0, r=r: r ** 2 - float((v - v0) @ (v - v0)),
                 jac=lambda v, v0=v0: -2.0 * (v - v0))]
    for b, Sset in zip(prob.blocks, prob.sets):
        if isinstance(Sset, FreeSet):
            continue
        if not isinstance(Sset, Polyhedron):
            raise TypeError("local_gap supports polyhedral (or free) block sets")
        G = np.zeros((Sset.G.shape[0], nx + nz))
        G[:, b] = Sset.G
        h = np.asarray(Sset.h, float)
        cons.append(dict(type="ineq",
                         fun=lambda v, G=G, h=h: h - G @ v,
                         jac=lambda v, G=G: -G))

    best = L_at_iterate
    for s in range(n_starts):
        if s == 0:
            v_start = v0.copy()
        else:
            u = rng.standard_normal(nx + nz)
            u /= np.linalg.norm(u)
            v_start = v0 + 0.5 * r * u * rng.random()
        res = minimize(fun, v_start, jac=jac, constraints=cons, method="SLSQP",
                       options=dict(maxiter=maxiter, ftol=1e-14))
        if res.x is not None:
            v = np.asarray(res.x, float)
            # accept only points that genuinely satisfy the ball + set constraints
            if np.linalg.norm(v - v0) <= r * (1 + 1e-6):
                x, z = v[:nx], v[nx:]
                if prob.indicators_finite(x, tol=1e-7):
                    best = min(best, prob.lagrangian(x, z, wk, rho))

    return dict(L_at_iterate=float(L_at_iterate), inner_min=float(best),
                local_gap=float(L_at_iterate - best), radius=float(r))


def local_gap_sequence(prob: MAQEP, tr, rho: float, r: float, ks: np.ndarray,
                       n_starts: int = 3) -> dict:
    """G_k over a set of iteration indices (0-based into the stored trace)."""
    assert tr.X is not None, "trace must be recorded with store_iterates=True"
    gaps, details = [], []
    for i, k in enumerate(ks):
        d = local_gap(prob, tr.X[k], tr.Z[k], tr.W[k], rho, r,
                      n_starts=n_starts, seed=int(k))
        gaps.append(d["local_gap"])
        details.append(dict(k=int(k) + 1, **d))
    return dict(ks=[int(k) + 1 for k in ks], gaps=gaps, details=details)
