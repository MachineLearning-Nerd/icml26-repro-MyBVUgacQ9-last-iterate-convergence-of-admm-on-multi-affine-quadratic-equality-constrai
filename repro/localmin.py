"""Direct test of "(x*,z*) is a local minimum of problem (1)".

Because Q has full row rank (Assumption 2.6), for *any* x the constraint
A(x) + Qz = 0 is solvable in z.  The feasible set therefore projects onto
{x : x_i in X_i} without restriction, and problem (1) is equivalent to the
reduced problem

    min_x  V(x) := f(x) + sum_i I_i(x_i) + min{ phi(z) : Qz = -A(x) } .

The inner minimisation is a strongly convex equality-constrained QP with a
closed-form solution, so V is cheap to evaluate exactly.  `(x*,z*)` is a local
minimum of (1) iff x* is a local minimum of V, which is tested here by sampling
feasible points in shells of increasing radius around x* and looking for any
strict decrease.  This is a genuine search for a counterexample, not a
restatement of a second-order condition.
"""

from __future__ import annotations

import numpy as np

from repro.core import MAQEP, FreeSet, Polyhedron, _solve_qp


class ReducedObjective:
    def __init__(self, prob: MAQEP):
        self.prob = prob
        S_inv = np.linalg.inv(prob.S)
        self.S_inv = S_inv
        self.M = np.linalg.inv(prob.Q @ S_inv @ prob.Q.T)   # (Q S^-1 Q^T)^-1
        self.QSs = prob.Q @ S_inv @ prob.s

    def z_of_x(self, x):
        b = -self.prob.A(x)
        nu = -self.M @ (b + self.QSs)
        return -self.S_inv @ (self.prob.s + self.prob.Q.T @ nu)

    def __call__(self, x) -> float:
        z = self.z_of_x(x)
        return self.prob.f(x) + self.prob.phi(z)


def _as_box(Sset):
    """Return (lo, hi) if this Polyhedron is exactly a box, else None.

    A row of G that is a signed unit vector is a bound on one coordinate; if
    every row is of that form the set is a box, whose Euclidean projection is
    coordinatewise clipping.
    """
    if not isinstance(Sset, Polyhedron):
        return None
    G, h, n = Sset.G, Sset.h, Sset.G.shape[1]
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    for g, hv in zip(G, h):
        nz = np.flatnonzero(g)
        if nz.size != 1 or not np.isclose(abs(g[nz[0]]), 1.0):
            return None
        i = int(nz[0])
        if g[i] > 0:
            hi[i] = min(hi[i], hv)          #  x_i <= hv
        else:
            lo[i] = max(lo[i], -hv)         # -x_i <= hv  <=>  x_i >= -hv
    return lo, hi


def project_onto_X(prob: MAQEP, x):
    """Blockwise Euclidean projection onto prod_i X_i (exact).

    Boxes are clipped in closed form.  Going through the interior-point QP for
    them is not merely slower, it is WRONG at the accuracy this test needs: an
    interior-point solve never returns a point exactly on a face, so projecting
    a limit point that sits on the boundary moves it ~1e-7 into the interior and
    inflates V(x*) by the same amount.  Sampled points then appear to beat it,
    and the test reports a phantom "strictly better feasible point" -- with an
    identical -9.85e-08 on four different instances, which is what gave this
    away.  Only genuinely non-box sets (e.g. friction pyramids) use the solver.
    """
    out = np.array(x, float)
    for b, Sset in zip(prob.blocks, prob.sets):
        if isinstance(Sset, FreeSet):
            continue
        bounds = _as_box(Sset)
        if bounds is not None:
            lo, hi = bounds
            out[b] = np.clip(out[b], lo, hi)
            continue
        n = b.size
        v, _ = _solve_qp(np.eye(n), -out[b], Sset, tol=1e-12)
        out[b] = v
    return out


def local_minimality_test(prob: MAQEP, x_star, radii=(1e-6, 1e-4, 1e-2, 1e-1),
                          n_samples: int = 200, seed: int = 20260726) -> dict:
    """Search for a feasible point strictly better than x* within each radius."""
    rng = np.random.default_rng(seed)
    V = ReducedObjective(prob)
    x_star = project_onto_X(prob, np.asarray(x_star, float))
    V0 = V(x_star)
    scale = max(float(np.linalg.norm(x_star)), 1.0)
    worst = []
    best_decrease = 0.0
    best_point_radius = None
    for r in radii:
        step = r * scale
        vals = []
        for _ in range(n_samples):
            u = rng.standard_normal(prob.n_x)
            u /= np.linalg.norm(u)
            xs = project_onto_X(prob, x_star + step * u)
            vals.append(V(xs) - V0)
        vals = np.asarray(vals)
        dec = float(vals.min())
        worst.append(dict(radius=float(r), abs_step=float(step),
                          min_delta_V=dec, mean_delta_V=float(vals.mean()),
                          n_strict_decrease=int((vals < -1e-10 * max(abs(V0), 1.0)).sum())))
        if dec < best_decrease:
            best_decrease = dec
            best_point_radius = float(r)
    tol = 1e-9 * max(abs(V0), 1.0)
    return dict(
        V_at_limit=float(V0),
        shells=worst,
        min_delta_V=float(best_decrease),
        counterexample_radius=best_point_radius,
        is_local_minimum=bool(best_decrease >= -tol),
        tolerance=float(tol),
        n_samples_per_shell=n_samples,
    )


def indicator_activity(prob: MAQEP, x) -> dict:
    """Which block indicators are active at x.

    L is second-order differentiable at the limit point exactly when no
    indicator is active there (an active polyhedral face makes L non-smooth),
    so this is the machine-checkable form of the Theorem 3.2 hypothesis and the
    discriminator between Theorem 3.2 and Theorem 3.3.
    """
    n_active, min_slack, total = 0, np.inf, 0
    for b, Sset in zip(prob.blocks, prob.sets):
        if isinstance(Sset, FreeSet):
            continue
        sl = np.asarray(Sset.slack(x[b]), float)
        total += sl.size
        n_active += int((sl <= 1e-8).sum())
        min_slack = min(min_slack, float(sl.min()))
    return dict(n_active_constraints=n_active, n_constraints=total,
                min_slack=(None if not np.isfinite(min_slack) else float(min_slack)),
                any_active=bool(n_active > 0),
                lagrangian_second_order_differentiable=bool(n_active == 0))
