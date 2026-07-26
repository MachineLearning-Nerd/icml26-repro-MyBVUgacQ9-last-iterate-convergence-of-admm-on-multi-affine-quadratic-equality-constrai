"""Gap sequences, limit-point audits and the Theorem-3.2 bound of Equation (4).

Everything here is computed from *a priori* problem data or from an independently
converged reference run -- never from the convergence rate that is being tested.
"""

from __future__ import annotations

import numpy as np

from repro.core import MAQEP, AdmmTrace, Ball, FreeSet, Polyhedron, _solve_qp, admm
from repro.rates import classify


def gap_sequence(prob: MAQEP, rho: float, x0, K: int, ref_mult: int = 4,
                 z0=None, w0=None):
    """Run ADMM for ref_mult*K iterations, use the tail as the reference value
    L* and return the gap sequence over the first K iterations.

    The reference is taken from iterations the analysis window never touches, so
    the value of L* is not fitted to the decay it is used to measure.
    """
    tr = admm(prob, rho, x0, z0=z0, w0=w0, K=ref_mult * K)
    # L* is read off iterations the analysis window never touches, so the
    # reference is not fitted to the decay it is used to measure.  The ADMM
    # Lagrangian is not monotone, so the gap is taken in absolute value: the
    # theorems assert L^k - L* -> 0 at a stated rate, in either direction.
    L_ref = float(np.median(tr.L[int(0.8 * ref_mult * K):]))
    gap = np.abs(tr.L[:K] - L_ref)
    return gap, L_ref, tr


def limit_point_audit(prob: MAQEP, tr: AdmmTrace, rho: float) -> dict:
    """Theorem 3.1's characterisation of the limit point.

    For every block i,   x*_i in argmin_{x_i} f(x_i, x*_-i) + I_i(x_i)
                              s.t. A(x_i, x*_-i) + Q z* = 0,
    and                  z*   in argmin_z phi(z) s.t. A(x*) + Q z = 0.

    Because A is affine in each single block, each of these is a strongly convex
    QP with affine equality constraints and can be solved exactly and
    independently of ADMM.  We report the distance from the ADMM limit point.
    """
    x, z = tr.x.copy(), tr.z.copy()
    res_star = prob.residual(x, z)   # achieved feasibility level at the limit
    devs = []
    for b, Sset in zip(prob.blocks, prob.sets):
        Cx = prob._Cx(x)
        M = Cx[:, b] + prob.d[:, b]
        Ax = prob.A(x, Cx)
        r = Ax - M @ x[b] + prob.Q @ z          # A(x_i, x*_-i) + Qz* = M x_i + r
        # min 0.5 x_i^T P_bb x_i + g^T x_i  s.t. M x_i + r = res*, x_i in X_i.
        # Constraining to the *achieved* residual rather than to 0 keeps the
        # problem feasible at finite precision; x*_i is feasible by construction,
        # so what is tested is exactly the claim's content -- its optimality.
        Pbb = prob.P[np.ix_(b, b)]
        g = (prob.P @ x)[b] - Pbb @ x[b] + prob.p[b]
        v = _solve_eq_qp(Pbb, g, M, res_star - r, Sset)
        devs.append(float(np.linalg.norm(v - x[b])) if v is not None else np.nan)
    # z*: min phi(z) s.t. A(x*) + Qz = res*
    Ax = prob.A(x)
    vz = _solve_eq_qp(prob.S, prob.s, prob.Q, res_star - Ax, FreeSet(prob.n_z))
    dz = float(np.linalg.norm(vz - z)) if vz is not None else np.nan
    return dict(
        max_block_deviation=float(np.nanmax(devs)) if devs else 0.0,
        block_deviations=[float(v) for v in devs],
        z_deviation=dz,
        final_violation=float(tr.viol[-1]),
        n_blocks_checked=len(devs),
        any_unsolved=bool(np.isnan(devs).any() or np.isnan(dz)),
    )


def _solve_eq_qp(H, g, Aeq, beq, Sset, tol=1e-12):
    """min 0.5 v^T H v + g^T v  s.t. Aeq v = beq, v in Sset (exact)."""
    import clarabel
    import scipy.sparse as sp

    n = H.shape[0]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol
    settings.tol_feas = tol
    rows = [sp.csc_matrix(np.atleast_2d(Aeq))]
    rhs = [np.atleast_1d(np.asarray(beq, float))]
    cones = [clarabel.ZeroConeT(rhs[0].size)]
    if isinstance(Sset, Polyhedron):
        rows.append(sp.csc_matrix(Sset.G))
        rhs.append(np.asarray(Sset.h, float))
        cones.append(clarabel.NonnegativeConeT(Sset.h.size))
    elif isinstance(Sset, Ball):
        rows.append(sp.csc_matrix(np.vstack([np.zeros((1, n)), -np.eye(n)])))
        rhs.append(np.concatenate([[Sset.r], -np.asarray(Sset.c, float)]))
        cones.append(clarabel.SecondOrderConeT(n + 1))
    A_ = sp.vstack(rows, format="csc")
    b_ = np.concatenate(rhs)
    sol = clarabel.DefaultSolver(sp.csc_matrix(np.triu(H)), np.asarray(g, float),
                                 A_, b_, cones, settings).solve()
    if str(sol.status) not in ("Solved", "SolverStatus.Solved", "AlmostSolved",
                               "SolverStatus.AlmostSolved"):
        return None
    return np.asarray(sol.x, float)


# --------------------------------------------------------------------------
# Equation (4): the Theorem 3.2 nonlinearity bound
# --------------------------------------------------------------------------
def eq4_certificate(prob: MAQEP, x0, x_star_f=None, z_star_phi=None) -> dict:
    """Evaluate the paper's own sufficient condition for linear convergence.

    Appendix D (the derivation preceding Theorem D.1) shows that the role of
    Equation (4) is to force the reduced Hessian of the Lagrangian at the limit
    point to be positive definite:

        nabla^2 f(x*) + sum_i w*_i C_i  >  0 ,

    for which it suffices that  ||sum_i w*_i C_i|| < mu_f.  The same appendix
    bounds the left-hand side using only *a priori* data (eqs. 29-31):

        ||sum_i w*_i C_i||
          <= sqrt(n_c) ||C|| ||(QQ^T)^{-1}Q|| L_phi
             [ sqrt(L_f/mu_phi)(||x*_f|| + ||x0||)
               + sqrt(L_phi/mu_phi)( ||z*_phi||
                   + sqrt(n_c/lam_min(QQ^T)) (||x0||^2 ||C|| + ||x0|| ||d|| + ||e||) ) ]

    `eq4_margin` = mu_f / bound.  The certificate holds when eq4_margin > 1, i.e.
    when the a-priori bound already guarantees the reduced Hessian is positive
    definite.  Nothing here uses the observed convergence rate.
    """
    a = prob.audit()
    x0 = np.asarray(x0, float)
    xf = np.linalg.solve(prob.P, -prob.p) if x_star_f is None else np.asarray(x_star_f, float)
    zf = np.linalg.solve(prob.S, -prob.s) if z_star_phi is None else np.asarray(z_star_phi, float)
    nc, nC = a["n_c"], a["norm_C"]
    lam = a["lam_min_QQt"]
    inner = (np.sqrt(a["L_f"] / a["mu_phi"]) * (np.linalg.norm(xf) + np.linalg.norm(x0))
             + np.sqrt(a["L_phi"] / a["mu_phi"]) *
             (np.linalg.norm(zf)
              + np.sqrt(nc / lam) * (np.linalg.norm(x0) ** 2 * nC
                                     + np.linalg.norm(x0) * a["norm_d"] + a["norm_e"])))
    bound = np.sqrt(nc) * nC * a["norm_pinvQ"] * a["L_phi"] * inner
    return dict(
        norm_C=nC, mu_f=a["mu_f"], n_c=nc, lam_min_QQt=lam,
        norm_pinvQ=a["norm_pinvQ"], L_f=a["L_f"], L_phi=a["L_phi"], mu_phi=a["mu_phi"],
        apriori_bound_on_sum_w_C=float(bound),
        eq4_margin=float(a["mu_f"] / bound) if bound > 0 else float("inf"),
        eq4_certificate_holds=bool(bound < a["mu_f"]),
    )


def reduced_hessian_check(prob: MAQEP, tr: AdmmTrace) -> dict:
    """The *posterior* form of the same condition, evaluated at the ADMM limit
    point: lam_min( nabla^2 f(x*) + sum_i w*_i C_i ) and lam_min(nabla^2 phi)."""
    Hx = prob.P.copy()
    for wi, Cj in zip(tr.w, prob.Clist):
        Hx = Hx + wi * Cj.toarray()
    ev = np.linalg.eigvalsh(0.5 * (Hx + Hx.T))
    evz = np.linalg.eigvalsh(0.5 * (prob.S + prob.S.T))
    return dict(
        lam_min_reduced_hessian_x=float(ev.min()),
        lam_min_hessian_z=float(evz.min()),
        norm_sum_w_C=float(np.linalg.norm(Hx - prob.P, 2)),
        norm_w_star=float(np.linalg.norm(tr.w)),
        second_order_sufficient=bool(ev.min() > 0 and evz.min() > 0),
    )


