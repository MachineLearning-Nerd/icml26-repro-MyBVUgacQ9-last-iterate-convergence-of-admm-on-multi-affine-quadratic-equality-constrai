"""Faithful implementation of the paper's problem class and Algorithm 1 (ADMM).

Paper: "Last-iterate Convergence of ADMM on Multi-affine Quadratic Equality
Constrained Problem", arXiv:2603.11919.

Problem (eq. 1)
---------------
    min_{x,z}  F(x) + phi(z)     s.t.  A(x) + Q z = 0

with (Assumption 2.3)  F(x) = f(x) + sum_i I_i(x_i), f in C^2 and mu_f-strongly
convex, I_i the indicator of a closed convex set X_i; phi in C^2, mu_phi-strongly
convex.  (Definition 2.1)  A(x)_j = 0.5 x^T C_j x + d_j^T x + e_j, and A is affine
in each block x_j when the remaining blocks are fixed -- equivalently, the
diagonal blocks of every C_j vanish.  (Assumption 2.6) Q in R^{n_c x n_z} has
full row rank.

Augmented Lagrangian (Definition 2.7, eq. 3)
    L(x,z,w) = F(x) + phi(z) + <w, A(x)+Qz> + (rho/2) ||A(x)+Qz||^2

Algorithm 1
    for i = 1..n:  x_i <- argmin_{x_i} L(x_1:i-1^{k+1}, x_i, x_{i+1:n}^k, z^k, w^k)
    z <- argmin_z L(x^{k+1}, z, w^k)
    w <- w + rho (A(x^{k+1}) + Q z^{k+1})

All block sub-problems are solved *exactly* (Algorithm 1, not the approximated
variant of Appendix D.1):
  * unconstrained blocks  -> a dense linear solve (the sub-problem is a strongly
    convex quadratic, so this is the exact minimiser);
  * blocks with a polyhedral / ball set -> Clarabel interior point at tolerance
    1e-13, with the KKT stationarity residual recorded so that the exactness
    assumption is audited rather than assumed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

try:  # optional at import time so that pure-numpy checks still work
    import clarabel
except Exception:  # pragma: no cover
    clarabel = None


# --------------------------------------------------------------------------
# block feasible sets X_i
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FreeSet:
    """X_i = R^{n_i} (I_i == 0). Trivially polyhedral."""

    n: int
    label: str = "free"

    def contains(self, v, tol=1e-9):
        return True

    def is_polyhedral(self) -> bool:
        return True


@dataclass(frozen=True)
class Polyhedron:
    """X_i = {v : G v <= h}. Boxes and friction pyramids use this form."""

    G: np.ndarray
    h: np.ndarray
    label: str = "poly"

    @property
    def n(self) -> int:
        return self.G.shape[1]

    def contains(self, v, tol=1e-9):
        return bool(np.all(self.G @ v <= self.h + tol))

    def slack(self, v):
        return self.h - self.G @ v

    def is_polyhedral(self) -> bool:
        return True


@dataclass(frozen=True)
class Ball:
    """X_i = {v : ||v - c|| <= r}: convex and closed but NOT polyhedral.

    The negative control for Theorem 3.3, whose hypothesis is that the {I_i} are
    indicators of polyhedra.
    """

    c: np.ndarray
    r: float
    label: str = "ball"

    @property
    def n(self) -> int:
        return self.c.size

    def contains(self, v, tol=1e-9):
        return bool(np.linalg.norm(v - self.c) <= self.r + tol)

    def slack(self, v):
        return np.array([self.r - np.linalg.norm(v - self.c)])

    def is_polyhedral(self) -> bool:
        return False


def box(lo, hi, n: int) -> Polyhedron:
    lo = np.broadcast_to(np.asarray(lo, float), (n,)).copy()
    hi = np.broadcast_to(np.asarray(hi, float), (n,)).copy()
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.concatenate([hi, -lo])
    return Polyhedron(G, h, label=f"box[{lo.min():g},{hi.max():g}]")


# --------------------------------------------------------------------------
# the problem
# --------------------------------------------------------------------------
@dataclass
class MAQEP:
    """A multi-affine quadratic equality constrained problem (eq. 1).

    f(x)   = 0.5 x^T P x + p^T x            (P symmetric, P >= mu_f I > 0)
    phi(z) = 0.5 z^T S z + s^T z            (S symmetric, S >= mu_phi I > 0)
    A(x)_j = 0.5 x^T C_j x + d_j^T x + e_j  (C_j symmetric, zero diagonal blocks)
    """

    P: np.ndarray
    p: np.ndarray
    S: np.ndarray
    s: np.ndarray
    Clist: list  # list of n_c sparse/dense symmetric (n_x, n_x) matrices
    d: np.ndarray  # (n_c, n_x)
    e: np.ndarray  # (n_c,)
    Q: np.ndarray  # (n_c, n_z)
    blocks: list  # index arrays partitioning range(n_x)
    sets: list  # one feasible set per block
    name: str = "maqep"

    def __post_init__(self):
        self.Clist = [sp.csr_matrix(Cj) for Cj in self.Clist]
        # stacked operator: (Cstack @ x).reshape(n_c, n_x)[j] == C_j @ x
        self._Cstack = sp.vstack(self.Clist, format="csr") if self.Clist else None
        self.blocks = [np.asarray(b, dtype=int) for b in self.blocks]
        self._Pbb = [self.P[np.ix_(b, b)] for b in self.blocks]

    # ---------- sizes ----------
    @property
    def n_x(self) -> int:
        return self.P.shape[0]

    @property
    def n_z(self) -> int:
        return self.S.shape[0]

    @property
    def n_c(self) -> int:
        return self.Q.shape[0]

    # ---------- objective pieces ----------
    def f(self, x):
        return float(0.5 * x @ self.P @ x + self.p @ x)

    def phi(self, z):
        return float(0.5 * z @ self.S @ z + self.s @ z)

    def _Cx(self, x):
        """(n_c, n_x) matrix whose j-th row is C_j x."""
        if self._Cstack is None:
            return np.zeros((0, self.n_x))
        return (self._Cstack @ x).reshape(self.n_c, self.n_x)

    def A(self, x, Cx=None):
        Cx = self._Cx(x) if Cx is None else Cx
        return 0.5 * (Cx @ x) + self.d @ x + self.e

    def residual(self, x, z):
        return self.A(x) + self.Q @ z

    def indicators_finite(self, x, tol=1e-7) -> bool:
        return all(S.contains(x[b], tol) for b, S in zip(self.blocks, self.sets))

    def objective(self, x, z) -> float:
        return self.f(x) + self.phi(z)

    def lagrangian(self, x, z, w, rho) -> float:
        r = self.residual(x, z)
        return float(self.f(x) + self.phi(z) + w @ r + 0.5 * rho * r @ r)

    def grad_L(self, x, z, w, rho):
        """Smooth part of grad L wrt (x, z, w) (indicators excluded)."""
        Cx = self._Cx(x)
        r = self.A(x, Cx) + self.Q @ z
        JA = Cx + self.d  # Jacobian of A at x: row j = C_j x + d_j
        gx = self.P @ x + self.p + JA.T @ (w + rho * r)
        gz = self.S @ z + self.s + self.Q.T @ (w + rho * r)
        return gx, gz, r

    # ---------- assumption audit (Assumptions 2.3 / 2.6, Definition 2.1) ----------
    def audit(self) -> dict:
        eigP = np.linalg.eigvalsh(0.5 * (self.P + self.P.T))
        eigS = np.linalg.eigvalsh(0.5 * (self.S + self.S.T))
        svQ = np.linalg.svd(self.Q, compute_uv=False) if self.n_c else np.array([])
        maxdiag, sym, normC = 0.0, 0.0, 0.0
        for Cj in self.Clist:
            D = Cj.toarray()
            normC = max(normC, float(np.linalg.norm(D, 2)))
            sym = max(sym, float(np.abs(D - D.T).max(initial=0.0)))
            for b in self.blocks:
                sub = D[np.ix_(b, b)]
                maxdiag = max(maxdiag, float(np.abs(sub).max(initial=0.0)))
        rankQ = int((svQ > (svQ.max() * 1e-12)).sum()) if svQ.size else 0
        return dict(
            n_x=self.n_x, n_z=self.n_z, n_c=self.n_c, n_blocks=len(self.blocks),
            mu_f=float(eigP.min()), L_f=float(eigP.max()),
            mu_phi=float(eigS.min()), L_phi=float(eigS.max()),
            sigma_min_Q=float(svQ.min()) if svQ.size else 0.0,
            rank_Q=rankQ,
            Q_full_row_rank=bool(rankQ == self.n_c),          # Assumption 2.6
            lam_min_QQt=float(svQ.min() ** 2) if svQ.size else 0.0,
            norm_pinvQ=float(np.linalg.norm(np.linalg.pinv(self.Q), 2)) if svQ.size else float("inf"),
            norm_C=normC,                                      # ||C|| = max_j ||C_j||
            norm_d=float(np.linalg.norm(self.d, 2)) if self.n_c else 0.0,
            norm_e=float(np.linalg.norm(self.e)) if self.n_c else 0.0,
            C_zero_diagonal_blocks=bool(maxdiag < 1e-12),      # Definition 2.1
            C_max_diag_block_entry=maxdiag,
            C_symmetric=bool(sym < 1e-12),
            f_strongly_convex=bool(eigP.min() > 0),            # Assumption 2.3
            phi_strongly_convex=bool(eigS.min() > 0),          # Assumption 2.3
            all_sets_polyhedral=bool(all(S.is_polyhedral() for S in self.sets)),
            set_labels=[getattr(S, "label", "free") for S in self.sets],
        )

    def rho_threshold(self) -> float:
        """The rho bound of Theorem 3.1:

            rho >= max{ 4 L_phi^2 / (mu_phi lam_min^+(Q^T Q)),
                        4 L_phi^2 / (mu_phi sqrt(lam_min^+(Q Q^T))) }

        For full-row-rank Q, lam_min^+(Q^T Q) = lam_min^+(Q Q^T) = sigma_min(Q)^2.
        """
        a = self.audit()
        lam = a["lam_min_QQt"]
        if lam <= 0:
            return float("inf")
        return float(max(4 * a["L_phi"] ** 2 / (a["mu_phi"] * lam),
                         4 * a["L_phi"] ** 2 / (a["mu_phi"] * np.sqrt(lam))))


# --------------------------------------------------------------------------
# exact block minimisation
# --------------------------------------------------------------------------
def _solve_qp(H, g, S, tol=1e-13):
    """Exact minimiser of 0.5 v^T H v + g^T v over S (H positive definite).

    Returns (v, kkt_stationarity_residual).
    """
    if isinstance(S, FreeSet):
        v = np.linalg.solve(H, -g)
        return v, float(np.linalg.norm(H @ v + g))

    if clarabel is None:  # pragma: no cover
        raise RuntimeError("clarabel is required for constrained block sub-problems")

    n = H.shape[0]
    P_ = sp.csc_matrix(np.triu(H))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = tol
    settings.tol_gap_rel = tol
    settings.tol_feas = tol
    settings.max_iter = 400

    if isinstance(S, Polyhedron):
        A_ = sp.csc_matrix(S.G)
        b_ = np.asarray(S.h, float)
        cones = [clarabel.NonnegativeConeT(b_.size)]
    elif isinstance(S, Ball):
        # ||v - c|| <= r  <=>  (r, v - c) in SOC
        A_ = sp.csc_matrix(np.vstack([np.zeros((1, n)), -np.eye(n)]))
        b_ = np.concatenate([[S.r], -np.asarray(S.c, float)])
        cones = [clarabel.SecondOrderConeT(n + 1)]
    else:  # pragma: no cover
        raise TypeError(f"unsupported set {S!r}")

    sol = clarabel.DefaultSolver(P_, np.asarray(g, float), A_, b_, cones, settings).solve()
    v = np.asarray(sol.x, float)
    lam = np.asarray(sol.z, float)
    kkt = float(np.linalg.norm(H @ v + g + A_.T @ lam))
    return v, kkt


@dataclass
class AdmmTrace:
    L: np.ndarray       # augmented Lagrangian per iteration
    viol: np.ndarray    # ||A(x)+Qz||
    obj: np.ndarray     # F(x)+phi(z)
    norm_w: np.ndarray
    x: np.ndarray
    z: np.ndarray
    w: np.ndarray
    max_kkt: float
    iters: int
    seconds: float
    rho: float
    X: np.ndarray | None = None   # full iterate history when store_iterates=True
    Z: np.ndarray | None = None
    W: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def admm(prob: MAQEP, rho: float, x0, z0=None, w0=None, K: int = 2000,
         subtol: float = 1e-13, store_iterates: bool = False) -> AdmmTrace:
    """Algorithm 1 with exact block minimisation."""
    t0 = time.perf_counter()
    x = np.array(x0, float)
    z = np.zeros(prob.n_z) if z0 is None else np.array(z0, float)
    w = np.zeros(prob.n_c) if w0 is None else np.array(w0, float)
    nx = prob.n_x

    Lh, Vh, Oh, Wh = [], [], [], []
    Xh, Zh, Wfull = ([], [], []) if store_iterates else (None, None, None)
    max_kkt = 0.0
    Hz = prob.S + rho * prob.Q.T @ prob.Q
    Hz_inv = np.linalg.inv(Hz)

    for _ in range(K):
        # ---- x blocks, sequentially (Algorithm 1 inner loop) ----
        for b, Sset, Pbb in zip(prob.blocks, prob.sets, prob._Pbb):
            Cx = prob._Cx(x)                     # rows C_j x
            M = Cx[:, b] + prob.d[:, b]          # A affine in x_b: A = M x_b + r
            Ax = 0.5 * (Cx @ x) + prob.d @ x + prob.e
            r = Ax - M @ x[b] + prob.Q @ z
            H = Pbb + rho * (M.T @ M)
            g = (prob.P @ x)[b] - Pbb @ x[b] + prob.p[b] + M.T @ (w + rho * r)
            v, kkt = _solve_qp(H, g, Sset, tol=subtol)
            max_kkt = max(max_kkt, kkt)
            x[b] = v

        # ---- z block ----
        Ax = prob.A(x)
        z = Hz_inv @ (-(prob.s + prob.Q.T @ w + rho * prob.Q.T @ Ax))

        # ---- dual ----
        res = prob.A(x) + prob.Q @ z
        w = w + rho * res

        Lh.append(prob.lagrangian(x, z, w, rho))
        Vh.append(float(np.linalg.norm(res)))
        Oh.append(prob.objective(x, z))
        Wh.append(float(np.linalg.norm(w)))
        if store_iterates:
            Xh.append(x.copy()); Zh.append(z.copy()); Wfull.append(w.copy())

    return AdmmTrace(
        L=np.asarray(Lh), viol=np.asarray(Vh), obj=np.asarray(Oh),
        norm_w=np.asarray(Wh), x=x, z=z, w=w, max_kkt=max_kkt, iters=K,
        seconds=time.perf_counter() - t0, rho=float(rho),
        X=np.asarray(Xh) if store_iterates else None,
        Z=np.asarray(Zh) if store_iterates else None,
        W=np.asarray(Wfull) if store_iterates else None,
    )
