"""The three baselines of Figure 4, and the Appendix-B problems they run on.

DEVIATION, STATED UP FRONT
--------------------------
The paper compares against PADMM (Yashtini, 2021), IPDS-ADMM (Yuan, 2025) and
IADMM (Tang & Toh, 2024).  Those three sources were **not** retrievable through
the literature tooling available to this reproduction, so the algorithms below
are reimplementations of the *standard published form* of each named method, not
transcriptions of the original pseudocode:

  PADMM      proximal ADMM: the exact block sweep of Algorithm 1 with an added
             proximal term (beta_i/2)||x_i - x_i^k||^2 on every block.
  IADMM      inexact ADMM: each block update is replaced by a single projected
             gradient step on the augmented Lagrangian (the subproblem is solved
             inexactly rather than exactly).
  IPDS-ADMM  inexact primal-dual smoothing ADMM: linearised primal steps with a
             damped dual step size and an increasing penalty schedule
             rho_k = rho_0 (1+k)^{1/3}.

What this does and does not license: the comparison below is a faithful
comparison of *exact-block-minimisation ADMM against linearised/proximal ADMM
variants* on the paper's own Appendix-B problems.  It is evidence about the
mechanism the paper appeals to -- that methods which do not solve the block
subproblems exactly lose the advantage the multi-affine structure confers.  It
is not a certified reproduction of those three specific papers' algorithms, and
Claim 5 is scored accordingly.

The three Appendix-B problems (line 1977 of the source text):
  B1  min mu_x/2 sum_{i=1}^4 x_i^2 + mu_z/2 z^2  s.t. x1x2 - x3x4 + 1.5z + 1 = 0
  B2  min mu_x/2 (x1^2+x2^2)       + mu_z/2 z^2  s.t. x1 + x2 + z + 1 = 0
  B3  min mu_x/2 (x1^2 + 4 sin^2 x1 + x2^2) + mu_z/2 z^2 s.t. x1+x2+z+1 = 0

B3's objective is neither quadratic nor convex, so it violates Assumption 2.3;
the paper uses it only for this empirical comparison.  It is implemented
directly rather than through `MAQEP`.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# problem definitions (small, explicit, shared by all four algorithms)
# --------------------------------------------------------------------------
class AppendixProblem:
    """min f(x) + phi(z) s.t. a(x) + q z + const = 0, with x split into blocks."""

    def __init__(self, name, n_x, blocks, f, grad_f, a, grad_a, q, mu_z,
                 convex_objective, multiaffine):
        self.name, self.n_x, self.blocks = name, n_x, blocks
        self.f, self.grad_f, self.a, self.grad_a = f, grad_f, a, grad_a
        self.q, self.mu_z = float(q), float(mu_z)
        self.convex_objective, self.multiaffine = convex_objective, multiaffine

    def phi(self, z):
        return 0.5 * self.mu_z * z * z

    def residual(self, x, z):
        return self.a(x) + self.q * z

    def lagrangian(self, x, z, w, rho):
        r = self.residual(x, z)
        return float(self.f(x) + self.phi(z) + w * r + 0.5 * rho * r * r)

    def objective(self, x, z):
        return float(self.f(x) + self.phi(z))


def problem_b1(mu_x=1.0, mu_z=1.0) -> AppendixProblem:
    return AppendixProblem(
        "B1_convex_multiaffine", 4, [[0], [1], [2], [3]],
        f=lambda x: 0.5 * mu_x * float(x @ x),
        grad_f=lambda x: mu_x * x,
        a=lambda x: float(x[0] * x[1] - x[2] * x[3] + 1.0),
        grad_a=lambda x: np.array([x[1], x[0], -x[3], -x[2]]),
        q=1.5, mu_z=mu_z, convex_objective=True, multiaffine=True)


def problem_b2(mu_x=1.0, mu_z=1.0) -> AppendixProblem:
    return AppendixProblem(
        "B2_convex_linear", 2, [[0], [1]],
        f=lambda x: 0.5 * mu_x * float(x @ x),
        grad_f=lambda x: mu_x * x,
        a=lambda x: float(x[0] + x[1] + 1.0),
        grad_a=lambda x: np.array([1.0, 1.0]),
        q=1.0, mu_z=mu_z, convex_objective=True, multiaffine=False)


def problem_b3(mu_x=1.0, mu_z=1.0) -> AppendixProblem:
    return AppendixProblem(
        "B3_nonconvex_linear", 2, [[0], [1]],
        f=lambda x: 0.5 * mu_x * (x[0] ** 2 + 4 * np.sin(x[0]) ** 2 + x[1] ** 2),
        grad_f=lambda x: mu_x * np.array([x[0] + 4 * np.sin(x[0]) * np.cos(x[0]), x[1]]),
        a=lambda x: float(x[0] + x[1] + 1.0),
        grad_a=lambda x: np.array([1.0, 1.0]),
        q=1.0, mu_z=mu_z, convex_objective=False, multiaffine=False)


APPENDIX_PROBLEMS = {"B1": problem_b1, "B2": problem_b2, "B3": problem_b3}


# --------------------------------------------------------------------------
# the four algorithms
# --------------------------------------------------------------------------
def _z_update(prob, x, w, rho):
    """Exact argmin_z L: (mu_z + rho q^2) z = -(q w + rho q a(x))."""
    return -(prob.q * w + rho * prob.q * prob.a(x)) / (prob.mu_z + rho * prob.q ** 2)


def _block_exact(prob, x, i, w, rho, mu_x_hess, prox_beta=0.0):
    """Exact minimiser of L over block i for the quadratic-objective problems.

    a(x) is affine in x_i with a(x) = m x_i + c, so the sub-problem is a scalar
    (or small) strongly convex quadratic.
    """
    b = prob.blocks[i]
    ga = prob.grad_a(x)
    m = ga[b]
    c = prob.a(x) - float(m @ x[b])
    H = mu_x_hess * np.eye(len(b)) + rho * np.outer(m, m) + prox_beta * np.eye(len(b))
    g = (prob.grad_f(x)[b] - mu_x_hess * x[b]) + m * (w + rho * c) - prox_beta * x[b]
    return np.linalg.solve(H, -g)


def run_algorithm(prob: AppendixProblem, algo: str, x0, rho: float, K: int,
                  mu_x: float = 1.0, seed: int = 0) -> dict:
    """Run one of {admm, padmm, iadmm, ipds_admm} and return per-iteration traces."""
    x = np.array(x0, float)
    z, w = 0.0, 0.0
    quadratic = prob.name != "B3_nonconvex_linear"
    viol, obj, lag = [], [], []
    rho0 = rho

    for k in range(K):
        rho_k = rho0 * (1.0 + k) ** (1.0 / 3.0) if algo == "ipds_admm" else rho0

        for i in range(len(prob.blocks)):
            b = prob.blocks[i]
            if algo == "admm":
                if quadratic:
                    x[b] = _block_exact(prob, x, i, w, rho_k, mu_x)
                else:
                    x[b] = _newton_block(prob, x, i, w, rho_k)
            elif algo == "padmm":
                # proximal ADMM: exact block minimisation of L + (beta/2)||x_i-x_i^k||^2
                ga = prob.grad_a(x)
                beta = rho_k * float(ga[b] @ ga[b])
                if quadratic:
                    x[b] = _block_exact(prob, x, i, w, rho_k, mu_x, prox_beta=beta)
                else:
                    x[b] = _newton_block(prob, x, i, w, rho_k, prox_beta=beta,
                                         x_prev=x[b].copy())
            elif algo in ("iadmm", "ipds_admm"):
                # single (projected) gradient step on L: the sub-problem is solved
                # inexactly rather than exactly
                r = prob.residual(x, z)
                grad = prob.grad_f(x)[b] + prob.grad_a(x)[b] * (w + rho_k * r)
                ga = prob.grad_a(x)
                Lloc = mu_x + rho_k * float(ga[b] @ ga[b]) + 1e-12
                x[b] = x[b] - grad / Lloc
            else:
                raise ValueError(algo)

        z = _z_update(prob, x, w, rho_k)
        r = prob.residual(x, z)
        # IPDS-ADMM damps the dual step; the others use the full step
        step = rho_k / (1.0 + k) ** (1.0 / 3.0) if algo == "ipds_admm" else rho_k
        w = w + step * r

        viol.append(abs(prob.residual(x, z)))
        obj.append(prob.objective(x, z))
        lag.append(prob.lagrangian(x, z, w, rho_k))

    return dict(algo=algo, problem=prob.name, viol=np.asarray(viol),
                obj=np.asarray(obj), lag=np.asarray(lag), x=x, z=z, w=w)


def _newton_block(prob, x, i, w, rho, prox_beta=0.0, x_prev=None, iters=60):
    """Exact block minimisation for the non-quadratic B3 objective, by damped
    Newton to machine precision (so 'exact' really is exact for ADMM/PADMM)."""
    b = prob.blocks[i]
    v = x[b].copy()
    xv = x.copy()
    for _ in range(iters):
        xv[b] = v
        r = prob.a(xv) + prob.q * _z_update_frozen(prob, xv, w, rho)
        ga = prob.grad_a(xv)[b]
        g = prob.grad_f(xv)[b] + ga * (w + rho * r)
        if prox_beta:
            g = g + prox_beta * (v - x_prev)
        # Hessian of the smooth part (diagonal for these problems)
        if prob.name == "B3_nonconvex_linear" and b[0] == 0:
            h = 1.0 + 8.0 * np.cos(2.0 * v[0])
        else:
            h = 1.0
        H = max(float(h), 1e-6) + rho * float(ga @ ga) + prox_beta
        step = -g / H
        v = v + step
        if float(np.linalg.norm(step)) < 1e-15:
            break
    return v


def _z_update_frozen(prob, x, w, rho):
    return _z_update(prob, x, w, rho)
