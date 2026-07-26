"""The paper's Section 5 toy problem, the Appendix B comparison problems, and a
randomised ensemble of multi-affine quadratic equality constrained problems.

Section 5 (Figure 2):
    min  mu_x/2 (x1^2+x2^2+x3^2+x4^2) + mu_z/2 z^2
    s.t. x1 x2 - x3 x4 + q z + 1 = 0

Appendix B (Figure 4):
    B1  min mu_x/2 (x1^2+..+x4^2) + mu_z/2 z^2   s.t. x1x2 - x3x4 + 1.5 z + 1 = 0
    B2  min mu_x/2 (x1^2+x2^2)     + mu_z/2 z^2   s.t. x1 + x2 + z + 1 = 0
    B3  min mu_x/2 (x1^2+4 sin^2 x1 + x2^2) + mu_z/2 z^2  s.t. x1 + x2 + z + 1 = 0
        (B3 has a non-convex objective and therefore violates Assumption 2.3;
         the paper uses it only for the empirical comparison of Figure 4.)
"""

from __future__ import annotations

import numpy as np

from repro.core import MAQEP, Ball, FreeSet, box


def toy_q(q: float, mu_x: float = 1.0, mu_z: float = 1.0, blockset=None) -> MAQEP:
    """Section 5 toy problem with nonlinearity parameter q (Figure 2)."""
    C = np.zeros((4, 4))
    C[0, 1] = C[1, 0] = 1.0     # + x1 x2
    C[2, 3] = C[3, 2] = -1.0    # - x3 x4
    sets = [FreeSet(1) for _ in range(4)] if blockset is None else list(blockset)
    return MAQEP(
        P=mu_x * np.eye(4), p=np.zeros(4),
        S=np.array([[mu_z]]), s=np.zeros(1),
        Clist=[C], d=np.zeros((1, 4)), e=np.array([1.0]),
        Q=np.array([[float(q)]]),
        blocks=[np.array([0]), np.array([1]), np.array([2]), np.array([3])],
        sets=sets, name=f"toy(q={q:g})",
    )


def toy_q_boxed(q: float, lo: float, hi: float, mu_x: float = 1.0, mu_z: float = 1.0) -> MAQEP:
    """Section 5 toy problem with *polyhedral* (box) block indicators."""
    p = toy_q(q, mu_x, mu_z, blockset=[box(lo, hi, 1) for _ in range(4)])
    p.name = f"toy(q={q:g},box[{lo:g},{hi:g}])"
    return p


def toy_q_ball(q: float, radius: float, mu_x: float = 1.0, mu_z: float = 1.0) -> MAQEP:
    """Same problem with *non-polyhedral* (ball) indicators - the Theorem 3.3
    negative control."""
    sets = [Ball(np.zeros(1), radius) for _ in range(4)]
    p = toy_q(q, mu_x, mu_z, blockset=sets)
    p.name = f"toy(q={q:g},ball(r={radius:g}))"
    return p


def appendix_b1(mu_x: float = 1.0, mu_z: float = 1.0) -> MAQEP:
    """Convex objective with multi-affine constraint (Figure 4 left)."""
    p = toy_q(1.5, mu_x, mu_z)
    p.name = "B1_convex_multiaffine"
    return p


def appendix_b2(mu_x: float = 1.0, mu_z: float = 1.0) -> MAQEP:
    """Convex objective with linear constraint (Figure 4 centre)."""
    return MAQEP(
        P=mu_x * np.eye(2), p=np.zeros(2),
        S=np.array([[mu_z]]), s=np.zeros(1),
        Clist=[np.zeros((2, 2))], d=np.array([[1.0, 1.0]]), e=np.array([1.0]),
        Q=np.array([[1.0]]),
        blocks=[np.array([0]), np.array([1])],
        sets=[FreeSet(1), FreeSet(1)],
        name="B2_convex_linear",
    )


# Appendix B problem 3 has a non-quadratic, non-convex objective and is handled
# by the dedicated Figure-4 comparison module (repro/problems/baselines.py).
B3_SPEC = dict(
    name="B3_nonconvex_linear",
    f=lambda x, mu_x: 0.5 * mu_x * (x[0] ** 2 + 4 * np.sin(x[0]) ** 2 + x[1] ** 2),
    grad_f=lambda x, mu_x: np.array([mu_x * (x[0] + 4 * np.sin(x[0]) * np.cos(x[0])),
                                     mu_x * x[1]]),
)


# --------------------------------------------------------------------------
# randomised ensemble (scale)
# --------------------------------------------------------------------------
def random_maqep(n_blocks: int, block_size: int, n_c: int, n_z: int,
                 c_scale: float = 1.0, seed: int = 0, density: float = 0.2,
                 mu_f: float = 1.0, mu_phi: float = 1.0,
                 with_box: bool = False, box_halfwidth: float = 3.0) -> MAQEP:
    """A random instance of eq. (1) satisfying Assumptions 2.3 and 2.6.

    Every C_j is built block-off-diagonal by construction (Definition 2.1) and
    symmetrised; Q is drawn until it has full row rank (Assumption 2.6).
    """
    rng = np.random.default_rng(seed)
    n_x = n_blocks * block_size
    blocks = [np.arange(i * block_size, (i + 1) * block_size) for i in range(n_blocks)]

    Clist = []
    for _ in range(n_c):
        Craw = np.zeros((n_x, n_x))
        for bi in range(n_blocks):
            for bj in range(bi + 1, n_blocks):
                mask = rng.random((block_size, block_size)) < density
                vals = rng.standard_normal((block_size, block_size)) * mask
                Craw[np.ix_(blocks[bi], blocks[bj])] = vals
        Cs = Craw + Craw.T
        nrm = np.linalg.norm(Cs, 2)
        if nrm > 0:
            Cs = Cs / nrm * c_scale          # exact control of ||C_j||
        Clist.append(Cs)

    d = rng.standard_normal((n_c, n_x)) / np.sqrt(n_x)
    e = rng.standard_normal(n_c) * 0.1

    while True:
        Q = rng.standard_normal((n_c, n_z)) / np.sqrt(n_z)
        sv = np.linalg.svd(Q, compute_uv=False)
        if sv.min() > 1e-3:
            break

    Bp = rng.standard_normal((n_x, n_x)) / np.sqrt(n_x)
    P = mu_f * np.eye(n_x) + 0.5 * (Bp @ Bp.T)
    Bs = rng.standard_normal((n_z, n_z)) / np.sqrt(n_z)
    S = mu_phi * np.eye(n_z) + 0.5 * (Bs @ Bs.T)

    sets = ([box(-box_halfwidth, box_halfwidth, block_size) for _ in range(n_blocks)]
            if with_box else [FreeSet(block_size) for _ in range(n_blocks)])

    return MAQEP(P=P, p=rng.standard_normal(n_x) * 0.1, S=S,
                 s=rng.standard_normal(n_z) * 0.1,
                 Clist=Clist, d=d, e=e, Q=Q, blocks=blocks, sets=sets,
                 name=f"random(nx={n_x},nc={n_c},nz={n_z},|C|={c_scale:g},seed={seed})")
