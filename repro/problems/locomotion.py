"""The paper's locomotion problem (Section 4): eq. (5) reduced to eq. (6).

Centroidal momentum dynamics with a fixed contact sequence.  Appendix C gives

    c_i(f)  = c_init + cdot_init i dt + sum_{i'=0}^{i-2} (i-1-i') (sum_j f_{i'}^j/m + g) dt^2
    cdot_i(f) = cdot_init + sum_{i'=0}^{i-1} sum_j (f_{i'}^j/m + g) dt

and with k'_{i+1} := k_{i+1} - k_i the angular-momentum dynamics become eq. (6):

    k'_0     = k_init
    k'_{i+1} = sum_j ( (r_i^j - c_i(f)) x f_i^j ) dt ,   i = 0..T-1

which is *multi-affine quadratic* in the force blocks x_i := [f_i^1..f_i^N]:
substituting c_i(f) produces bilinear terms between block i' <= i-2 and block i
(never a block with itself, so every C_j has zero diagonal blocks, Definition
2.1), and those bilinear coefficients carry a factor dt * dt^2 = dt^3 -- the
"nonlinear term proportional to (dt)^3" of Section 4.

Written in the form of eq. (1) with z := k' and Q := I (full row rank, so
Assumption 2.6 holds by construction):

    A(x)_0     = -k_init
    A(x)_{i+1} = -sum_j ( (r_i^j - c_i(f)) x f_i^j ) dt

Objective (Section 5, "2D locomotion problem"):
    f(f) = 0.5 sum_i ||f_i||^2 ,   phi(k') = 5 sum_i ||k'_i||^2
Indicators: block-separable polyhedral approximations of the friction cones
Omega_i^j (eq. 5), i.e. inner pyramid approximations of ||f_tangential|| <= mu f_normal
together with 0 <= f_normal <= f_max.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from repro.core import MAQEP, Polyhedron


def _levi_civita(dim: int):
    """eps[c, a, b] such that (u x v)_c = sum_ab eps[c,a,b] u_a v_b."""
    if dim == 3:
        eps = np.zeros((3, 3, 3))
        for c, a, b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            eps[c, a, b] = 1.0
            eps[c, b, a] = -1.0
        return eps
    if dim == 2:  # scalar angular momentum: u x v = u_x v_y - u_y v_x
        eps = np.zeros((1, 2, 2))
        eps[0, 0, 1] = 1.0
        eps[0, 1, 0] = -1.0
        return eps
    raise ValueError("dim must be 2 or 3")


def friction_pyramid(mu: float, fmax: float, dim: int, normal: int = None) -> Polyhedron:
    """Polyhedral (inner pyramid) approximation of a friction cone in R^dim.

    normal axis is the last coordinate.  Constraints:
        |f_t| <= mu f_n  for each tangential axis t,   0 <= f_n <= fmax.
    """
    n = dim
    nz = dim - 1 if normal is None else normal
    rows, rhs = [], []
    for t in range(dim):
        if t == nz:
            continue
        for sgn in (1.0, -1.0):
            row = np.zeros(n)
            row[t] = sgn
            row[nz] = -mu
            rows.append(row)
            rhs.append(0.0)
    row = np.zeros(n); row[nz] = 1.0; rows.append(row); rhs.append(fmax)     # f_n <= fmax
    row = np.zeros(n); row[nz] = -1.0; rows.append(row); rhs.append(0.0)     # f_n >= 0
    return Polyhedron(np.array(rows), np.array(rhs), label=f"friction_pyramid(mu={mu:g})")


def build_locomotion(
    T: int = 20,
    N: int = 2,
    dim: int = 2,
    dt: float = 0.05,
    m: float = 2.0,
    g_acc: float = 9.81,
    c_init=None,
    cdot_init=None,
    k_init=None,
    mu_fric: float = 0.7,
    fmax: float = 200.0,
    phi_weight: float = 5.0,
    contacts: str = "alternating",
    seed: int = 0,
    with_indicators: bool = True,
) -> MAQEP:
    """Assemble eq. (6) as an MAQEP instance.

    T force blocks (i = 0..T-1), n_z = (T+1) * D momentum variables where
    D = 3 for dim=3 and D = 1 for dim=2 (scalar angular momentum).
    """
    rng = np.random.default_rng(seed)
    eps = _levi_civita(dim)
    D = eps.shape[0]                      # angular-momentum dimension
    nf = dim * N                          # variables per force block
    n_x = T * nf
    n_c = (T + 1) * D
    n_z = n_c

    c_init = np.zeros(dim) if c_init is None else np.asarray(c_init, float)
    cdot_init = np.zeros(dim) if cdot_init is None else np.asarray(cdot_init, float)
    k_init = np.zeros(D) if k_init is None else np.asarray(k_init, float)
    gvec = np.zeros(dim)
    gvec[-1] = -g_acc

    # ---- contact locations r_i^j (known) ----
    if contacts == "alternating":            # walking gait: feet swap along +x
        r = np.zeros((T, N, dim))
        step = 0.12
        for i in range(T):
            for j in range(N):
                r[i, j, 0] = c_init[0] + (j - (N - 1) / 2) * 0.2 + step * (i // 4)
                if dim == 3:
                    r[i, j, 1] = (j % 2 - 0.5) * 0.15
                r[i, j, -1] = 0.0
    elif contacts == "random":
        r = rng.uniform(-0.25, 0.25, size=(T, N, dim))
        r[:, :, -1] = 0.0
    else:
        raise ValueError(contacts)

    def fidx(i, j):
        base = i * nf + j * dim
        return np.arange(base, base + dim)

    # ---- constraint data ----
    Craw = [sp.lil_matrix((n_x, n_x)) for _ in range(n_c)]
    d = np.zeros((n_c, n_x))
    e = np.zeros(n_c)

    # row block 0:  k'_0 = k_init  ->  A_0 = -k_init, Q = I
    e[:D] = -k_init

    for i in range(T):
        rows = np.arange((i + 1) * D, (i + 2) * D)
        # constant part of c_i(f):  c_init + cdot_init i dt + sum_{i'} alpha g dt^2
        alpha = np.array([(i - 1 - ip) * dt ** 2 for ip in range(max(i - 1, 0))])
        c_const = c_init + cdot_init * (i * dt) + (alpha.sum() * gvec if alpha.size else 0.0)
        for j in range(N):
            vj = fidx(i, j)
            # linear part: -dt * ((r_i^j - c_const) x f_i^j)
            u0 = r[i, j] - c_const
            for cc in range(D):
                for a in range(dim):
                    for b in range(dim):
                        if eps[cc, a, b] != 0.0:
                            d[rows[cc], vj[b]] += -dt * eps[cc, a, b] * u0[a]
            # bilinear part: +dt/m * sum_{i'<=i-2} alpha_{i,i'} ( (sum_l f_{i'}^l) x f_i^j )
            for ip in range(max(i - 1, 0)):
                coef = dt * alpha[ip] / m
                for l in range(N):
                    ul = fidx(ip, l)
                    for cc in range(D):
                        for a in range(dim):
                            for b in range(dim):
                                w = eps[cc, a, b]
                                if w != 0.0:
                                    Craw[rows[cc]][ul[a], vj[b]] += coef * w

    Clist = []
    for Cr in Craw:
        Cr = Cr.tocsr()
        Clist.append((Cr + Cr.T).tocsr())      # 0.5 x^T C x reproduces the bilinear form

    # ---- objective ----
    P = np.eye(n_x)                            # f(f) = 0.5 ||f||^2
    p = np.zeros(n_x)
    S = 2.0 * phi_weight * np.eye(n_z)         # phi(z) = phi_weight * ||z||^2
    s = np.zeros(n_z)
    Q = np.eye(n_c)                            # Assumption 2.6 satisfied by construction

    blocks = [np.arange(i * nf, (i + 1) * nf) for i in range(T)]
    if with_indicators:
        pyr = friction_pyramid(mu_fric, fmax, dim)
        # one block = all N forces at time i -> block-diagonal stack of the pyramids
        G = np.zeros((pyr.G.shape[0] * N, nf))
        h = np.tile(pyr.h, N)
        for j in range(N):
            G[j * pyr.G.shape[0]:(j + 1) * pyr.G.shape[0], j * dim:(j + 1) * dim] = pyr.G
        sets = [Polyhedron(G, h, label=f"friction_pyramid(mu={mu_fric:g})xN{N}") for _ in range(T)]
    else:
        from repro.core import FreeSet
        sets = [FreeSet(nf) for _ in range(T)]

    prob = MAQEP(P=P, p=p, S=S, s=s, Clist=Clist, d=d, e=e, Q=Q,
                 blocks=blocks, sets=sets,
                 name=f"locomotion{dim}D(T={T},N={N},dt={dt:g})")
    prob.meta = dict(T=T, N=N, dim=dim, D=D, dt=dt, m=m, g=g_acc, mu_fric=mu_fric,
                     fmax=fmax, contacts=contacts, r=r, c_init=c_init,
                     cdot_init=cdot_init, k_init=k_init, with_indicators=with_indicators)
    return prob


def feasible_start(prob: MAQEP, rng=None, scale: float = 1.0) -> np.ndarray:
    """A strictly feasible initial force trajectory (inside every friction cone).

    Uses a gravity-compensating vertical force split equally over the contacts,
    which lies strictly inside the pyramid whenever fmax is large enough.
    """
    md = prob.meta
    dim, N, T, m, g = md["dim"], md["N"], md["T"], md["m"], md["g"]
    x0 = np.zeros(prob.n_x)
    nf = dim * N
    fn = m * g / N * scale
    for i in range(T):
        for j in range(N):
            base = i * nf + j * dim
            x0[base + dim - 1] = fn
    if rng is not None:
        x0 = x0 + rng.normal(0.0, 0.02 * fn, size=x0.shape)
        for i in range(T):  # keep normals positive
            for j in range(N):
                base = i * nf + j * dim
                x0[base + dim - 1] = abs(x0[base + dim - 1])
    return x0
