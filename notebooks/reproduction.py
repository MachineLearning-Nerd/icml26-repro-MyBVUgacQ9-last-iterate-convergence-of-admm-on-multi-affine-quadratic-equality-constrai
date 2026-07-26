"""Interactive walkthrough of the reproduction of arXiv:2603.11919.

Self-contained: it needs only `marimo`, `numpy` and `matplotlib`.  It does not
import the `repro` package.  It re-implements Algorithm 1 from the paper in ~40
lines so a reader can check the algorithm against the paper directly, then reads
the published raw artifacts (if present) to show the verdicts those runs
produced.

    uvx marimo edit notebooks/reproduction.py
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Last-iterate Convergence of ADMM on Multi-affine Quadratic Equality Constrained Problems

    Reproduction of **arXiv:2603.11919** (OpenReview `MyBVUgacQ9`).

    The paper studies

    $$\min_{x,z}\; F(x) + \varphi(z) \quad \text{s.t.}\quad A(x) + Qz = 0,
    \qquad A(x)_j = \tfrac12 x^\top C_j x + d_j^\top x + e_j$$

    where each $C_j$ has **zero diagonal blocks** — that is what makes $A$
    *multi-affine*: it is affine in each block of $x$ with the others fixed,
    but jointly quadratic. That structure is exactly what lets Algorithm 1
    minimise each block **exactly**, in closed form.

    This notebook rebuilds Algorithm 1 from scratch and reproduces the
    Section-5 toy problem, then surfaces the verdicts from the full runs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The Section-5 problem

    $n_x = 4$ split into four scalar blocks, one constraint, one $z$:

    $$C_1 = \begin{pmatrix}0&1&0&0\\1&0&0&0\\0&0&0&-1\\0&0&-1&0\end{pmatrix},
    \quad e_1 = 1,\quad Q = [q].$$

    $C_1$ has a zero diagonal in every $1\times1$ block, so the instance
    satisfies Definition 2.1. The paper sweeps $q$; Theorem 3.1's threshold
    makes $\rho$ scale like $1/q^2$, so larger $q$ needs *less* penalty.
    """)
    return


@app.cell
def _(np):
    def toy_q(q: float):
        """The Section-5 instance. Returns the problem data as a plain dict."""
        C = np.zeros((4, 4))
        C[0, 1] = C[1, 0] = 1.0
        C[2, 3] = C[3, 2] = -1.0
        return dict(
            C=[C],
            d=np.zeros((1, 4)),
            e=np.array([1.0]),
            Q=np.array([[q]]),
            P=np.eye(4),      # f(x) = 1/2 ||x||^2
            p=np.zeros(4),
            R=np.eye(1),      # phi(z) = 1/2 ||z||^2
            blocks=[[0], [1], [2], [3]],
        )

    return (toy_q,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Algorithm 1

    One sweep = exact minimisation of the augmented Lagrangian over each
    $x$-block in sequence, then over $z$, then a dual ascent step.

    The key line is `M = C[:, b] @ x + d[:, b]`: with the other blocks held
    fixed, the constraint is **affine** in block `b` with Jacobian `M`, so
    the block subproblem is an ordinary strongly convex QP and the exact
    minimiser is a single linear solve. No inner loop, no linearisation —
    that exactness is what the convergence theory assumes.
    """)
    return


@app.cell
def _(np):
    def admm(prob, rho, x0, z0, w0, K):
        """Algorithm 1. Returns the Lagrangian and constraint-violation traces."""
        C, d, e = prob["C"], prob["d"], prob["e"]
        Q, P, p, R = prob["Q"], prob["P"], prob["p"], prob["R"]
        blocks = prob["blocks"]
        x, z, w = x0.copy(), z0.copy(), w0.copy()
        Ls, Vs = [], []

        def A_of(xv):
            return np.array([0.5 * xv @ Cj @ xv for Cj in C]) + d @ xv + e

        for _k in range(K):
            for b in blocks:
                # constraint Jacobian w.r.t. this block, others frozen
                M = np.stack([Cj[np.ix_(range(len(x)), b)].T @ x for Cj in C]) + d[:, b]
                r = A_of(x) - M @ x[b] + Q @ z
                Pbb = P[np.ix_(b, b)]
                H = Pbb + rho * (M.T @ M)
                g = (P @ x)[b] - Pbb @ x[b] + p[b] + M.T @ (w + rho * r)
                x[b] = np.linalg.solve(H, -g)
            Ax = A_of(x)
            z = np.linalg.solve(R + rho * (Q.T @ Q), -(Q.T @ (w + rho * Ax)))
            viol = A_of(x) + Q @ z
            w = w + rho * viol
            L = (0.5 * x @ P @ x + p @ x + 0.5 * z @ R @ z
                 + w @ viol + 0.5 * rho * viol @ viol)
            Ls.append(L)
            Vs.append(np.linalg.norm(viol))
        return np.array(Ls), np.array(Vs)

    return (admm,)


@app.cell
def _(mo):
    q_slider = mo.ui.slider(1, 30, value=10, label="q")
    rho_mult = mo.ui.slider(1.0, 8.0, step=0.5, value=2.0, label="rho / rho_threshold")
    mo.vstack([q_slider, rho_mult])
    return q_slider, rho_mult


@app.cell
def _(admm, np, q_slider, rho_mult, toy_q):
    prob = toy_q(float(q_slider.value))
    # Theorem 3.1 threshold, with L_phi = mu_phi = 1 for this instance
    lam = float(np.linalg.eigvalsh(prob["Q"].T @ prob["Q"])[0])
    rho_thm = max(4.0 / lam, 4.0 / np.sqrt(lam))
    rho = rho_mult.value * rho_thm

    rng = np.random.default_rng(0)
    L_trace, viol = admm(prob, rho, rng.standard_normal(4), rng.standard_normal(1),
                         np.zeros(1), 60)
    # L is non-monotone for ADMM, so the gap is taken in absolute value against
    # the converged value rather than as a signed decrease.
    gap = np.abs(L_trace - L_trace[-1])
    return gap, rho, rho_thm, viol


@app.cell
def _(gap, mo, rho, rho_thm, viol):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].semilogy(gap[gap > 0], lw=1.6)
    ax[0].set_xlabel("iteration $k$")
    ax[0].set_ylabel(r"$|L_k - L_\infty|$")
    ax[0].set_title("Lagrangian gap")
    ax[1].semilogy(viol, lw=1.6, color="crimson")
    ax[1].set_xlabel("iteration $k$")
    ax[1].set_ylabel(r"$\|A(x)+Qz\|$")
    ax[1].set_title("constraint violation")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()

    mo.vstack([
        mo.md(f"Theorem 3.1 threshold `rho = {rho_thm:.3g}`, using `rho = {rho:.3g}`."),
        fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A closed-form check on the implementation

    This instance has a known optimum. Taking $x = 0$ makes the constraint
    $1 + qz = 0$, so $z = -1/q$ and the objective is $1/(2q^2)$; no $x \neq 0$
    does better. So the solver can be checked against an exact value rather
    than against itself — the cheapest possible defence against a plausible
    but wrong implementation.
    """)
    return


@app.cell
def _(admm, mo, np, toy_q):
    check_rows = ["| $q$ | ADMM $L_\\infty$ | exact $1/(2q^2)$ | rel. error |",
                  "|---|---|---|---|"]
    for q_val in (1.0, 10.0, 30.0):
        p_chk = toy_q(q_val)
        lam_chk = float(np.linalg.eigvalsh(p_chk["Q"].T @ p_chk["Q"])[0])
        rng_chk = np.random.default_rng(0)
        L_chk, _v = admm(p_chk, 2.0 * max(4.0 / lam_chk, 4.0 / np.sqrt(lam_chk)),
                         rng_chk.standard_normal(4), rng_chk.standard_normal(1),
                         np.zeros(1), 400)
        exact = 1.0 / (2.0 * q_val**2)
        check_rows.append(f"| {q_val:g} | {L_chk[-1]:.10f} | {exact:.10f} | "
                          f"{abs(L_chk[-1] - exact) / exact:.2e} |")
    mo.md("\n".join(check_rows))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Verdicts from the full runs

    The curves above are one small instance. The recorded verdicts come from
    the full suite (locomotion in 2D/3D, a randomised multi-affine ensemble,
    and the Appendix-B problems), whose raw outputs are published alongside
    this notebook under `raw/`. Each claim is **VERIFIED**, **FALSIFIED** or
    **BLOCKED** — never "passing by default".
    """)
    return


@app.cell
def _(mo):
    import json
    import os

    _cands = ["raw/summary.json", "../raw/summary.json",
              ".openresearch/artifacts/summary.json",
              "../.openresearch/artifacts/summary.json"]
    _path = next((p for p in _cands if os.path.exists(p)), None)
    if _path is None:
        _out = mo.md(
            "> No `summary.json` found next to this notebook. Run the suite with "
            "`uv run --frozen python -m repro.run_all`, or open this notebook "
            "beside the published `raw/` directory, to populate this table."
        )
    else:
        _s = json.load(open(_path))
        _rows = ["| claim | verdict | confidence | headline |", "|---|---|---|---|"]
        for _k, _v in _s["claims"].items():
            _rows.append(
                f"| {_k} | **{_v.get('verdict')}** | {_v.get('confidence', '-')} | "
                f"{_v.get('headline', _v.get('reason', ''))} |")
        _out = mo.md("\n".join(_rows))
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. What this notebook does *not* show

    - The rate **classifier** is what turns these curves into a verdict, and
      it is calibrated against sequences of known rate — including a
      $\Theta(1/k)$ sequence it must *reject* as $o(1/k)$. Eyeballing a
      log plot is not evidence; see `raw/calibration/`.
    - The real-robot conjunct of the hardware claim cannot be run here and is
      reported **BLOCKED**, not passing.

    See the *Limitations and deviations* page of the logbook for the full list.
    """)
    return


if __name__ == "__main__":
    app.run()
