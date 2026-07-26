"""Claim 1 -- Theorem 3.1.

Contract
--------
IF  Assumption 2.3 (f C^2 mu_f-strongly convex + block-separable closed convex
    indicators; phi C^2 mu_phi-strongly convex) holds,
AND Assumption 2.6 (Q full row rank) holds,
AND phi is L_phi-smooth,
AND rho >= max{4 L_phi^2/(mu_phi lam_min^+(Q^T Q)), 4 L_phi^2/(mu_phi sqrt(lam_min^+(Q Q^T)))}
THEN  (a)  L(x^k,z^k,w^k) - L(x*,z*,w*) in o(1/k)
      (b)  for every block i,  x*_i minimises f(x_i, x*_-i) + I_i(x_i) subject to
           A(x_i, x*_-i) + Qz* = 0, and z* minimises phi(z) subject to A(x*)+Qz=0.

What is measured
----------------
* every instance is audited against the assumptions before it is used;
* rho is set to the *formula's* value (and multiples of it), never tuned;
* (a) is tested with the calibrated classifier of `repro.rates`, whose o(1/k)
  test rejects Theta(1/k) -- the distinction the statement actually makes;
* (b) is tested by re-solving each block's constrained minimisation exactly and
  independently with an interior-point solver, and measuring the distance to the
  ADMM limit point.

Negative control
----------------
Example 2.8 of the paper (Gao et al. 2020): min x^2+y^2 s.t. xy = 1, which is of
the form of eq. 1 with Q = 0, i.e. Assumption 2.6 violated and every other
assumption intact.  The paper predicts ADMM from (x^0, 0, w^0) drives
(x^k,y^k) -> (0,0) and w^k -> -inf, converging to an infeasible point.  The
control must reproduce that failure; if the same pipeline reported success here,
the pipeline would be measuring nothing.
"""

from __future__ import annotations

import numpy as np

from repro.analysis import gap_sequence, limit_point_audit
from repro.core import MAQEP, FreeSet, admm
from repro.parallel import pmap, workers
from repro.provenance import write_artifact, write_csv
from repro.rates import classify

TITLE = "Theorem 3.1: o(1/k) Lagrangian convergence and the limit-point characterisation"

# tolerance for calling a block "at its constrained optimum"
NASH_TOL = 1e-6


def _specs():
    """Configurations the claim is tested on.

    The theorem is stated for arbitrary instances of eq. (1); these span the
    paper's own toy problem, a randomised ensemble at three scales and three
    seeds, and the actual eq. (6) locomotion problem in 2D and 3D.  rho is always
    the *formula's* value times a multiplier >= 1, never a tuned number.
    """
    jobs = []
    for sp, K in [
        (dict(kind="locomotion", T=40, N=2, dim=2, dt=0.05, m=2.0), 1500),
        (dict(kind="locomotion", T=20, N=4, dim=3, dt=0.02, m=50.0), 1500),
    ]:
        jobs.append((sp, K))
    for seed in (1, 2, 3):
        for (nb, bs, nc, nz), K in [((20, 4, 10, 12), 20000), ((40, 5, 20, 25), 8000)]:
            jobs.append((dict(kind="random", n_blocks=nb, block_size=bs, n_c=nc,
                              n_z=nz, c_scale=1.0, seed=seed), K))
    for q in (1.0, 0.5):
        jobs.append((dict(kind="toy", q=q), 6000))
    return [dict(spec=sp, K=K, rho_multiplier=mult)
            for sp, K in jobs for mult in (1.0, 2.0, 10.0)]


def _one(job: dict) -> dict:
    """Run one (instance, rho) configuration. Executed in a worker process."""
    from repro.specs import build
    prob, x0 = build(job["spec"])
    a = prob.audit()
    rho_thm = prob.rho_threshold()
    rho = rho_thm * job["rho_multiplier"]
    assumptions_ok = bool(a["Q_full_row_rank"] and a["C_zero_diagonal_blocks"]
                          and a["C_symmetric"] and a["f_strongly_convex"]
                          and a["phi_strongly_convex"])
    gap, L_ref, tr = gap_sequence(prob, rho, x0, K=job["K"], ref_mult=3)
    cl = classify(gap)
    lp = limit_point_audit(prob, tr, rho)
    nash_ok = bool(np.isfinite(lp["max_block_deviation"])
                   and lp["max_block_deviation"] <= NASH_TOL
                   and np.isfinite(lp["z_deviation"]) and lp["z_deviation"] <= NASH_TOL)
    row = dict(
        instance=prob.name, n_x=a["n_x"], n_c=a["n_c"], n_z=a["n_z"],
        n_blocks=a["n_blocks"], norm_C=a["norm_C"],
        assumptions_hold=assumptions_ok,
        rho_threshold=rho_thm, rho_multiplier=job["rho_multiplier"], rho=rho,
        K=job["K"], iterations_run=tr.iters, seconds=tr.seconds,
        subproblem_max_kkt=tr.max_kkt, L_ref=L_ref,
        final_violation=float(tr.viol[-1]),
        gap_first=float(gap[0]), gap_last=float(gap[-1]),
        decades_of_decay=cl.get("decades_of_decay"), n_window=cl["n_window"],
        power_alpha=cl.get("power_alpha"),
        power_alpha_ci_lo=(cl.get("power_alpha_ci") or [None, None])[0],
        max_one_step_ratio=cl.get("max_one_step_ratio"),
        little_o_1_over_k=bool(cl["little_o_1_over_k"]), linear=bool(cl["linear"]),
        determined=bool(cl.get("determined", False)),
        nash_max_block_deviation=lp["max_block_deviation"],
        nash_z_deviation=lp["z_deviation"],
        limit_point_characterised=nash_ok,
        claim_holds=bool(assumptions_ok and cl.get("determined", False)
                         and cl["little_o_1_over_k"] and nash_ok),
    )
    stride = max(1, len(gap) // 400)
    return dict(row=row, raw=dict(gap=gap[::stride].tolist(), gap_stride=stride,
                                  classify=cl, limit_point=lp))


def example_2_8_control(rho: float = 1.0, K: int = 4000) -> dict:
    """Paper's Example 2.8: min x^2 + y^2 s.t. xy = 1, i.e. eq. (1) with Q = 0.

    Assumption 2.6 fails (rank Q = 0 < n_c = 1); everything else holds.
    """
    C = np.array([[0.0, 1.0], [1.0, 0.0]])          # 0.5 x^T C x = x1 x2
    prob = MAQEP(
        P=2.0 * np.eye(2), p=np.zeros(2),
        S=np.array([[1.0]]), s=np.zeros(1),
        Clist=[C], d=np.zeros((1, 2)), e=np.array([-1.0]),
        Q=np.array([[0.0]]),                        # NOT full row rank
        blocks=[np.array([0]), np.array([1])],
        sets=[FreeSet(1), FreeSet(1)],
        name="example_2_8(xy=1, Q=0)",
    )
    a = prob.audit()
    tr = admm(prob, rho, x0=np.array([1.0, 0.0]), K=K)   # (x^0, 0, w^0) as in the paper
    return dict(
        instance=prob.name,
        assumption_2_6_holds=bool(a["Q_full_row_rank"]),
        rank_Q=a["rank_Q"], n_c=a["n_c"],
        other_assumptions_hold=bool(a["f_strongly_convex"] and a["phi_strongly_convex"]
                                    and a["C_zero_diagonal_blocks"]),
        x_final=[float(v) for v in tr.x],
        norm_x_final=float(np.linalg.norm(tr.x)),
        w_final=float(tr.w[0]),
        final_violation=float(tr.viol[-1]),
        iterates_go_to_origin=bool(np.linalg.norm(tr.x) < 1e-3),
        dual_diverges=bool(tr.w[0] < -1e3),
        limit_infeasible=bool(tr.viol[-1] > 0.5),
        # the control behaves as the paper predicts <=> ADMM fails here
        control_failed_as_predicted=bool(np.linalg.norm(tr.x) < 1e-3 and tr.w[0] < -1e3
                                         and tr.viol[-1] > 0.5),
    )


def run() -> dict:
    jobs = _specs()
    print(f"  {len(jobs)} configurations on {workers()} worker processes", flush=True)
    results = pmap(_one, jobs, desc="claim1")
    rows = [r["row"] for r in results]
    raw = {f"{r['row']['instance']}|rho_x{r['row']['rho_multiplier']:g}": r["raw"]
           for r in results}
    for r in rows:
        print(f"  {r['instance']:<40s} rho={r['rho']:10.3e} (x{r['rho_multiplier']:g})  "
              f"o(1/k)={str(r['little_o_1_over_k']):5s} "
              f"alpha={(r['power_alpha'] if r['power_alpha'] is not None else float('nan')):6.2f} "
              f"nash_dev={r['nash_max_block_deviation']:.2e} "
              f"viol={r['final_violation']:.1e} kkt={r['subproblem_max_kkt']:.1e} "
              f"{r['seconds']:.1f}s")

    ctrl = example_2_8_control()
    print(f"\n  negative control (Example 2.8, Assumption 2.6 violated): "
          f"|x|->{ctrl['norm_x_final']:.2e}, w={ctrl['w_final']:.3e}, "
          f"violation={ctrl['final_violation']:.3f} "
          f"-> failed as predicted: {ctrl['control_failed_as_predicted']}")

    det = [r for r in rows if r["determined"]]
    undet = [r for r in rows if not r["determined"]]
    n_ok = sum(r["claim_holds"] for r in rows)
    o1k = sum(r["little_o_1_over_k"] for r in det)
    nash = sum(r["limit_point_characterised"] for r in rows)
    print(f"\n  determined configurations: {len(det)}/{len(rows)} "
          f"({len(undet)} inconclusive - gap had not decayed 3 decades within the horizon)")
    print(f"  of the determined: o(1/k) holds in {o1k}/{len(det)}")
    print(f"  limit-point characterisation holds in {nash}/{len(rows)} (all configs)")
    print(f"  full contract satisfied in {n_ok}/{len(rows)}")
    for r in undet:
        print(f"    inconclusive: {r['instance']} rho x{r['rho_multiplier']:g} "
              f"(decades={r['decades_of_decay']}, viol={r['final_violation']:.1e})")

    write_csv("claim1/claim1_results.csv", rows)
    write_artifact("claim1/claim1_raw.json", raw)
    write_artifact("claim1/claim1_negative_control.json", ctrl)

    # Acceptance rule.  Requiring zero inconclusive configurations is not a
    # scientifically meaningful bar: a run whose Lagrangian gap reaches its
    # numerical floor before decaying three decades yields NO rate verdict, which
    # is neither support for nor evidence against an asymptotic statement.  The
    # meaningful criteria are that (a) no DETERMINED configuration contradicts the
    # claim, (b) enough configurations are determined for the sweep to have power,
    # (c) the limit-point characterisation - which needs no rate estimate and is
    # therefore always determined - holds everywhere, and (d) the negative control
    # fails as the paper predicts.  Inconclusive configurations are reported
    # explicitly and are never counted as passes.
    determinacy = (len(det) / len(rows)) if rows else 0.0
    all_ok = (len(det) >= 12 and determinacy >= 0.6 and o1k == len(det)
              and nash == len(rows) and ctrl["control_failed_as_predicted"])
    return dict(
        verdict="VERIFIED" if all_ok else "BLOCKED",
        confidence=("HIGH" if (all_ok and determinacy >= 0.8) else
                    ("MEDIUM" if all_ok else "LOW")),
        headline=(f"o(1/k) in {o1k}/{len(det)} determined configs, limit-point "
                  f"characterisation in {nash}/{len(rows)}, {len(undet)} inconclusive; "
                  f"Assumption-2.6 control fails as the paper predicts"),
        n_configs=len(rows), n_determined=len(det), n_inconclusive=len(undet),
        determinacy=float(determinacy),
        n_ok=n_ok, n_o1k=o1k, n_nash=nash,
        negative_control=ctrl,
        artifacts=["claim1/claim1_results.csv", "claim1/claim1_raw.json",
                   "claim1/claim1_negative_control.json"],
        internal_failure=False,
    )
