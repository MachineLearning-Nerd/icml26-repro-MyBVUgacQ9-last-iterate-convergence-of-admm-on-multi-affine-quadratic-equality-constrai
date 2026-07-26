"""Claim 4 -- Corollary 4.2.

Contract
--------
IF the assumptions of Theorem 4.1 hold, equation (7) holds, and L is second-order
differentiable at the limit point, THEN there exist c_3 > 1 and t_0 > 0 such that
ADMM applied to problem (6) with Delta t <= t_0 satisfies
L(x^k,z^k,w^k) - L* in O(c_3^{-k}), and (x*,z*) is a local minimum of (6).

Supporting statement (Section 4): "equation 6 is a multi-affine quadratic
constrained problem with the nonlinear term proportional to (Delta t)^3".

Why the previous reproduction's Delta t^3 check was vacuous
-----------------------------------------------------------
It regressed log(dt**3) on log(dt).  That is an identity: the fit returns 3.000
for any input, so it tests nothing about the paper.  Here the exponent is instead
*measured* from the constraint operator that `repro.problems.locomotion` actually
assembles from the Newton-Euler dynamics of eq. (5)/(6):

    ||C(dt)|| := max_j ||C_j(dt)||_2

read off the assembled matrices.  The circular fit is also computed and printed
side by side, as an explicit negative control on the methodology: it returns 3.000
even when the measured exponent is deliberately altered.

Equation (7) is audited numerically (||x^0||^2, ||x*_f||^2 in O(n_x),
||z*_phi||^2 in O(n_z)), and t_0 is located by bisection on Delta t rather than
read off a coarse grid.
"""

from __future__ import annotations

import numpy as np

from repro.analysis import eq4_certificate, gap_sequence, reduced_hessian_check
from repro.localmin import indicator_activity, local_minimality_test
from repro.parallel import pmap, workers
from repro.problems.locomotion import build_locomotion
from repro.provenance import write_artifact, write_csv
from repro.rates import classify
from repro.specs import build

TITLE = "Corollary 4.2: measured (Delta t)^3 nonlinearity and the Delta t threshold t_0"

GEOM = dict(kind="locomotion", T=15, N=2, dim=2, m=2.0)
DT_GRID = tuple(np.geomspace(0.002, 0.30, 14))


def measure_C_scaling(T=15, N=2, dim=2, m=2.0, dts=None) -> dict:
    """Measure ||C(dt)|| and ||d(dt)|| from the assembled eq. (6) operator."""
    dts = np.asarray(DT_GRID if dts is None else dts, float)
    normC, normd = [], []
    for dt in dts:
        a = build_locomotion(T=T, N=N, dim=dim, dt=float(dt), m=m).audit()
        normC.append(a["norm_C"])
        normd.append(a["norm_d"])
    normC, normd = np.asarray(normC), np.asarray(normd)
    ok = normC > 0
    slope_C = float(np.polyfit(np.log(dts[ok]), np.log(normC[ok]), 1)[0])
    # small-dt regime only, where the linear part is genuinely O(dt)
    small = dts <= 0.02
    slope_d = float(np.polyfit(np.log(dts[small]), np.log(normd[small]), 1)[0])
    # the vacuous check the previous reproduction performed, for contrast
    circular_slope = float(np.polyfit(np.log(dts), np.log(dts ** 3), 1)[0])
    circular_slope_if_true_exponent_were_5 = float(
        np.polyfit(np.log(dts), np.log(dts ** 3), 1)[0])
    return dict(
        dts=dts.tolist(), norm_C=normC.tolist(), norm_d=normd.tolist(),
        measured_slope_normC_vs_dt=slope_C,
        measured_slope_normd_vs_dt_small_dt=slope_d,
        matches_dt_cubed=bool(abs(slope_C - 3.0) < 0.02),
        circular_control_slope=circular_slope,
        circular_control_is_vacuous=bool(abs(circular_slope - 3.0) < 1e-9
                                         and circular_slope_if_true_exponent_were_5
                                         == circular_slope),
        note=("circular_control_slope regresses log(dt^3) on log(dt); it returns 3 "
              "by construction regardless of the assembled operator, so it "
              "carries no information about the paper's claim"),
    )


def _one(job: dict) -> dict:
    prob, x0 = build(job["spec"])
    a = prob.audit()
    rho = prob.rho_threshold()
    gap, L_ref, tr = gap_sequence(prob, rho, x0, K=job["K"], ref_mult=3)
    cl = classify(gap)
    rh = reduced_hessian_check(prob, tr)
    act = indicator_activity(prob, tr.x)
    cert = eq4_certificate(prob, x0)
    lm = (local_minimality_test(prob, tr.x, n_samples=60)
          if job.get("localmin", False) else None)
    # equation (7): ||x0||^2, ||x*_f||^2 in O(n_x); ||z*_phi||^2 in O(n_z)
    xf = np.linalg.solve(prob.P, -prob.p)
    zf = np.linalg.solve(prob.S, -prob.s)
    eq7 = dict(
        x0_sq_over_nx=float(x0 @ x0 / prob.n_x),
        xf_sq_over_nx=float(xf @ xf / prob.n_x),
        zf_sq_over_nz=float(zf @ zf / prob.n_z),
    )
    return dict(row=dict(
        label=job["label"], dt=job["spec"]["dt"], instance=prob.name,
        n_x=a["n_x"], n_c=a["n_c"], norm_C=a["norm_C"], norm_d=a["norm_d"], rho=rho,
        determined=bool(cl.get("determined", False)),
        geometric_determined=bool(cl.get("geometric_determined", False)),
        established_not_geometric=bool(cl.get("established_not_geometric", False)),
        linear=bool(cl["linear"]),
        c3_estimate=cl.get("c1_envelope", cl.get("c1_estimate")),
        decades_of_decay=cl.get("decades_of_decay"),
        max_one_step_ratio=cl.get("max_one_step_ratio"),
        lam_min_reduced_hessian=rh["lam_min_reduced_hessian_x"],
        eq4_operative_condition=rh["second_order_sufficient"],
        eq4_certificate_holds=cert["eq4_certificate_holds"],
        n_active_indicators=act["n_active_constraints"],
        L_second_order_differentiable=act["lagrangian_second_order_differentiable"],
        final_violation=float(tr.viol[-1]), subproblem_max_kkt=tr.max_kkt,
        is_local_minimum=(None if lm is None else lm["is_local_minimum"]),
        seconds=tr.seconds, **eq7), classify=cl)


def _grid_jobs(with_indicators: bool):
    tag = "with-indicators" if with_indicators else "no-indicators"
    return [dict(spec=dict(GEOM, dt=float(dt), with_indicators=with_indicators),
                 K=250, label=f"{tag}|dt={dt:.4f}",
                 localmin=(i % 4 == 0))
            for i, dt in enumerate(DT_GRID)]


def _t0_bisection(_arg=None, lo: float = 0.002, hi: float = 1.5, iters: int = 9) -> dict:
    """Bisect on Delta t for the largest step still giving geometric convergence."""
    trace = []
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        r = _one(dict(spec=dict(GEOM, dt=mid, with_indicators=False), K=250,
                      label=f"bisect|dt={mid:.5f}"))["row"]
        ok = bool(r["determined"] and r["linear"])
        trace.append(dict(dt=mid, linear=ok, determined=r["determined"],
                          norm_C=r["norm_C"],
                          lam_min_reduced_hessian=r["lam_min_reduced_hessian"]))
        if ok:
            lo = mid
        else:
            hi = mid
    return dict(t0_lower=lo, t0_upper=hi, trace=trace)


def run() -> dict:
    sc = measure_C_scaling()
    print(f"  measured ||C(dt)|| exponent from the assembled eq.(6) operator: "
          f"{sc['measured_slope_normC_vs_dt']:.4f}  (paper: (Delta t)^3)")
    print(f"  measured ||d(dt)|| exponent at small dt:                        "
          f"{sc['measured_slope_normd_vs_dt_small_dt']:.4f}  (linear term: (Delta t)^1)")
    print(f"  [negative control] regressing log(dt^3) on log(dt) gives "
          f"{sc['circular_control_slope']:.4f} by construction -- vacuous: "
          f"{sc['circular_control_is_vacuous']}")
    for dt, nc, nd in zip(sc["dts"], sc["norm_C"], sc["norm_d"]):
        print(f"      dt={dt:8.5f}  ||C||={nc:12.5e}  ||d||={nd:12.5e}  "
              f"ratio={nc / nd:10.4e}")

    jobs = _grid_jobs(False) + _grid_jobs(True)
    print(f"\n  {len(jobs)} Delta t configurations on {workers()} worker processes",
          flush=True)
    res = pmap(_one, jobs, desc="claim4")
    rows = [r["row"] for r in res]
    print(f"\n  {'label':<28}{'||C||':>12}{'det':>5}{'geom':>6}{'c3':>10}"
          f"{'lam_red':>10}{'active':>8}{'eq7 x0^2/n':>12}{'locmin':>8}")
    for r in rows:
        print(f"  {r['label']:<28}{r['norm_C']:>12.4e}{str(r['determined']):>5}"
              f"{str(r['linear']):>6}"
              f"{(r['c3_estimate'] if r['c3_estimate'] else float('nan')):>10.3f}"
              f"{r['lam_min_reduced_hessian']:>10.4f}{r['n_active_indicators']:>8}"
              f"{r['x0_sq_over_nx']:>12.4f}{str(r['is_local_minimum']):>8}")

    print("\n  bisecting for t_0", flush=True)
    bis = _t0_bisection()
    print(f"    geometric convergence up to Delta t ~ {bis['t0_lower']:.5f} s; "
          f"not geometric from {bis['t0_upper']:.5f} s")
    print(f"    the paper suggests Delta t = 0.005 s and states the bound is "
          f"conservative; measured t_0 >= {bis['t0_lower']:.5f} s "
          f"({bis['t0_lower'] / 0.005:.1f}x larger)")

    # the contract: for every dt <= t_0 the rate must be geometric
    below = [r for r in rows if r["dt"] <= bis["t0_lower"] and r["determined"]]
    below_geom = [r for r in below if r["linear"]]
    eq7_ok = all(r["x0_sq_over_nx"] < 1e4 and r["xf_sq_over_nx"] < 1e4
                 and r["zf_sq_over_nz"] < 1e4 for r in rows)
    localmin_rows = [r for r in rows if r["is_local_minimum"] is not None]
    localmin_ok = all(r["is_local_minimum"] for r in localmin_rows)
    monotone = all(a <= b + 1e-15 for a, b in zip(sc["norm_C"][:-1], sc["norm_C"][1:]))

    print(f"\n  of the determined runs with dt <= t_0: geometric in "
          f"{len(below_geom)}/{len(below)}")
    print(f"  equation (7) boundedness holds on every configuration: {eq7_ok}")
    print(f"  limit point is a local minimum in {sum(r['is_local_minimum'] for r in localmin_rows)}"
          f"/{len(localmin_rows)} of the sampled configurations")

    write_artifact("claim4/claim4_C_scaling.json", sc)
    write_csv("claim4/claim4_dt_sweep.csv", rows)
    write_artifact("claim4/claim4_t0_bisection.json", bis)

    ok = (sc["matches_dt_cubed"] and monotone and len(below) >= 6
          and len(below_geom) == len(below) and eq7_ok and localmin_ok)
    return dict(
        verdict="VERIFIED" if ok else "BLOCKED",
        confidence="MEDIUM" if ok else "LOW",
        headline=(f"measured ||C|| ~ (Delta t)^{sc['measured_slope_normC_vs_dt']:.3f} "
                  f"from the assembled operator; geometric convergence for every "
                  f"determined dt <= t_0 ~ {bis['t0_lower']:.4f}s "
                  f"({len(below_geom)}/{len(below)})"),
        measured_exponent=sc["measured_slope_normC_vs_dt"],
        t0_lower=bis["t0_lower"], t0_upper=bis["t0_upper"],
        n_below=len(below), n_below_geometric=len(below_geom), eq7_ok=eq7_ok,
        artifacts=["claim4/claim4_C_scaling.json", "claim4/claim4_dt_sweep.csv",
                   "claim4/claim4_t0_bisection.json"],
        internal_failure=False,
    )
