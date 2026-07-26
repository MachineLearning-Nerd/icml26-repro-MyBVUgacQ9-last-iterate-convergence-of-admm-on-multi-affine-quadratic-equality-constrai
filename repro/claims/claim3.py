"""Claim 3 -- Theorem 3.3.

Contract
--------
IF  the assumptions of Theorem 3.1 hold,
AND Q satisfies equation (4),
AND every {I_i} is the indicator of a polyhedron,
THEN there exist c_2 > 1 and r > 0 with

    L(x^k,z^k,w^k) - min_{(x,z) in B(x^k,z^k;r)} L(x,z,w^k)  in  O(c_2^{-k}),

AND lim_k (x^k,z^k) = (x*,z*) is a local minimum of problem (1).

Two things the previous reproduction got wrong, and what is done instead
-----------------------------------------------------------------------
1. *The polyhedral indicator must actually bind.*  Theorem 3.3 exists precisely
   to drop the second-order differentiability of L at the limit point that
   Theorem 3.2 assumes, and L is second-order differentiable there whenever no
   indicator is active.  A box centred on the origin is never active for these
   problems (the least-norm objective drives x* to the origin), so such a run
   tests Theorem 3.2's setting, not Theorem 3.3's.  Every instance used here is
   therefore checked to have at least one *active* constraint at the limit, and
   the run is only counted when L is verifiably NOT second-order differentiable
   there.
2. *The decaying quantity is the local gap, not L^k - L*.*  The theorem freezes
   the dual variable at w^k and compares against the best point of a ball of
   fixed radius r.  That quantity is computed explicitly in `repro.localgap`.

Negative control
----------------
The same instances with ||C|| pushed far above the equation-(4) regime, which
violates a hypothesis Theorem 3.3 shares with Theorem 3.2.  The local gap must
then fail to decay geometrically.  A second, descriptive comparison replaces the
polyhedra by (non-polyhedral) Euclidean balls that are also active at the limit:
Theorem 3.3 says nothing there, so this is reported, not scored.
"""

from __future__ import annotations

import numpy as np

from repro.analysis import eq4_certificate, reduced_hessian_check
from repro.core import admm
from repro.localmin import indicator_activity, local_minimality_test
from repro.parallel import pmap, workers
from repro.provenance import write_artifact, write_csv
from repro.rates import classify
from repro.specs import build

TITLE = "Theorem 3.3: geometric decay of the local gap under active polyhedral indicators"

# radius r of the ball B(x^k,z^k;r): a constant, as the theorem requires, chosen
# per instance as a fixed fraction of the initial iterate scale (never tuned to
# the result)
R_FRACTION = 0.05
LOCAL_GAP_ITERS = 120


def _one(job: dict) -> dict:
    from repro.localgap import local_gap_sequence
    prob, x0 = build(job["spec"])
    a = prob.audit()
    rho = prob.rho_threshold() * job.get("rho_multiplier", 1.0)
    K = job["K"]
    tr = admm(prob, rho, x0, K=K, store_iterates=True)
    act = indicator_activity(prob, tr.x)
    cert = eq4_certificate(prob, x0)
    rh = reduced_hessian_check(prob, tr)
    r_ball = R_FRACTION * max(float(np.linalg.norm(x0)), 1.0)

    ks = np.arange(min(LOCAL_GAP_ITERS, K))
    lg = local_gap_sequence(prob, tr, rho, r=r_ball, ks=ks, n_starts=job.get("n_starts", 2))
    gaps = np.asarray(lg["gaps"], float)
    cl = classify(np.abs(gaps))

    lm = (local_minimality_test(prob, tr.x, n_samples=job.get("n_localmin", 120))
          if job.get("localmin", True) else None)

    row = dict(
        label=job["label"], instance=prob.name,
        n_x=a["n_x"], n_c=a["n_c"], n_z=a["n_z"], norm_C=a["norm_C"], rho=rho, K=K,
        all_sets_polyhedral=a["all_sets_polyhedral"],
        n_active_constraints=act["n_active_constraints"],
        n_constraints=act["n_constraints"],
        min_slack=act["min_slack"],
        indicator_active_at_limit=act["any_active"],
        L_second_order_differentiable=act["lagrangian_second_order_differentiable"],
        eq4_certificate_holds=cert["eq4_certificate_holds"],
        eq4_margin=cert["eq4_margin"],
        lam_min_reduced_hessian=rh["lam_min_reduced_hessian_x"],
        norm_sum_w_C=rh["norm_sum_w_C"], norm_w_star=rh["norm_w_star"],
        eq4_operative_condition=rh["second_order_sufficient"],
        r_ball=r_ball,
        local_gap_first=float(gaps[0]), local_gap_last=float(gaps[-1]),
        determined=bool(cl.get("determined", False)),
        local_gap_geometric=bool(cl["linear"]),
        c2_estimate=cl.get("c1_envelope", cl.get("c1_estimate")),
        max_one_step_ratio=cl.get("max_one_step_ratio"),
        decades_of_decay=cl.get("decades_of_decay"),
        final_violation=float(tr.viol[-1]),
        subproblem_max_kkt=tr.max_kkt,
        is_local_minimum=(None if lm is None else lm["is_local_minimum"]),
        localmin_min_delta_V=(None if lm is None else lm["min_delta_V"]),
        seconds=tr.seconds,
    )
    stride = max(1, len(gaps) // 200)
    return dict(row=row, gaps=gaps[::stride].tolist(), classify=cl, localmin=lm)


def _jobs():
    jobs = []
    # (a) the paper's locomotion problem: friction pyramids + the support band,
    #     which bind at the optimum by construction
    for T, N, dim, m, dt in [(15, 2, 2, 2.0, 0.05), (30, 2, 2, 2.0, 0.05),
                             (15, 4, 3, 50.0, 0.02)]:
        jobs.append(dict(spec=dict(kind="locomotion", T=T, N=N, dim=dim, dt=dt, m=m),
                         K=200, label=f"locomotion{dim}D|T={T}", n_localmin=60))
    # (b) the Section-5 toy with a box whose lower face binds
    for q in (10.0, 20.0):
        for mult in (1.0, 50.0):
            jobs.append(dict(spec=dict(kind="toy", q=q, box=(0.05, 0.6),
                                       x0=[0.3, 0.3, 0.3, 0.3]),
                             K=200, rho_multiplier=mult,
                             label=f"toy|q={q:g}|box[0.05,0.6]|rho_x{mult:g}"))
    # (c) randomised ensemble with a binding box, small ||C|| (eq. 4 regime)
    for seed in (1, 2, 3):
        jobs.append(dict(spec=dict(kind="random", n_blocks=20, block_size=4, n_c=10,
                                   n_z=12, c_scale=0.01, seed=seed,
                                   box_bounds=(0.1, 0.5)),
                         K=1500, label=f"random|seed={seed}|box[0.1,0.5]", n_localmin=60))
    return jobs


def _control_jobs():
    """Negative control: same binding polyhedra, ||C|| far outside eq. (4)."""
    return [dict(spec=dict(kind="random", n_blocks=20, block_size=4, n_c=10, n_z=12,
                           c_scale=10.0, seed=seed, box_bounds=(0.1, 0.5)),
                 K=400, label=f"CONTROL|large-C|seed={seed}|box[0.1,0.5]",
                 localmin=False)
            for seed in (1, 2, 3)]


def _ball_jobs():
    """Descriptive comparison: active but NON-polyhedral (ball) indicators.

    Theorem 3.3 does not apply; reported, not scored.  The local-gap solver
    handles polyhedra only, so only the assumption audit and the Lagrangian gap
    are reported for these.
    """
    return [dict(spec=dict(kind="toy", q=q, ball=0.2, x0=[0.3, 0.3, 0.3, 0.3]),
                 K=400, label=f"BALL|q={q:g}") for q in (10.0, 20.0)]


def _ball_one(job: dict) -> dict:
    from repro.analysis import gap_sequence
    prob, x0 = build(job["spec"])
    a = prob.audit()
    rho = prob.rho_threshold()
    gap, L_ref, tr = gap_sequence(prob, rho, x0, K=job["K"], ref_mult=3)
    cl = classify(gap)
    act = indicator_activity(prob, tr.x)
    return dict(label=job["label"], instance=prob.name,
                all_sets_polyhedral=a["all_sets_polyhedral"],
                n_active_constraints=act["n_active_constraints"],
                determined=bool(cl.get("determined", False)),
                lagrangian_gap_geometric=bool(cl["linear"]),
                note="non-polyhedral indicators: outside the hypothesis of Theorem 3.3")


def run() -> dict:
    jobs, ctrl_jobs = _jobs(), _control_jobs()
    print(f"  {len(jobs)} main + {len(ctrl_jobs)} control configurations on "
          f"{workers()} worker processes", flush=True)
    res = pmap(_one, jobs + ctrl_jobs, desc="claim3")
    rows = [r["row"] for r in res]
    main = [r for r in rows if not r["label"].startswith("CONTROL")]
    ctrl = [r for r in rows if r["label"].startswith("CONTROL")]

    print(f"\n  {'label':<32}{'poly':>6}{'active':>8}{'2nd-ord':>9}{'eq4':>6}"
          f"{'det':>5}{'geom':>6}{'c2':>10}{'decades':>9}{'locmin':>8}")
    for r in rows:
        print(f"  {r['label']:<32}{str(r['all_sets_polyhedral']):>6}"
              f"{r['n_active_constraints']:>4}/{r['n_constraints']:<3}"
              f"{str(r['L_second_order_differentiable']):>9}"
              f"{str(r['eq4_certificate_holds']):>6}{str(r['determined']):>5}"
              f"{str(r['local_gap_geometric']):>6}"
              f"{(r['c2_estimate'] if r['c2_estimate'] else float('nan')):>10.3f}"
              f"{(r['decades_of_decay'] or 0):>9.1f}{str(r['is_local_minimum']):>8}")

    # every main instance must genuinely be in Theorem 3.3's regime
    # Regime membership uses the *operative* form of equation (4) established in
    # Appendix D -- positive definiteness of nabla^2 f(x*) + sum_i w*_i C_i --
    # because the a-priori bound of eq. (29)-(31) is, as the paper itself notes
    # for the locomotion setting, very conservative.  The count of instances
    # passing the stricter a-priori certificate is reported alongside.
    in_regime = [r for r in main if r["all_sets_polyhedral"]
                 and r["indicator_active_at_limit"]
                 and not r["L_second_order_differentiable"]
                 and r["eq4_operative_condition"]]
    n_apriori = sum(r["eq4_certificate_holds"] for r in in_regime)
    geom = [r for r in in_regime if r["determined"] and r["local_gap_geometric"]]
    lmin = [r for r in in_regime if r["is_local_minimum"] is not False]
    print(f"\n  instances genuinely in the Theorem 3.3 regime "
          f"(polyhedral AND active AND L not C^2 at the limit AND eq.(4)): "
          f"{len(in_regime)}/{len(main)}")
    print(f"    local gap decays geometrically in {len(geom)}/{len(in_regime)}")
    print(f"    limit point is a local minimum in  {len(lmin)}/{len(in_regime)}")
    print(f"    (of these, {n_apriori} also pass the stricter a-priori eq.(4) certificate)")

    ctrl_fail = [r for r in ctrl if not (r["determined"] and r["local_gap_geometric"])]
    ctrl_stagnant = [r for r in ctrl if (r["decades_of_decay"] or 0.0) < 3.0]
    control_ok = bool(ctrl) and len(ctrl_fail) == len(ctrl)
    print(f"\n  negative control (same binding polyhedra, ||C||=10 outside eq.(4)): "
          f"{len(ctrl_fail)}/{len(ctrl)} do NOT decay geometrically -> "
          f"control failed as intended: {control_ok} "
          f"({len(ctrl_stagnant)}/{len(ctrl)} stagnate, i.e. under 3 decades of decay)")

    balls = pmap(_ball_one, _ball_jobs(), desc="claim3-ball")
    for b in balls:
        print(f"  [descriptive] {b['label']}: non-polyhedral active indicators "
              f"({b['n_active_constraints']} active), Lagrangian gap geometric="
              f"{b['lagrangian_gap_geometric']} - outside Theorem 3.3's hypothesis")

    write_csv("claim3/claim3_results.csv", rows)
    write_artifact("claim3/claim3_local_gaps.json",
                   {r["row"]["label"]: r["gaps"] for r in res})
    write_artifact("claim3/claim3_ball_comparison.json", balls)

    ok = (len(in_regime) >= 6 and len(geom) == len(in_regime)
          and len(lmin) == len(in_regime) and control_ok)
    return dict(
        verdict="VERIFIED" if ok else "BLOCKED",
        confidence="MEDIUM" if ok else "LOW",
        headline=(f"{len(in_regime)}/{len(main)} instances have ACTIVE polyhedral "
                  f"indicators with L not second-order differentiable at the limit; "
                  f"the theorem's local gap decays geometrically in {len(geom)} of them; "
                  f"large-||C|| control does not: {control_ok}"),
        n_in_regime=len(in_regime), n_geometric=len(geom), n_localmin=len(lmin),
        n_apriori_certificate=n_apriori,
        control_ok=control_ok,
        artifacts=["claim3/claim3_results.csv", "claim3/claim3_local_gaps.json",
                   "claim3/claim3_ball_comparison.json"],
        internal_failure=False,
    )
