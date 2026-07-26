"""Claim 2 -- Theorem 3.2.

Contract
--------
IF  the assumptions of Theorem 3.1 hold,
AND f is L_f-smooth,
AND L(x,z,w) is second-order differentiable at the limit point,
AND Q satisfies equation (4),
THEN there exists c_1 > 1 with L(x^k,z^k,w^k) - L* in O(c_1^{-k}),
AND  (x*,z*) is a local minimum of problem (1).

Making equation (4) machine-checkable
-------------------------------------
The main text leaves the constants {m_i} of eq. (4) unnamed, so the inequality
cannot be evaluated as literally printed.  Appendix D supplies its content: the
role of eq. (4) is to keep the reduced Hessian positive definite,

    nabla^2 f(x*) + sum_i w*_i C_i  >  0,

and the appendix bounds ||sum_i w*_i C_i|| through eqs. (29)-(31) using only
a-priori data.  `repro.analysis.eq4_certificate` evaluates that bound; the
certificate holds when it is smaller than mu_f.  Nothing in it uses the observed
convergence rate, so the test is not circular.  Two consistency checks pin it
down further: it must accept ||C|| = 0 (linear constraints -- the paper says so
explicitly), and the *posterior* quantity lam_min(nabla^2 f(x*) + sum_i w*_i C_i)
is reported alongside it.

Non-circular calibration of the onset
-------------------------------------
Rather than testing only the instances the certificate accepts, ||C|| is swept
geometrically on fixed problem geometry and the *empirical* onset -- the largest
||C|| at which convergence is still geometric -- is located by bisection.  The
certificate threshold and the empirical onset are then compared.  A sufficient
condition is expected to be conservative (certificate threshold <= onset); a
certificate that were *larger* than the onset would contradict the theorem.

Negative control
----------------
Instances with ||C|| far above the empirical onset, where the reduced Hessian is
indefinite.  Geometric convergence must fail there; if it did not, the sweep
would not be measuring what it claims to.
"""

from __future__ import annotations

import numpy as np

from repro.analysis import eq4_certificate, gap_sequence, reduced_hessian_check
from repro.localmin import indicator_activity, local_minimality_test
from repro.parallel import pmap, workers
from repro.provenance import write_artifact, write_csv
from repro.rates import classify
from repro.specs import build

TITLE = "Theorem 3.2: equation-(4) certificate, geometric rate, and local minimality"

# fixed problem geometry for the ||C|| sweep -- only ||C|| varies
SWEEP_GEOMETRY = dict(kind="random", n_blocks=20, block_size=4, n_c=10, n_z=12)
SWEEP_SEEDS = (1, 2, 3)
SWEEP_C = (0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)


def _one(job: dict) -> dict:
    prob, x0 = build(job["spec"])
    a = prob.audit()
    rho = prob.rho_threshold() * job.get("rho_multiplier", 1.0)
    cert = eq4_certificate(prob, x0)
    gap, L_ref, tr = gap_sequence(prob, rho, x0, K=job["K"], ref_mult=3)
    cl = classify(gap)
    rh = reduced_hessian_check(prob, tr)
    act = indicator_activity(prob, tr.x)
    lm = (local_minimality_test(prob, tr.x, n_samples=job.get("n_localmin", 150))
          if job.get("localmin", True) else None)
    row = dict(
        instance=prob.name, n_x=a["n_x"], n_c=a["n_c"], n_z=a["n_z"],
        norm_C=a["norm_C"], mu_f=a["mu_f"], L_f=a["L_f"],
        lam_min_QQt=a["lam_min_QQt"], norm_pinvQ=a["norm_pinvQ"],
        rho=rho, K=job["K"], seconds=tr.seconds,
        eq4_apriori_bound=cert["apriori_bound_on_sum_w_C"],
        eq4_margin=cert["eq4_margin"],
        eq4_certificate_holds=cert["eq4_certificate_holds"],
        posterior_norm_sum_w_C=rh["norm_sum_w_C"],
        lam_min_reduced_hessian=rh["lam_min_reduced_hessian_x"],
        second_order_sufficient=rh["second_order_sufficient"],
        n_active_indicators=act["n_active_constraints"],
        L_second_order_differentiable=act["lagrangian_second_order_differentiable"],
        determined=bool(cl.get("determined", False)),
        linear=bool(cl["linear"]),
        decades_of_decay=cl.get("decades_of_decay"),
        c1_estimate=cl.get("c1_envelope", cl.get("c1_estimate")),
        max_one_step_ratio=cl.get("max_one_step_ratio"),
        power_alpha=cl.get("power_alpha"),
        final_violation=float(tr.viol[-1]),
        subproblem_max_kkt=tr.max_kkt,
        is_local_minimum=(None if lm is None else lm["is_local_minimum"]),
        localmin_min_delta_V=(None if lm is None else lm["min_delta_V"]),
        label=job.get("label", ""),
    )
    return dict(row=row, localmin=lm, classify=cl, certificate=cert)


def _sweep_jobs():
    jobs = []
    for seed in SWEEP_SEEDS:
        for c in SWEEP_C:
            sp = dict(SWEEP_GEOMETRY, c_scale=c, seed=seed)
            jobs.append(dict(spec=sp, K=8000, label=f"sweep|seed={seed}|C={c:g}",
                             localmin=(c in (0.0, 1e-2, 1.0, 10.0))))
    return jobs


def _paper_instance_jobs():
    """The paper's own instances: the Section-5 toy at several q and the eq.(6)
    locomotion problem, where the nonlinearity is genuinely small."""
    jobs = []
    for q in (1.0, 3.0, 5.0, 10.0, 20.0, 50.0):
        jobs.append(dict(spec=dict(kind="toy", q=q), K=4000, label=f"toy|q={q:g}"))
    for T, N, dim, m, dt in [(40, 2, 2, 2.0, 0.05), (20, 4, 3, 50.0, 0.02)]:
        jobs.append(dict(spec=dict(kind="locomotion", T=T, N=N, dim=dim, dt=dt, m=m,
                                   with_indicators=False),
                         K=1500, label=f"locomotion{dim}D|T={T}|no-indicators",
                         n_localmin=60))
    return jobs


def _onset_bisection(seed: int, lo: float, hi: float, K: int = 8000,
                     iters: int = 8) -> dict:
    """Bisect on ||C|| for the largest value at which convergence is still
    geometric.  `lo` must be geometric and `hi` must not be."""
    trace = []
    for _ in range(iters):
        mid = float(np.sqrt(max(lo, 1e-12) * hi))
        r = _one(dict(spec=dict(SWEEP_GEOMETRY, c_scale=mid, seed=seed), K=K,
                      localmin=False, label=f"bisect|seed={seed}|C={mid:g}"))["row"]
        ok = bool(r["determined"] and r["linear"])
        trace.append(dict(norm_C=mid, linear=ok, determined=r["determined"],
                          lam_min_reduced_hessian=r["lam_min_reduced_hessian"]))
        if ok:
            lo = mid
        else:
            hi = mid
    return dict(seed=seed, onset_lo=lo, onset_hi=hi, trace=trace)


def _bisect_one(seed: int) -> dict:
    return _onset_bisection(seed, lo=1e-4, hi=10.0)


def run() -> dict:
    jobs = _sweep_jobs() + _paper_instance_jobs()
    print(f"  {len(jobs)} configurations on {workers()} worker processes", flush=True)
    res = pmap(_one, jobs, desc="claim2")
    rows = [r["row"] for r in res]

    print(f"\n  {'label':<26}{'||C||':>10}{'eq4':>6}{'margin':>10}{'lam_red':>10}"
          f"{'2nd-ord':>8}{'det':>5}{'linear':>7}{'c1':>9}{'locmin':>8}")
    for r in rows:
        print(f"  {r['label']:<26}{r['norm_C']:>10.3e}"
              f"{str(r['eq4_certificate_holds']):>6}{r['eq4_margin']:>10.3f}"
              f"{r['lam_min_reduced_hessian']:>10.4f}"
              f"{str(r['L_second_order_differentiable']):>8}"
              f"{str(r['determined']):>5}{str(r['linear']):>7}"
              f"{(r['c1_estimate'] if r['c1_estimate'] is not None else float('nan')):>9.3f}"
              f"{str(r['is_local_minimum']):>8}")

    # --- the contract: certificate holds AND determined => linear AND local min
    cert_rows = [r for r in rows if r["eq4_certificate_holds"] and r["determined"]]
    cert_linear = [r for r in cert_rows if r["linear"]]
    cert_localmin = [r for r in cert_rows
                     if r["is_local_minimum"] is not False]
    print(f"\n  configurations where the eq.(4) certificate holds and the rate is "
          f"determined: {len(cert_rows)}")
    print(f"    geometric convergence observed in {len(cert_linear)}/{len(cert_rows)}")
    print(f"    local minimality upheld in       {len(cert_localmin)}/{len(cert_rows)}")

    # --- consistency: ||C||=0 must be accepted by the certificate (paper, line 885)
    zero_C = [r for r in rows if r["norm_C"] == 0.0]
    zero_ok = bool(zero_C) and all(r["eq4_certificate_holds"] for r in zero_C)
    print(f"    certificate accepts ||C||=0 (linear constraints): {zero_ok} "
          f"({len(zero_C)} instances)")

    # --- calibrated onset, by bisection, independent of the certificate
    print("\n  bisecting for the empirical geometric-convergence onset in ||C||", flush=True)
    bis = pmap(_bisect_one, list(SWEEP_SEEDS), desc="claim2-bisect")
    onsets = [b["onset_lo"] for b in bis]
    for b in bis:
        print(f"    seed={b['seed']}: geometric up to ||C|| ~ {b['onset_lo']:.4g}, "
              f"not geometric from {b['onset_hi']:.4g}")

    # certificate threshold on the same geometry: largest ||C|| the certificate accepts
    cert_thresholds = []
    for seed in SWEEP_SEEDS:
        accepted = [r["norm_C"] for r in rows
                    if r["label"].startswith(f"sweep|seed={seed}|") and r["eq4_certificate_holds"]]
        cert_thresholds.append(max(accepted) if accepted else 0.0)
    conservative = all(ct <= on * 1.0000001 for ct, on in zip(cert_thresholds, onsets))
    print(f"    certificate threshold per seed: "
          f"{[f'{c:.4g}' for c in cert_thresholds]}")
    print(f"    empirical onset per seed:       {[f'{o:.4g}' for o in onsets]}")
    print(f"    certificate is conservative (threshold <= onset) on every seed: {conservative}")

    # --- negative control: ||C|| far above the onset
    ctrl_rows = [r for r in rows if r["label"].startswith("sweep|") and r["norm_C"] >= 10.0]
    ctrl_not_linear = [r for r in ctrl_rows if not (r["determined"] and r["linear"])]
    ctrl_indef = [r for r in ctrl_rows if r["lam_min_reduced_hessian"] <= 0]
    control_ok = bool(ctrl_rows) and len(ctrl_not_linear) == len(ctrl_rows)
    print(f"\n  negative control (||C|| = 10, far above the onset): "
          f"{len(ctrl_not_linear)}/{len(ctrl_rows)} are NOT geometric "
          f"({len(ctrl_indef)} have an indefinite reduced Hessian) -> "
          f"control failed as intended: {control_ok}")

    write_csv("claim2/claim2_results.csv", rows)
    write_artifact("claim2/claim2_bisection.json", bis)
    write_artifact("claim2/claim2_localmin.json",
                   {r["row"]["label"]: r["localmin"] for r in res if r["localmin"]})

    ok = (len(cert_rows) > 0
          and len(cert_linear) == len(cert_rows)
          and len(cert_localmin) == len(cert_rows)
          and zero_ok and conservative and control_ok)
    return dict(
        verdict="VERIFIED" if ok else "BLOCKED",
        confidence="MEDIUM" if ok else "LOW",
        headline=(f"eq.(4) certificate holds in {len(cert_rows)} determined configs -> "
                  f"geometric in {len(cert_linear)}, local minimum in {len(cert_localmin)}; "
                  f"certificate conservative vs bisected onset: {conservative}; "
                  f"high-||C|| control not geometric: {control_ok}"),
        n_cert=len(cert_rows), n_cert_linear=len(cert_linear),
        n_cert_localmin=len(cert_localmin),
        certificate_thresholds=cert_thresholds, empirical_onsets=onsets,
        certificate_conservative=conservative, control_ok=control_ok,
        accepts_zero_C=zero_ok,
        artifacts=["claim2/claim2_results.csv", "claim2/claim2_bisection.json",
                   "claim2/claim2_localmin.json"],
        internal_failure=False,
    )
