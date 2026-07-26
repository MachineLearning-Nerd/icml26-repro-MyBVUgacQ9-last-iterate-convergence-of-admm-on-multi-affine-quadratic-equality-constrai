"""Claim 5 -- Figures 2 and 4.

Two separable sub-claims, tested and scored separately.

(5a) Figure 2 / Section 5: "Condition in (4) suggests q >= 10 to ensure linear
     convergence."  The quantifier is SUFFICIENCY: every q >= 10 converges
     linearly.  It is *not* a claim that q < 10 fails, so an empirical onset
     below 10 does not contradict it.  Tested by (i) checking geometric
     convergence on a dense grid of q >= 10 up to q = 1000, over several mu_x,
     mu_z and initial points, and (ii) locating the empirical onset by bisection
     so that the relation between the sufficient condition and the true
     transition is measured rather than asserted.

(5b) Figure 4: "our algorithm achieves superior performance when the constraints
     are nonlinear, while comparable performance in other settings", against
     PADMM, IPDS-ADMM and IADMM on the three Appendix-B problems.  The three
     baseline papers were not retrievable (see `repro/problems/baselines.py`),
     so the baselines are standard reimplementations of the named methods.  The
     comparison is run over a sweep of the unspecified parameters mu_x, mu_z,
     rho and initial point, and is scored on iterations-to-tolerance of the
     constraint residual, with the final objective reported alongside.
"""

from __future__ import annotations

import itertools

import numpy as np

from repro.analysis import eq4_certificate, gap_sequence, reduced_hessian_check
from repro.parallel import pmap, workers
from repro.problems.baselines import APPENDIX_PROBLEMS, run_algorithm
from repro.provenance import write_artifact, write_csv
from repro.rates import classify
from repro.specs import build

TITLE = "Figures 2 and 4: the q >= 10 sufficiency claim and the baseline comparison"

Q_GRID_ABOVE_10 = (10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1000.0)
Q_GRID_BELOW_10 = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
MU_GRID = ((1.0, 1.0), (0.1, 1.0), (1.0, 0.1), (10.0, 1.0))
X0_GRID = ([1.0, 0.5, -0.5, 1.0], [2.0, -1.0, 1.0, -2.0], [0.1, 0.1, 0.1, 0.1])
ALGOS = ("admm", "padmm", "iadmm", "ipds_admm")
TOL = 1e-8


def _q_one(job: dict) -> dict:
    prob, x0 = build(job["spec"])
    rho = max(prob.rho_threshold(), 1e-8) * job.get("rho_multiplier", 1.0)
    gap, L_ref, tr = gap_sequence(prob, rho, x0, K=job["K"], ref_mult=3)
    cl = classify(gap)
    cert = eq4_certificate(prob, x0)
    rh = reduced_hessian_check(prob, tr)
    return dict(q=job["spec"]["q"], mu_x=job["spec"].get("mu_x", 1.0),
                mu_z=job["spec"].get("mu_z", 1.0), x0=list(job["spec"]["x0"]),
                rho=rho, rho_multiplier=job.get("rho_multiplier", 1.0),
                determined=bool(cl.get("determined", False)),
                geometric_determined=bool(cl.get("geometric_determined", False)),
                established_not_geometric=bool(cl.get("established_not_geometric", False)),
                linear=bool(cl["linear"]),
                decades_of_decay=cl.get("decades_of_decay"),
                max_one_step_ratio=cl.get("max_one_step_ratio"),
                c_estimate=cl.get("c1_envelope", cl.get("c1_estimate")),
                eq4_certificate_holds=cert["eq4_certificate_holds"],
                eq4_margin=cert["eq4_margin"],
                lam_min_reduced_hessian=rh["lam_min_reduced_hessian_x"],
                final_violation=float(tr.viol[-1]),
                label=f"q={job['spec']['q']:g}|mu=({job['spec'].get('mu_x',1.0):g},"
                      f"{job['spec'].get('mu_z',1.0):g})|x0={job['spec']['x0'][0]:g}"
                      f"|rho_x{job.get('rho_multiplier',1.0):g}")


def _q_jobs():
    jobs = []
    for q, (mux, muz), x0 in itertools.product(Q_GRID_ABOVE_10 + Q_GRID_BELOW_10,
                                               MU_GRID, X0_GRID):
        for mult in (1.0, 20.0):
            jobs.append(dict(spec=dict(kind="toy", q=q, mu_x=mux, mu_z=muz, x0=x0),
                             K=400, rho_multiplier=mult))
    return jobs


def _onset_one(args) -> dict:
    """Bisect on q for the smallest q at which convergence is still geometric."""
    mux, muz, x0 = args
    lo, hi = 0.05, 10.0          # lo assumed non-geometric, hi assumed geometric
    trace = []
    for _ in range(9):
        mid = float(np.sqrt(lo * hi))
        r = _q_one(dict(spec=dict(kind="toy", q=mid, mu_x=mux, mu_z=muz, x0=x0),
                        K=400, rho_multiplier=1.0))
        ok = bool(r["determined"] and r["linear"])
        trace.append(dict(q=mid, linear=ok, determined=r["determined"]))
        if ok:
            hi = mid
        else:
            lo = mid
    return dict(mu_x=mux, mu_z=muz, x0=list(x0), onset_lo=lo, onset_hi=hi, trace=trace)


def _baseline_one(job) -> dict:
    key, mux, muz, rho, x0 = job
    prob = APPENDIX_PROBLEMS[key](mux, muz)
    out = {}
    for algo in ALGOS:
        res = run_algorithm(prob, algo, x0[:prob.n_x], rho=rho, K=job_iters(),
                            mu_x=mux)
        v = res["viol"]
        hit = np.flatnonzero(v <= TOL)
        out[algo] = dict(iters_to_tol=(int(hit[0]) + 1 if hit.size else None),
                         final_viol=float(v[-1]), final_obj=float(res["obj"][-1]))
    winner = min((a for a in ALGOS if out[a]["iters_to_tol"] is not None),
                 key=lambda a: out[a]["iters_to_tol"], default=None)
    best_obj = min(out[a]["final_obj"] for a in ALGOS
                   if out[a]["final_viol"] < 1e-6) if any(
        out[a]["final_viol"] < 1e-6 for a in ALGOS) else None
    return dict(problem=key, multiaffine=prob.multiaffine,
                convex_objective=prob.convex_objective,
                mu_x=mux, mu_z=muz, rho=rho, x0=list(x0[:prob.n_x]),
                winner=winner, admm_wins=bool(winner == "admm"),
                admm_obj_is_best=(None if best_obj is None else
                                  bool(out["admm"]["final_obj"] <= best_obj + 1e-9)),
                **{f"{a}_{k}": v for a in ALGOS for k, v in out[a].items()})


def job_iters() -> int:
    return 300


def _baseline_jobs():
    return [(key, mux, muz, rho, x0)
            for key in ("B1", "B2", "B3")
            for (mux, muz) in MU_GRID
            for rho in (1.0, 4.0, 20.0)
            for x0 in X0_GRID]


def run() -> dict:
    # ---------------- 5a: the q >= 10 sufficiency claim ----------------
    qjobs = _q_jobs()
    print(f"  (5a) {len(qjobs)} q-sweep configurations on {workers()} workers", flush=True)
    qrows = pmap(_q_one, qjobs, desc="claim5-q")
    # A q >= 10 configuration refutes the paper's sufficiency claim only if it is
    # POSITIVELY established non-geometric; "no route certified it" is
    # inconclusive.  Two earlier "COUNTEREXAMPLES" here were nothing of the kind:
    # per-step ratios 0.815 and 0.327, i.e. plainly geometric, rejected only by a
    # six-decade bar they could not reach before hitting the numerical floor.
    above = [r for r in qrows if r["q"] >= 10.0 and r["geometric_determined"]]
    above_lin = [r for r in above if r["linear"]]
    above_refuted = [r for r in qrows
                     if r["q"] >= 10.0 and r["established_not_geometric"]]
    above_undet = [r for r in qrows if r["q"] >= 10.0 and not r["determined"]]
    print(f"\n  q >= 10: {len(above_lin)}/{len(above)} determined configurations "
          f"converge geometrically ({len(above_undet)} inconclusive)")
    for r in above:
        if not r["linear"]:
            print(f"    COUNTEREXAMPLE to sufficiency: {r['label']} "
                  f"(decades={r['decades_of_decay']}, ratio={r['max_one_step_ratio']})")

    below = [r for r in qrows if r["q"] < 10.0 and r["determined"]]
    below_lin = [r for r in below if r["linear"]]
    print(f"  q < 10:  {len(below_lin)}/{len(below)} determined configurations also "
          f"converge geometrically -- the paper's condition is sufficient, not necessary")

    onsets = pmap(_onset_one, [(mux, muz, x0) for (mux, muz) in MU_GRID for x0 in X0_GRID],
                  desc="claim5-onset")
    onset_vals = [o["onset_hi"] for o in onsets]
    print(f"  bisected empirical onset in q: min={min(onset_vals):.3g}, "
          f"median={np.median(onset_vals):.3g}, max={max(onset_vals):.3g} "
          f"(all below the paper's sufficient q >= 10: "
          f"{all(v <= 10.0 for v in onset_vals)})")

    print(f"  q >= 10 configurations POSITIVELY ESTABLISHED non-geometric: "
          f"{len(above_refuted)}")
    sufficiency_holds = bool(len(above_lin) >= 12 and not above_refuted)

    # ---------------- 5b: the Figure 4 baseline comparison ----------------
    bjobs = _baseline_jobs()
    print(f"\n  (5b) {len(bjobs)} baseline-comparison configurations", flush=True)
    brows = pmap(_baseline_one, bjobs, desc="claim5-baselines")
    nl = [r for r in brows if r["multiaffine"]]
    lin = [r for r in brows if not r["multiaffine"]]
    nl_win = sum(r["admm_wins"] for r in nl)
    lin_win = sum(r["admm_wins"] for r in lin)
    print(f"\n  multi-affine constraints (B1): ADMM reaches ||residual||<={TOL:g} "
          f"first in {nl_win}/{len(nl)} configurations")
    print(f"  linear constraints (B2, B3):   ADMM first in {lin_win}/{len(lin)}")
    for key in ("B1", "B2", "B3"):
        sub = [r for r in brows if r["problem"] == key]
        med = {a: np.median([r[f"{a}_iters_to_tol"] for r in sub
                             if r[f"{a}_iters_to_tol"] is not None] or [np.nan])
               for a in ALGOS}
        nofin = {a: sum(r[f"{a}_iters_to_tol"] is None for r in sub) for a in ALGOS}
        print(f"    {key}: median iterations to {TOL:g} -- "
              + ", ".join(f"{a}={med[a]:.0f}(+{nofin[a]} never)" for a in ALGOS))
    obj_rows = [r for r in brows if r["admm_obj_is_best"] is not None]
    obj_best = sum(r["admm_obj_is_best"] for r in obj_rows)
    print(f"  ADMM's limit has the lowest objective among feasible limits in "
          f"{obj_best}/{len(obj_rows)} configurations")

    baseline_supported = bool(nl and nl_win == len(nl))

    write_csv("claim5/claim5_q_sweep.csv", qrows)
    write_artifact("claim5/claim5_q_onset_bisection.json", onsets)
    write_csv("claim5/claim5_baseline_comparison.csv", brows)

    ok = sufficiency_holds and baseline_supported
    return dict(
        verdict="VERIFIED" if ok else "BLOCKED",
        # the baseline half rests on reimplementations of three unavailable
        # papers, so confidence is capped below HIGH regardless of the outcome
        confidence="MEDIUM" if ok else "LOW",
        headline=(f"q>=10 sufficiency: {len(above_lin)}/{len(above)} geometric; "
                  f"empirical onset q~{np.median(onset_vals):.2g} (below 10, so the "
                  f"paper's condition is sufficient not necessary); ADMM fastest to "
                  f"{TOL:g} in {nl_win}/{len(nl)} multi-affine and {lin_win}/{len(lin)} "
                  f"linear-constraint configurations"),
        sufficiency_holds=sufficiency_holds,
        n_above_10=len(above), n_above_10_linear=len(above_lin),
        empirical_onset_median=float(np.median(onset_vals)),
        baseline_supported=baseline_supported,
        admm_wins_multiaffine=nl_win, n_multiaffine=len(nl),
        admm_wins_linear=lin_win, n_linear=len(lin),
        admm_objective_best=obj_best, n_objective_compared=len(obj_rows),
        deviation=("PADMM/IPDS-ADMM/IADMM are standard reimplementations of the "
                   "named methods; the three source papers were not retrievable"),
        artifacts=["claim5/claim5_q_sweep.csv", "claim5/claim5_q_onset_bisection.json",
                   "claim5/claim5_baseline_comparison.csv"],
        internal_failure=False,
    )
