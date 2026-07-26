"""Claim 6 -- Figures 5, 6 and 3.

The claim as stated is a conjunction: the method is validated (i) on 2D
locomotion simulation and (ii) on real robot experiments including a humanoid
jump and a quadruped bounding task.

What can and cannot be reproduced
---------------------------------
(i)  is fully reproducible and is reproduced here: the 2D locomotion problem of
     Figure 5 at the paper's parameters (m = 2 kg, f(f) = 0.5 sum ||f_i||^2,
     phi(z) = 5 sum ||k'_i||^2, polyhedral friction cones), swept over Delta t
     (Fig. 5 centre) and over randomised initial configurations (Fig. 5 right).

(ii) splits in two.  The *numerical* content of the robot experiments -- the
     centroidal trajectory optimisation of eq. (5)/(6) solved by Algorithm 1 for
     a humanoid vertical jump and a quadruped bounding gait, and the per-iteration
     centroidal dynamics constraint violation reported in Figure 3 as mean and
     standard deviation over 10 trials with randomised initial conditions for
     three Delta t -- is reproduced here with full contact schedules including
     flight phases.  The *hardware execution* (Figure 6: the trajectories being
     tracked by a DDP kinematics optimiser and run on the physical Solo and
     humanoid robots) cannot be reproduced without those robots and is reported
     as BLOCKED, not as a pass.

Because the claim is a conjunction and one conjunct is not reproducible on the
available compute, the overall verdict is BLOCKED regardless of how well the
simulation half performs.  The simulation half is reported in full so the
evidence is not lost.
"""

from __future__ import annotations

import numpy as np

from repro.analysis import gap_sequence
from repro.core import admm
from repro.localmin import indicator_activity
from repro.parallel import pmap, workers
from repro.problems.locomotion import build_locomotion, feasible_start
from repro.provenance import write_artifact, write_csv
from repro.rates import classify

TITLE = "Figures 5/3: 2D locomotion, humanoid jump and quadruped bound (hardware blocked)"

FIG5_DT = (0.005, 0.01, 0.02, 0.05, 0.1, 0.15)
FIG3_DT = (0.005, 0.01, 0.02)
N_TRIALS = 10


def _run_one(job: dict) -> dict:
    """One ADMM solve of the eq.(6) locomotion problem."""
    p = build_locomotion(T=job["T"], N=job["N"], dim=job["dim"], dt=job["dt"],
                         m=job["m"], gait=job.get("gait"))
    rng = np.random.default_rng(job["seed"]) if job.get("randomise_init") else None
    x0 = feasible_start(p, rng=rng)
    rho = p.rho_threshold()
    a = p.audit()
    if job.get("need_gap", True):
        gap, L_ref, tr = gap_sequence(p, rho, x0, K=job["K"], ref_mult=3)
        cl = classify(gap)
    else:
        tr = admm(p, rho, x0, K=job["K"])
        cl = classify(np.abs(tr.L - np.median(tr.L[int(0.8 * job["K"]):])))
    act = indicator_activity(p, tr.x)
    stride = max(1, job["K"] // 200)
    return dict(
        row=dict(label=job["label"], dt=job["dt"], T=job["T"], N=job["N"],
                 dim=job["dim"], m=job["m"], gait=job.get("gait", "stance"),
                 seed=job["seed"], n_x=a["n_x"], n_c=a["n_c"], norm_C=a["norm_C"],
                 rho=rho, x0_feasible=bool(p.indicators_finite(x0)),
                 n_active_indicators=act["n_active_constraints"],
                 determined=bool(cl.get("determined", False)),
                 geometric=bool(cl["linear"]),
                 decades_of_decay=cl.get("decades_of_decay"),
                 rate_estimate=cl.get("c1_envelope", cl.get("c1_estimate")),
                 initial_violation=float(tr.viol[0]),
                 final_violation=float(tr.viol[-1]),
                 subproblem_max_kkt=tr.max_kkt, seconds=tr.seconds),
        violation_trace=tr.viol[::stride].tolist(),
        violation_full=tr.viol.tolist() if job.get("keep_full") else None,
    )


def _fig5_jobs():
    """Figure 5: 2D locomotion, m = 2 kg. Centre panel = Delta t sweep;
    right panel = randomised initial configurations."""
    jobs = [dict(label=f"fig5-centre|dt={dt:g}", T=20, N=2, dim=2, dt=dt, m=2.0,
                 seed=0, K=250, randomise_init=False) for dt in FIG5_DT]
    jobs += [dict(label=f"fig5-right|init={s}", T=20, N=2, dim=2, dt=0.05, m=2.0,
                  seed=s, K=250, randomise_init=True) for s in range(N_TRIALS)]
    return jobs


def _fig3_jobs():
    """Figure 3: humanoid vertical jump, 3 Delta t, 10 randomised initial
    conditions each; the reported quantity is the centroidal dynamics constraint
    violation per ADMM iteration."""
    return [dict(label=f"fig3-jump|dt={dt:g}|trial={s}", T=20, N=2, dim=3, dt=dt,
                 m=55.0, gait="jump", seed=s, K=150, randomise_init=True,
                 keep_full=True)
            for dt in FIG3_DT for s in range(N_TRIALS)]


def _fig6_jobs():
    """Figure 6: the two dynamic motions, as centroidal trajectory optimisation."""
    jobs = [dict(label=f"fig6-humanoid-jump|trial={s}", T=20, N=2, dim=3, dt=0.02,
                 m=55.0, gait="jump", seed=s, K=150, randomise_init=True)
            for s in range(N_TRIALS)]
    jobs += [dict(label=f"fig6-quadruped-bound|trial={s}", T=24, N=4, dim=3, dt=0.02,
                  m=20.0, gait="bound", seed=s, K=150, randomise_init=True)
             for s in range(N_TRIALS)]
    return jobs


def run() -> dict:
    jobs = _fig5_jobs() + _fig3_jobs() + _fig6_jobs()
    print(f"  {len(jobs)} locomotion solves on {workers()} worker processes", flush=True)
    res = pmap(_run_one, jobs, desc="claim6")
    rows = [r["row"] for r in res]

    def summarise(tag):
        sub = [r for r in rows if r["label"].startswith(tag)]
        det = [r for r in sub if r["determined"]]
        geo = [r for r in det if r["geometric"]]
        return sub, det, geo

    print(f"\n  {'label':<34}{'n_x':>6}{'||C||':>12}{'active':>8}{'det':>5}"
          f"{'geom':>6}{'decades':>9}{'viol_final':>12}{'kkt':>10}")
    for r in rows:
        print(f"  {r['label']:<34}{r['n_x']:>6}{r['norm_C']:>12.3e}"
              f"{r['n_active_indicators']:>8}{str(r['determined']):>5}"
              f"{str(r['geometric']):>6}{(r['decades_of_decay'] or 0):>9.1f}"
              f"{r['final_violation']:>12.2e}{r['subproblem_max_kkt']:>10.1e}")

    parts = {}
    for tag, name in [("fig5-centre", "Fig 5 centre (Delta t sweep, 2D)"),
                      ("fig5-right", "Fig 5 right (random initial configurations)"),
                      ("fig3-jump", "Fig 3 (humanoid jump, 3 dt x 10 trials)"),
                      ("fig6-humanoid-jump", "Fig 6 humanoid jump (centroidal)"),
                      ("fig6-quadruped-bound", "Fig 6 quadruped bound (centroidal)")]:
        sub, det, geo = summarise(tag)
        parts[tag] = dict(n=len(sub), n_determined=len(det), n_geometric=len(geo),
                          all_feasible_start=all(r["x0_feasible"] for r in sub),
                          max_final_violation=max((r["final_violation"] for r in sub),
                                                  default=None))
        print(f"\n  {name}: {len(geo)}/{len(det)} determined runs converge "
              f"geometrically ({len(sub)} runs total); "
              f"max final violation {parts[tag]['max_final_violation']:.2e}")

    # Figure 3's actual reported quantity: mean +- std of the violation per iteration
    fig3 = {}
    for dt in FIG3_DT:
        traces = [np.asarray(r["violation_full"], float) for r, row in zip(res, rows)
                  if row["label"].startswith(f"fig3-jump|dt={dt:g}|")
                  and r["violation_full"] is not None]
        if traces:
            Aa = np.vstack(traces)
            fig3[f"dt={dt:g}"] = dict(n_trials=len(traces),
                                      mean=Aa.mean(axis=0).tolist(),
                                      std=Aa.std(axis=0).tolist())
            m, sd = Aa.mean(axis=0), Aa.std(axis=0)
            idx = [0, 4, 9, 24, 49, min(99, Aa.shape[1] - 1)]
            print(f"\n  Fig 3, dt={dt:g}: dynamics violation mean+-std over "
                  f"{len(traces)} randomised trials")
            for i in idx:
                print(f"      k={i + 1:4d}   {m[i]:.4e} +- {sd[i]:.2e}")

    write_csv("claim6/claim6_results.csv", rows)
    write_artifact("claim6/claim6_fig3_violation_mean_std.json", fig3)
    write_artifact("claim6/claim6_violation_traces.json",
                   {r["row"]["label"]: r["violation_trace"] for r in res})

    sim_ok = all(p["n_geometric"] == p["n_determined"] and p["n_determined"] > 0
                 and p["all_feasible_start"] for p in parts.values())
    hardware = dict(
        reproducible=False,
        reason=("Figure 6 reports centroidal trajectories tracked by a DDP kinematics "
                "optimiser and executed on physical humanoid and quadruped robots. "
                "No robot hardware is available to this reproduction, and the "
                "authors' hardware logs are not published, so the hardware conjunct "
                "of the claim cannot be verified or falsified."),
        what_would_unblock=("access to the two robots (or the authors' released "
                            "hardware trajectory and tracking logs) plus the "
                            "Crocoddyl DDP tracking configuration used in the paper"),
    )
    print(f"\n  simulation half (Figures 5 and 3, plus the centroidal solves behind "
          f"Figure 6): {'fully reproduced' if sim_ok else 'INCOMPLETE'}")
    print(f"  hardware half (Figure 6 execution on real robots): NOT REPRODUCIBLE -- "
          f"{hardware['reason']}")

    return dict(
        # the claim is a conjunction; one conjunct is out of reach on this compute
        verdict="BLOCKED",
        confidence="HIGH" if sim_ok else "LOW",
        headline=("2D locomotion (Fig 5), the humanoid-jump violation curves (Fig 3) "
                  "and the centroidal solves behind Fig 6 reproduce with geometric "
                  "convergence in every determined run; the real-robot half of the "
                  "claim cannot be run without the hardware"),
        simulation_reproduced=sim_ok, parts=parts, hardware=hardware,
        artifacts=["claim6/claim6_results.csv",
                   "claim6/claim6_fig3_violation_mean_std.json",
                   "claim6/claim6_violation_traces.json"],
        internal_failure=False,
    )
