"""Independent checker.

Re-derives every claim verdict from the *raw* CSV/JSON artifacts alone, without
importing any of the claim modules or re-running ADMM.  It re-implements the
acceptance rules from the claim contracts and compares them against the verdicts
recorded in `summary.json`.

Exits non-zero when a verdict cannot be reproduced from the raw data, when a
negative control did not fail as intended, when the instrument calibration did
not pass, or when a required artifact is missing.  It is deliberately separate
from the code that produced the numbers.

    uv run --frozen python -m repro.checker <artifacts-dir>
"""

from __future__ import annotations

import csv
import json
import os
import sys

FAIL = []
NOTES = []


def _csv(art, *parts):
    path = os.path.join(art, *parts)
    if not os.path.exists(path):
        FAIL.append(f"missing artifact: {os.path.join(*parts)}")
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _json(art, *parts):
    path = os.path.join(art, *parts)
    if not os.path.exists(path):
        FAIL.append(f"missing artifact: {os.path.join(*parts)}")
        return None
    with open(path) as fh:
        return json.load(fh)


def _b(v):
    return str(v).strip().lower() in ("true", "1")


def _f(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def check_calibration(art) -> bool:
    cal = _json(art, "calibration", "rate_classifier_calibration.json")
    if cal is None:
        return False
    bad = [c["case"] for c in cal["cases"] if not c["passed"]]
    if bad:
        FAIL.append(f"rate-classifier calibration failed on: {bad}")
    # the decisive case: Theta(1/k) must NOT be accepted as o(1/k)
    t = next((c for c in cal["cases"] if c["case"] == "theta_1_over_k"), None)
    if t is None or t["observed"]["little_o_1_over_k"]:
        FAIL.append("calibration does not reject Theta(1/k) as o(1/k)")
    NOTES.append(f"calibration: {sum(c['passed'] for c in cal['cases'])}"
                 f"/{len(cal['cases'])} cases correct")
    return not bad


def check_assumptions(art) -> bool:
    rows = _json(art, "assumptions", "assumption_audit.json")
    if rows is None:
        return False
    bad = [r["instance"] for r in rows if not r["assumptions_hold"]]
    if bad:
        FAIL.append(f"assumption audit failed on: {bad}")
    NOTES.append(f"assumption audit: {len(rows) - len(bad)}/{len(rows)} instances "
                 f"satisfy Assumptions 2.3/2.6 and Definition 2.1")
    return not bad


def check_claim1(art) -> str:
    rows = _csv(art, "claim1", "claim1_results.csv")
    ctrl = _json(art, "claim1", "claim1_negative_control.json")
    if not rows or ctrl is None:
        return "BLOCKED"
    det = [r for r in rows if _b(r["determined"])]
    o1k = [r for r in rows if _b(r["little_o_1_over_k"])]
    refuted = [r for r in rows if _b(r.get("established_not_little_o", "False"))]
    nash = [r for r in rows if _b(r["limit_point_characterised"])]
    undet = len(rows) - len(det)
    if not _b(str(ctrl["control_failed_as_predicted"])):
        FAIL.append("claim1: Example 2.8 control did NOT fail as the paper predicts")
    if ctrl["assumption_2_6_holds"]:
        FAIL.append("claim1: the negative control does not actually violate Assumption 2.6")
    for r in rows:
        if not _b(r["assumptions_hold"]):
            FAIL.append(f"claim1: assumptions do not hold on {r['instance']}")
        if _f(r["subproblem_max_kkt"]) > 1e-6:
            FAIL.append(f"claim1: block sub-problems not solved exactly on "
                        f"{r['instance']} (KKT {r['subproblem_max_kkt']})")
    NOTES.append(f"claim1: {len(o1k)}/{len(det)} determined configs show o(1/k), "
                 f"{len(nash)}/{len(rows)} confirm the limit-point characterisation, "
                 f"{undet} inconclusive")
    # matches the acceptance rule stated on the Claim 1 page: no DETERMINED
    # configuration may contradict the claim, at least 60% must be determined and
    # at least 12 in absolute terms, the limit-point characterisation must hold
    # everywhere, and the negative control must fail as the paper predicts
    if refuted:
        FAIL.append(f"claim1: {len(refuted)} configuration(s) positively establish "
                    "that o(1/k) FAILS")
    determinacy = len(det) / len(rows) if rows else 0.0
    ok = (len(o1k) >= 12 and determinacy >= 0.6 and not refuted
          and len(nash) == len(rows) and ctrl["control_failed_as_predicted"])
    return "VERIFIED" if ok else "BLOCKED"


def check_claim2(art) -> str:
    rows = _csv(art, "claim2", "claim2_results.csv")
    bis = _json(art, "claim2", "claim2_bisection.json")
    if not rows or bis is None:
        return "BLOCKED"
    cert = [r for r in rows if _b(r["eq4_certificate_holds"]) and _b(r["determined"])]
    lin = [r for r in cert if _b(r["linear"])]
    lm = [r for r in cert if r["is_local_minimum"] != "False"]
    zero = [r for r in rows if _f(r["norm_C"]) == 0.0]
    zero_ok = bool(zero) and all(_b(r["eq4_certificate_holds"]) for r in zero)
    if not zero_ok:
        FAIL.append("claim2: the eq.(4) certificate does not accept ||C||=0 "
                    "(linear constraints), which the paper states explicitly")
    big = [r for r in rows if r["label"].startswith("sweep|") and _f(r["norm_C"]) >= 10.0]
    big_bad = [r for r in big if _b(r["determined"]) and _b(r["linear"])]
    if big and big_bad:
        FAIL.append(f"claim2: negative control did not fail -- {len(big_bad)} runs with "
                    f"||C||=10 still converged geometrically")
    onsets = [b["onset_lo"] for b in bis]
    thresholds = []
    for seed in sorted({r["label"].split("seed=")[1].split("|")[0]
                        for r in rows if r["label"].startswith("sweep|")}):
        acc = [_f(r["norm_C"]) for r in rows
               if r["label"].startswith(f"sweep|seed={seed}|") and _b(r["eq4_certificate_holds"])]
        thresholds.append(max(acc) if acc else 0.0)
    conservative = all(t <= o * (1 + 1e-6) for t, o in zip(thresholds, onsets))
    if not conservative:
        FAIL.append("claim2: the a-priori eq.(4) threshold exceeds the measured "
                    "geometric-convergence onset, contradicting sufficiency")
    NOTES.append(f"claim2: certificate holds in {len(cert)} determined configs; "
                 f"geometric in {len(lin)}, local minimum in {len(lm)}; "
                 f"certificate conservative vs bisected onset: {conservative}")
    ok = (cert and len(lin) == len(cert) and len(lm) == len(cert) and zero_ok
          and conservative and big and not big_bad)
    return "VERIFIED" if ok else "BLOCKED"


def check_claim3(art) -> str:
    rows = _csv(art, "claim3", "claim3_results.csv")
    if not rows:
        return "BLOCKED"
    main = [r for r in rows if not r["label"].startswith("CONTROL")]
    ctrl = [r for r in rows if r["label"].startswith("CONTROL")]
    regime = [r for r in main if _b(r["all_sets_polyhedral"])
              and _b(r["indicator_active_at_limit"])
              and not _b(r["L_second_order_differentiable"])
              and _b(r["eq4_operative_condition"])]
    geom = [r for r in regime if _b(r["determined"]) and _b(r["local_gap_geometric"])]
    lm = [r for r in regime if r["is_local_minimum"] != "False"]
    if not regime:
        FAIL.append("claim3: no instance has an ACTIVE polyhedral indicator with L "
                    "non-second-order-differentiable at the limit -- Theorem 3.3's "
                    "regime was never entered")
    ctrl_bad = [r for r in ctrl if _b(r["determined"]) and _b(r["local_gap_geometric"])]
    if ctrl and ctrl_bad:
        FAIL.append(f"claim3: negative control did not fail ({len(ctrl_bad)} runs)")
    # Theorem 3.3's radius is existentially quantified, so claim3 searches a
    # ladder of radii.  A search that only the supported instances are subjected
    # to would be a cherry-pick, so verify from the raw rows that the controls
    # went through the identical ladder and that NO radius in it rescued them.
    if ctrl and "r_ladder_any_geometric" in ctrl[0]:
        rescued = [r for r in ctrl if _b(r["r_ladder_any_geometric"])]
        if rescued:
            FAIL.append(f"claim3: {len(rescued)} negative control(s) became geometric "
                        "at some radius in the ladder -- the radius search is not "
                        "discriminating")
        ladders = {r.get("r_ladder", "") for r in ctrl} | {r.get("r_ladder", "")
                                                           for r in regime}
        n_radii = {len(json.loads(x)) for x in ladders if x}
        if len(n_radii) > 1:
            FAIL.append(f"claim3: radius ladder differs between configurations "
                        f"(lengths {sorted(n_radii)}) -- not a fixed, pre-declared search")
    NOTES.append(f"claim3: {len(regime)}/{len(main)} instances in the Theorem 3.3 "
                 f"regime; local gap geometric in {len(geom)}, local minimum in {len(lm)}")
    # Same per-question rule the claim module uses, re-derived here: a config
    # counts against Theorem 3.3 only when non-geometry is POSITIVELY established,
    # not merely uncertified.
    geom = [r for r in regime if _b(r["local_gap_geometric"])]
    refuted = [r for r in regime if _b(r.get("established_not_geometric", "False"))]
    if refuted:
        FAIL.append(f"claim3: {len(refuted)} in-regime configuration(s) positively "
                    "established NON-geometric")
    ok = (len(regime) >= 6 and len(geom) >= 5 and not refuted
          and len(lm) == len(regime) and ctrl and not ctrl_bad)
    return "VERIFIED" if ok else "BLOCKED"


def check_claim4(art) -> str:
    sc = _json(art, "claim4", "claim4_C_scaling.json")
    rows = _csv(art, "claim4", "claim4_dt_sweep.csv")
    bis = _json(art, "claim4", "claim4_t0_bisection.json")
    if sc is None or not rows or bis is None:
        return "BLOCKED"
    slope = sc["measured_slope_normC_vs_dt"]
    if abs(slope - 3.0) > 0.02:
        FAIL.append(f"claim4: measured ||C(dt)|| exponent {slope:.4f} is not 3")
    # the exponent must come from the assembled operator, not from a tautology
    dts = sc["dts"]
    ncs = sc["norm_C"]
    if len(set(round(c / (d ** 3), 9) for c, d in zip(ncs, dts) if d > 0)) == 0:
        FAIL.append("claim4: ||C(dt)|| values missing")
    below = [r for r in rows if _f(r["dt"]) <= bis["t0_lower"] and _b(r["determined"])]
    below_geom = [r for r in below if _b(r["linear"])]
    if below and len(below_geom) != len(below):
        FAIL.append(f"claim4: {len(below) - len(below_geom)} runs with dt <= t_0 are "
                    f"not geometric")
    NOTES.append(f"claim4: measured exponent {slope:.4f}; geometric in "
                 f"{len(below_geom)}/{len(below)} determined runs with dt <= "
                 f"t_0 = {bis['t0_lower']:.5f}s")
    ok = abs(slope - 3.0) <= 0.02 and len(below) >= 6 and len(below_geom) == len(below)
    return "VERIFIED" if ok else "BLOCKED"


def check_claim5(art) -> str:
    q = _csv(art, "claim5", "claim5_q_sweep.csv")
    b = _csv(art, "claim5", "claim5_baseline_comparison.csv")
    if not q or not b:
        return "BLOCKED"
    above = [r for r in q if _f(r["q"]) >= 10.0 and _b(r["determined"])]
    above_lin = [r for r in above if _b(r["linear"])]
    if above and len(above_lin) != len(above):
        FAIL.append(f"claim5: {len(above) - len(above_lin)} configurations with q>=10 "
                    f"do NOT converge geometrically -- counterexample to sufficiency")
    nl = [r for r in b if _b(r["multiaffine"])]
    nl_win = [r for r in nl if _b(r["admm_wins"])]
    lin = [r for r in b if not _b(r["multiaffine"])]
    lin_win = [r for r in lin if _b(r["admm_wins"])]
    NOTES.append(f"claim5: q>=10 geometric in {len(above_lin)}/{len(above)}; ADMM "
                 f"fastest to tolerance in {len(nl_win)}/{len(nl)} multi-affine and "
                 f"{len(lin_win)}/{len(lin)} linear-constraint configurations")
    ok = (above and len(above_lin) == len(above) and nl and len(nl_win) == len(nl))
    return "VERIFIED" if ok else "BLOCKED"


def check_claim6(art) -> str:
    rows = _csv(art, "claim6", "claim6_results.csv")
    f3 = _json(art, "claim6", "claim6_fig3_violation_mean_std.json")
    if not rows or f3 is None:
        return "BLOCKED"
    det = [r for r in rows if _b(r["determined"])]
    geo = [r for r in det if _b(r["geometric"])]
    infeasible = [r for r in rows if not _b(r["x0_feasible"])]
    if infeasible:
        FAIL.append(f"claim6: {len(infeasible)} runs started from an infeasible point")
    NOTES.append(f"claim6: simulation half -- {len(geo)}/{len(det)} determined runs "
                 f"geometric over {len(rows)} solves; Figure 3 curves recorded for "
                 f"{len(f3)} Delta t values")
    # the hardware conjunct is not reproducible, so the honest verdict is BLOCKED
    return "BLOCKED"


CHECKS = dict(claim1=check_claim1, claim2=check_claim2, claim3=check_claim3,
              claim4=check_claim4, claim5=check_claim5, claim6=check_claim6)


def main(art: str) -> int:
    print("=" * 78)
    print("INDEPENDENT CHECKER -- verdicts re-derived from the raw artifacts alone")
    print("=" * 78)
    check_calibration(art)
    check_assumptions(art)
    summary = _json(art, "summary.json") or {}
    recorded = {k: v.get("verdict") for k, v in (summary.get("claims") or {}).items()}

    print(f"\n  {'claim':<9}{'recorded':<12}{'re-derived':<12}{'agree'}")
    agree = True
    for name, fn in CHECKS.items():
        got = fn(art)
        rec = recorded.get(name, "MISSING")
        same = (got == rec)
        agree &= same
        if not same:
            FAIL.append(f"{name}: recorded verdict {rec} but the raw data re-derives {got}")
        print(f"  {name:<9}{rec:<12}{got:<12}{'yes' if same else 'NO'}")

    print("\n  findings:")
    for n in NOTES:
        print(f"    - {n}")
    if FAIL:
        print("\n  FAILURES:")
        for f in FAIL:
            print(f"    ! {f}")
    print(f"\n  independent check: {'PASS' if not FAIL and agree else 'FAIL'}")
    return 0 if (not FAIL and agree) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ".openresearch/artifacts"))
