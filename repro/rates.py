"""Rate classification for Lagrangian gap sequences, with an explicit calibration.

The paper states two distinct asymptotic statements, and they are tested with
two distinct estimators:

  Theorem 3.1   L(x^k,z^k,w^k) - L* in o(1/k)
                <=>  k * gap_k -> 0.  Tested by (a) fitting gap_k ~ k^{-alpha}
                on a numerically valid window and requiring the *lower* end of a
                bootstrap CI on alpha to exceed 1, or (b) detecting geometric
                decay (which implies o(1/k)).

  Theorem 3.2/3.3   gap_k in O(c^{-k}), c > 1
                <=>  geometric decay.  Tested by fitting log gap_k ~ a - beta k,
                requiring beta > 0 with a CI bounded away from 0, R^2 >= 0.995,
                and the geometric model to beat the power-law model on the same
                window (Delta AIC).

`calibrate()` runs both estimators on synthetic sequences with *known* rates.
It is a genuine negative control: an estimator that always says "linear" or
always says "o(1/k)" fails it.
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20260726)

# Fixed multiplicative-noise draw used by the noisy calibration cases.  Drawn once
# from a dedicated seeded generator so the calibration is deterministic and does
# not consume RNG state shared with the bootstrap.
_CAL_NOISE = np.random.default_rng(4242).normal(0.0, 0.35, 8000)


def valid_window(gap: np.ndarray, floor: float, min_points: int = 4):
    """Indices where the gap is informative.

    The window starts just after the peak of the sequence (dropping the initial
    transient, whose length is *not* fixed a priori but read off the data) and
    ends at the last index still above the numerical floor.  This is deliberately
    independent of the total number of iterations run, so that lengthening a run
    cannot by itself change the verdict.
    """
    ok = np.isfinite(gap) & (gap > floor)
    if not ok.any():
        return np.array([], int)
    peak = int(np.nanargmax(np.where(ok, gap, -np.inf)))
    idx = np.flatnonzero(ok)
    idx = idx[idx >= peak]
    if idx.size < min_points:
        return np.array([], int)
    # every index at or after the peak that is still above the numerical floor.
    # Contiguity is deliberately NOT required: the ADMM Lagrangian gap is not
    # monotone, and truncating at the first dip would shorten the window (and so
    # change the verdict) for reasons unrelated to the asymptotic rate.
    run = np.asarray(idx, int)
    return run if run.size >= min_points else np.array([], int)


def _ols(xv, yv):
    n = xv.size
    X = np.vstack([np.ones(n), xv]).T
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ coef
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coef, r2, ss_res


def _aic(ss_res, n, kparams=2):
    ss_res = max(ss_res, 1e-300)
    return n * np.log(ss_res / n) + 2 * kparams


def _bootstrap_slope(xv, yv, n_boot=400):
    n = xv.size
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        coef, _, _ = _ols(xv[idx], yv[idx])
        slopes[i] = coef[1]
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def iters_to_tolerance(gap: np.ndarray, eps_grid=None) -> dict:
    """Second, resolution-independent estimator.

    k(eps) := first k with gap_k <= eps.
      * geometric decay gap_k ~ c^{-k}  =>  k(eps) is *affine* in log(1/eps);
      * power decay gap_k ~ k^{-alpha}  =>  k(eps) ~ eps^{-1/alpha}, i.e. affine
        in log(1/eps) only after a further log, so log k(eps) is affine instead.

    Comparing the two fits discriminates the regimes even when the run reaches
    machine precision in a handful of iterations, where a log-gap regression has
    too few points to be trustworthy.
    """
    gap = np.asarray(gap, float)
    if eps_grid is None:
        top = float(np.nanmax(gap[np.isfinite(gap)])) if np.isfinite(gap).any() else 1.0
        eps_grid = top * np.logspace(-1, -12, 12)
    ks, es = [], []
    for eps in eps_grid:
        hit = np.flatnonzero(np.isfinite(gap) & (gap <= eps))
        if hit.size:
            ks.append(float(hit[0] + 1))
            es.append(float(eps))
    out = dict(n_levels=len(ks), eps=es, k_eps=ks)
    if len(ks) < 4:
        out.update(k_eps_linear=False, reason="fewer than 4 tolerance levels reached")
        return out
    ks_a, es_a = np.asarray(ks), np.asarray(es)
    L = np.log10(es_a[0] / es_a)                    # log10 of tolerance decades
    _, r2_affine, _ = _ols(L, ks_a)                 # geometric signature
    _, r2_log, _ = _ols(L, np.log(np.maximum(ks_a, 1)))  # power-law signature
    span = float(ks_a[-1] / max(ks_a[0], 1.0))
    out.update(r2_k_vs_logeps=float(r2_affine), r2_logk_vs_logeps=float(r2_log),
               k_growth_factor=span,
               k_eps_linear=bool(r2_affine >= 0.98 and r2_affine >= r2_log - 1e-9
                                 and span < 50.0))
    return out


def classify(gap: np.ndarray, floor: float | None = None, n_boot: int = 400) -> dict:
    """Classify a non-negative gap sequence.

    Returns a dict with `linear` (Theorem 3.2/3.3 style) and `little_o_1_over_k`
    (Theorem 3.1 style) booleans plus all supporting statistics.
    """
    gap = np.asarray(gap, float)
    gap = np.where(np.isfinite(gap), gap, np.nan)
    pos = gap[np.isfinite(gap) & (gap > 0)]
    if floor is None:
        # Numerical floor.  Two regimes have to be told apart, and conflating
        # them is a live failure mode:
        #
        #   (a) the sequence bottoms out on a plateau of round-off / solver-
        #       tolerance noise.  Those points carry no rate information and must
        #       be excluded, so the floor is set just above the plateau.
        #   (b) the sequence is STILL DECAYING when the run ends.  Then its
        #       smallest value is a genuine measurement, not noise.
        #
        # An earlier version used `pos.min() * 1e3` unconditionally, which is
        # only valid in regime (a).  In regime (b) it puts the floor three
        # decades ABOVE the last real value - and, on a sequence spanning under
        # three decades, above every point in it, emptying the window and
        # silently turning a converging run into "no verdict".
        #
        # So the plateau term is applied only when a plateau is actually
        # detected: the tail flat to within a factor of two.  Otherwise the floor
        # is just double-precision resolution of the sequence scale.
        scale = float(np.nanmax(np.abs(gap))) if pos.size else 1.0
        floor = scale * 1e-13
        if pos.size >= 12:
            # A plateau is a suffix that has STOPPED DECAYING: under half a decade
            # of spread from there to the end.  Testing "stopped decaying" rather
            # than "is flat to within 2x" is what separates a round-off shelf from
            # a sequence still on its way down.
            #
            # Take the LONGEST such suffix, not a fixed-size tail.  A shelf is
            # never flat and often drifts downward in steps (2.3e-13 early,
            # 1.1e-13 late), so a floor derived from the last few samples sits
            # BELOW the shelf's own upper step and leaves those points in the
            # window -- which is what flattened R^2 to 0.64 and pinned the
            # one-step ratio at 1.0 on runs that had contracted 25x per iteration
            # for eight straight iterations before flooring.
            #
            # The suffix must also be long (>= a quarter of the sequence) for a
            # plateau to be declared, so that a slowly decaying sequence -- whose
            # every SHORT suffix looks flat -- is never truncated.
            smax = np.maximum.accumulate(pos[::-1])[::-1]
            smin = np.minimum.accumulate(pos[::-1])[::-1]
            spread = np.log10(smax / np.maximum(smin, 1e-300))
            need = max(8, pos.size // 4)
            cand = np.flatnonzero((spread < 0.5) & (pos.size - np.arange(pos.size) >= need))
            if cand.size:
                floor = max(floor, float(smax[cand[0]]) * 1.5)
        # An EXACT zero is unambiguous: the computed gap has reached machine
        # precision.  Every strictly positive reading at or after the first zero
        # is therefore round-off, whatever its magnitude, and belongs below the
        # floor.  This catches the very fast runs, where the whole post-collapse
        # shelf is too short to qualify as a plateau above (a run that contracts
        # 25x per iteration produces only a handful of points either side).
        zi = np.flatnonzero(np.isfinite(gap) & (gap == 0.0))
        if zi.size:
            after = gap[zi[0]:]
            ap = after[np.isfinite(after) & (after > 0)]
            if ap.size:
                floor = max(floor, float(ap.max()) * 1.5)
    win = valid_window(gap, floor)
    ke = iters_to_tolerance(gap)
    out = dict(floor=float(floor), n_window=int(win.size), k_eps=ke,
               window=[int(win[0]) + 1, int(win[-1]) + 1] if win.size else None)
    if win.size == 0:
        # too few points above the floor for a regression: fall back on the
        # resolution-independent iterations-to-tolerance estimator alone
        lin = bool(ke.get("k_eps_linear", False))
        out.update(linear=lin, little_o_1_over_k=lin, determined=lin,
                   # An empty window means the estimator saw nothing to measure,
                   # so nothing is established either way -- least of all that
                   # the sequence is NOT geometric.
                   established_not_geometric=False,
                   geometric_determined=lin,
                   little_o_by_direct_decay=False,
                   established_not_little_o=False,
                   little_o_determined=lin,
                   reason="regression window empty; k(eps) estimator used")
        return out

    k = (win + 1).astype(float)
    g = gap[win]
    decades = float(np.log10(g[0] / g[-1]))
    out["decades_of_decay"] = decades

    # --- geometric model: log gap = a - beta k ---
    (a_g, b_g), r2_geo, ss_geo = _ols(k, np.log(g))
    lo_g, hi_g = _bootstrap_slope(k, np.log(g), n_boot)
    c1 = float(np.exp(-b_g))  # gap_k ~ const * c1^{-k}
    out.update(geo_slope=float(b_g), geo_slope_ci=[lo_g, hi_g], geo_r2=float(r2_geo),
               c1_estimate=c1)

    # --- power-law model: log gap = a - alpha log k ---
    (a_p, b_p), r2_pow, ss_pow = _ols(np.log(k), np.log(g))
    lo_p, hi_p = _bootstrap_slope(np.log(k), np.log(g), n_boot)
    alpha = -float(b_p)
    out.update(power_alpha=alpha, power_alpha_ci=[-hi_p, -lo_p], power_r2=float(r2_pow))

    d_aic = _aic(ss_geo, k.size) - _aic(ss_pow, k.size)
    out["delta_aic_geo_minus_pow"] = float(d_aic)

    # --- tail-supremum envelope ------------------------------------------------
    # The theorems assert gap_k <= M c^{-k}: an ENVELOPE over the sequence, not a
    # monotone per-step contraction.  A sequence can satisfy it while going up on
    # individual steps, which the one-step-ratio certificate below cannot see and
    # a raw log-linear fit scores badly.  That matters for quantities recovered by
    # an inner numerical solve (Theorem 3.3's local gap), where the iterate-to-
    # iterate wobble is solver noise rather than a property of the algorithm.
    #
    # sup_{j>=k} gap_j is the smallest non-increasing envelope of the sequence, so
    # fitting IT is a direct test of the theorem's actual statement.  It is not a
    # weaker test: for a smooth decreasing sequence the tail-sup equals the
    # sequence, so Theta(1/k) is rejected here exactly as it is by the raw fit
    # (its envelope is still a power law, not a geometric).  calibrate() checks
    # both the noisy-geometric and noisy-power-law cases.
    tsup = np.maximum.accumulate(g[::-1])[::-1]
    tsup_decades = float(np.log10(tsup[0] / tsup[-1])) if tsup[-1] > 0 else 0.0
    (_a_t, b_t), r2_t, ss_t = _ols(k, np.log(tsup))
    lo_t, hi_t = _bootstrap_slope(k, np.log(tsup), n_boot)
    _, _, ss_tp = _ols(np.log(k), np.log(tsup))
    d_aic_t = _aic(ss_t, k.size) - _aic(ss_tp, k.size)
    out.update(tail_sup_slope=float(b_t), tail_sup_r2=float(r2_t),
               tail_sup_decades=tsup_decades,
               c1_tail_envelope=float(np.exp(-b_t)))
    out["linear_by_tail_envelope"] = bool(
        k.size >= 25
        and hi_t < 0.0             # envelope strictly decaying, 95% CI
        and r2_t >= 0.995
        and tsup_decades >= 3.0
        and d_aic_t < 0.0          # geometric beats power law on the envelope too
        and np.exp(-b_t) > 1.0
    )

    # --- geometric envelope certificate ---
    # If every one-step ratio on the window is <= cmax < 1 then, on the observed
    # range, gap_k <= gap_peak * cmax^(k - peak) -- a verified O(c^{-k}) envelope
    # with c = 1/cmax > 1.  This works when the run reaches machine precision in
    # a handful of iterations, where a regression has too few points, and it
    # necessarily fails for Theta(1/k) (whose ratios tend to 1).
    consec = np.flatnonzero(np.diff(win) == 1)   # only consecutive-iteration pairs
    ratios = (g[consec + 1] / g[consec]) if consec.size else np.array([])
    cmax = float(ratios.max()) if ratios.size else 1.0
    out.update(max_one_step_ratio=cmax, n_ratios=int(ratios.size),
               median_one_step_ratio=float(np.median(ratios)) if ratios.size else 1.0)
    # Materiality bar: three decades of genuine decay, the SAME bar used by the
    # regression path and by `determined` below.  An earlier version demanded six
    # decades here.  That was internally inconsistent and, worse, biased against
    # the claim: the envelope is a *verified per-step bound*, i.e. stronger
    # evidence than the regression it was being held to twice the standard of,
    # and the sequences it rejected were the ones that converged FASTEST (a run
    # contracting by 25x per iteration reaches the numerical floor after ~5.6
    # decades, so it could never clear a 6-decade bar no matter how geometric it
    # was).  None of the certificate's discriminating power lives in the decade
    # count: Theta(1/k), Theta(1/log k) and k^-1.5 are all rejected by cmax <= 0.9
    # alone, since their one-step ratios tend to 1.  calibrate() re-verifies this.
    out["linear_by_envelope"] = bool(ratios.size >= 3 and cmax <= 0.9 and decades >= 3.0)
    if out["linear_by_envelope"]:
        out["c1_envelope"] = float(1.0 / cmax)

    # --- linear (geometric) verdict ---
    enough = win.size >= 25   # regressions are only trusted with enough points
    out["regression_trusted"] = bool(enough)
    out["linear_by_regression"] = bool(
        enough
        and hi_g < 0.0            # slope strictly negative with 95% CI
        and r2_geo >= 0.995
        and decades >= 3.0        # at least three decades of genuine decay
        and d_aic < 0.0           # geometric fits better than power law
        and c1 > 1.0
    )
    out["linear"] = bool(out["linear_by_regression"] or out["linear_by_envelope"]
                         or out["linear_by_tail_envelope"]
                         or ke.get("k_eps_linear", False))

    # Per-question determinacy.  `linear = False` must NOT be read as "this
    # sequence is not geometric": it means no route certified that it is, which
    # covers both a genuine power law and a sequence the estimator simply cannot
    # resolve (a handful of window points, a wobbly inner solve, R^2 = 0.41).
    # Treating the second as a counterexample to a theorem is the same category
    # error as reporting "could not establish alpha > 1" as "not o(1/k)".
    #
    # So non-geometry has to be POSITIVELY established: three decades of real
    # decay that a power law fits well AND fits BETTER than a geometric.  That is
    # what Theta(1/k) and k^-1.5 do, so the negative controls still land here.
    out["established_not_geometric"] = bool(
        (not out["linear"]) and decades >= 3.0
        and r2_pow >= 0.995 and d_aic > 0.0)
    out["geometric_determined"] = bool(out["linear"] or out["established_not_geometric"])
    # Is there enough decay for ANY verdict to be meaningful?  A run that stops
    # while the gap is still on its plateau is inconclusive, not a refutation.
    # `decades` uses the raw endpoints, which understate the decay of a noisy
    # sequence whose last sample happens to sit on an up-step; the tail-sup
    # measures the same decay without that artefact.
    out["determined"] = bool(
        max(decades, tsup_decades) >= 3.0
        and (enough or out["linear_by_envelope"] or ke.get("n_levels", 0) >= 4))

    # --- o(1/k) verdict ---
    t = k * g                       # k * gap_k must tend to 0
    tail_max = np.maximum.accumulate(t[::-1])[::-1]  # sup_{j>=k} j*gap_j
    tail_decreasing = bool(tail_max[-1] <= tail_max[0] * (1 - 1e-9))
    ratio_end_start = float(t[-1] / t[0]) if t[0] > 0 else np.inf
    # Direct test of the DEFINITION.  o(1/k) means k*gap_k -> 0, so the honest
    # measurement is how far sup_{j>=k} j*gap_j actually falls -- no power-law
    # model, no fitted exponent.  The exponent route below is a proxy for this,
    # and it can be inconclusive (a wide bootstrap CI on alpha) on a sequence
    # whose k*gap_k has collapsed by nine orders of magnitude.  Reporting that
    # as "not o(1/k)" would confuse "my estimator could not decide" with
    # "the claim fails here", so the definitional route is tested directly.
    #
    # This cannot wave through the sequences that matter: for Theta(1/k),
    # k*gap_k is CONSTANT (exactly 1), so it falls zero decades and is rejected;
    # for Theta(1/log k) it grows.  calibrate() re-checks both.
    t_tail_decades = (float(np.log10(tail_max[0] / tail_max[-1]))
                      if tail_max[-1] > 0 else np.inf)
    out.update(t_start=float(t[0]), t_end=float(t[-1]),
               t_ratio_end_over_start=ratio_end_start,
               tail_sup_decreasing=tail_decreasing,
               k_gap_tail_decades=t_tail_decades)
    out["little_o_by_direct_decay"] = bool(t_tail_decades >= 3.0)
    out["little_o_1_over_k"] = bool(
        out["linear"]                      # geometric decay implies o(1/k)
        or out["little_o_by_direct_decay"]
        or (enough and -hi_p > 1.0 and tail_decreasing and ratio_end_start < 1.0)
    )
    # The o(1/k) counterpart of `established_not_geometric`, and again stated in
    # terms of the definition rather than a fitted exponent: o(1/k) FAILS exactly
    # when k*gap_k does not tend to 0, so the positive evidence is that
    # sup_{j>=k} j*gap_j is still essentially flat over a long window.  This is
    # what Theta(1/k) does (k*gap_k is identically 1, so it falls zero decades),
    # which an exponent-based rule would miss: alpha = 1 exactly, so any CI on
    # alpha straddles 1 and nothing is ever "established".
    out["established_not_little_o"] = bool(
        (not out["little_o_1_over_k"]) and enough and t_tail_decades < 0.1)
    out["little_o_determined"] = bool(out["little_o_1_over_k"]
                                      or out["established_not_little_o"])
    return out


# --------------------------------------------------------------------------
# calibration / negative control of the classifier itself
# --------------------------------------------------------------------------
def calibrate(K: int = 4000) -> list[dict]:
    """Run the estimators on sequences with known asymptotics.

    Expected: Theta(1/k) and Theta(1/log k) are NOT o(1/k) and NOT linear;
    k^{-1.5} is o(1/k) but NOT linear; 0.7^k is both.
    """
    k = np.arange(1, K + 1, dtype=float)
    cases = [
        # Theta(1/k) is the sequence the o(1/k) test must reject: it converges,
        # it is smooth, and a lax test would wave it through.
        ("theta_1_over_k", 1.0 / k,
         dict(linear=False, little_o_1_over_k=False, determined=True)),
        # Theta(1/log k) barely moves over the horizon; the instrument must say
        # "not determined" rather than guess.
        ("theta_1_over_logk", 1.0 / np.log(k + 2),
         dict(linear=False, little_o_1_over_k=False, determined=False)),
        ("k_pow_-1.5", k ** -1.5,
         dict(linear=False, little_o_1_over_k=True, determined=True)),
        ("geometric_0.7", 0.7 ** k,
         dict(linear=True, little_o_1_over_k=True, determined=True)),
        ("geometric_0.995", 0.995 ** k,
         dict(linear=True, little_o_1_over_k=True, determined=True)),
        # --- floored cases ---------------------------------------------------
        # Every case above runs with floor=0.0 and so never reaches a numerical
        # floor.  Real ADMM runs do, and the fast ones reach it within a handful
        # of iterations, which is precisely where a rate estimator is easiest to
        # get wrong.  These two cases pin down that regime from both sides.
        #
        # Contracting 25x per iteration from 4e-11 down to a 1e-16 floor gives
        # only ~5.6 decades and ~4 usable points: too few for a regression, but
        # a per-step ratio of 0.04 is about as geometric as a sequence can be.
        # The instrument must say so.  (An earlier six-decade bar on the envelope
        # certificate failed exactly this case -- it rejected sequences for
        # converging too fast.)
        ("geometric_0.04_floored", 4e-11 * (0.04 ** np.arange(K, dtype=float)),
         dict(linear=True, little_o_1_over_k=True, determined=True), 1e-16),
        # The mirror image: a Theta(1/k) sequence truncated by the same floor
        # must STILL be rejected, so the case above cannot be passed by any rule
        # that merely waves through short windows.
        ("theta_1_over_k_floored", 4e-11 / k,
         dict(linear=False, little_o_1_over_k=False), 1e-16),
        # --- noisy cases -----------------------------------------------------
        # Theorem 3.3's local gap is recovered by an inner constrained solve, so
        # it wobbles from iterate to iterate even when the underlying decay is
        # clean.  These two cases pin the tail-supremum envelope route from both
        # sides using the SAME multiplicative noise, so the only thing separating
        # them is the underlying rate.
        ("geometric_0.9_noisy",
         0.9 ** k * np.exp(_CAL_NOISE[:K]),
         dict(linear=True, little_o_1_over_k=True, determined=True)),
        # Same noise on a power law: must still be rejected, so the case above
        # cannot be passed by any rule that merely tolerates noise.
        ("k_pow_-1_noisy",
         (1.0 / k) * np.exp(_CAL_NOISE[:K]),
         dict(linear=False, little_o_1_over_k=False, determined=True)),
        # --- auto-floor cases ------------------------------------------------
        # Every case above passes an EXPLICIT floor, so none of them exercises
        # the floor that classify() infers when called with floor=None - which is
        # how the claim modules call it.  These two cover both regimes.
        #
        # (a) plateau: geometric decay that bottoms out on a solver-tolerance
        # shelf at 1e-9.  The shelf must be excluded and the decay still seen.
        ("geometric_then_plateau_autofloor",
         np.maximum(0.85 ** k, 1e-9),
         dict(linear=True, little_o_1_over_k=True, determined=True), None),
        # (b) still decaying at the end: a geometric run stopped early, spanning
        # only ~4 decades and nowhere near any floor.  The inferred floor must
        # not swallow it (the old `pos.min()*1e3` rule put the floor above every
        # point here and reported "no verdict").
        ("geometric_truncated_autofloor",
         0.93 ** np.arange(140, dtype=float),
         dict(linear=True, little_o_1_over_k=True, determined=True), None),
        # The mirror of both: a truncated Theta(1/k) under the same inferred
        # floor must still be rejected, so neither auto-floor case above can be
        # passed by a rule that simply widens the window.
        ("theta_1_over_k_truncated_autofloor",
         1.0 / np.arange(1, 141, dtype=float),
         dict(linear=False, little_o_1_over_k=False), None),
    ]
    rows = []
    for name, seq, expect, *rest in cases:
        res = classify(seq, floor=(rest[0] if rest else 0.0))
        ok = all(bool(res.get(key, False)) == val for key, val in expect.items())
        rows.append(dict(case=name, expected=expect,
                         observed=dict(linear=res["linear"],
                                       little_o_1_over_k=res["little_o_1_over_k"],
                                       determined=res.get("determined", False)),
                         power_alpha=res.get("power_alpha"),
                         c1_estimate=res.get("c1_estimate"),
                         passed=ok))
    return rows
