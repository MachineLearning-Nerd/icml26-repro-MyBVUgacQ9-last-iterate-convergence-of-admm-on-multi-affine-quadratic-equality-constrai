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
    # contiguous run from the peak until the sequence first drops to the floor
    run = [idx[0]]
    for i in idx[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            break
    run = np.asarray(run, int)
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
        # numerical floor: 1e3 x the smallest strictly positive value observed,
        # or double-precision resolution of the sequence scale, whichever larger
        scale = float(np.nanmax(np.abs(gap))) if pos.size else 1.0
        floor = max(scale * 1e-13, (pos.min() * 1e3) if pos.size else 0.0)
    win = valid_window(gap, floor)
    ke = iters_to_tolerance(gap)
    out = dict(floor=float(floor), n_window=int(win.size), k_eps=ke,
               window=[int(win[0]) + 1, int(win[-1]) + 1] if win.size else None)
    if win.size == 0:
        # too few points above the floor for a regression: fall back on the
        # resolution-independent iterations-to-tolerance estimator alone
        out.update(linear=bool(ke.get("k_eps_linear", False)),
                   little_o_1_over_k=bool(ke.get("k_eps_linear", False)),
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

    # --- geometric envelope certificate ---
    # If every one-step ratio on the window is <= cmax < 1 then, on the observed
    # range, gap_k <= gap_peak * cmax^(k - peak) -- a verified O(c^{-k}) envelope
    # with c = 1/cmax > 1.  This works when the run reaches machine precision in
    # a handful of iterations, where a regression has too few points, and it
    # necessarily fails for Theta(1/k) (whose ratios tend to 1).
    ratios = g[1:] / g[:-1]
    cmax = float(ratios.max()) if ratios.size else 1.0
    out.update(max_one_step_ratio=cmax, n_ratios=int(ratios.size),
               median_one_step_ratio=float(np.median(ratios)) if ratios.size else 1.0)
    out["linear_by_envelope"] = bool(ratios.size >= 3 and cmax <= 0.9 and decades >= 6.0)
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
                         or ke.get("k_eps_linear", False))

    # --- o(1/k) verdict ---
    t = k * g                       # k * gap_k must tend to 0
    tail_max = np.maximum.accumulate(t[::-1])[::-1]  # sup_{j>=k} j*gap_j
    tail_decreasing = bool(tail_max[-1] <= tail_max[0] * (1 - 1e-9))
    ratio_end_start = float(t[-1] / t[0]) if t[0] > 0 else np.inf
    out.update(t_start=float(t[0]), t_end=float(t[-1]),
               t_ratio_end_over_start=ratio_end_start,
               tail_sup_decreasing=tail_decreasing)
    out["little_o_1_over_k"] = bool(
        out["linear"]                      # geometric decay implies o(1/k)
        or (enough and -hi_p > 1.0 and tail_decreasing and ratio_end_start < 1.0)
    )
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
        ("theta_1_over_k", 1.0 / k, dict(linear=False, little_o_1_over_k=False)),
        ("theta_1_over_logk", 1.0 / np.log(k + 2), dict(linear=False, little_o_1_over_k=False)),
        ("k_pow_-1.5", k ** -1.5, dict(linear=False, little_o_1_over_k=True)),
        ("geometric_0.7", 0.7 ** k, dict(linear=True, little_o_1_over_k=True)),
        ("geometric_0.995", 0.995 ** k, dict(linear=True, little_o_1_over_k=True)),
    ]
    rows = []
    for name, seq, expect in cases:
        res = classify(seq, floor=0.0)
        ok = all(bool(res[key]) == val for key, val in expect.items())
        rows.append(dict(case=name, expected=expect,
                         observed=dict(linear=res["linear"],
                                       little_o_1_over_k=res["little_o_1_over_k"]),
                         power_alpha=res.get("power_alpha"),
                         c1_estimate=res.get("c1_estimate"),
                         passed=ok))
    return rows
