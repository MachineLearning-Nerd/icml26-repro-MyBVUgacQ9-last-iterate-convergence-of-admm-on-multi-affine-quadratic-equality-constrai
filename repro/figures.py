"""Figures for the report and the candidate logbook, built from extracted artifacts."""

from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = dict(admm="#1f77b4", padmm="#d62728", iadmm="#2ca02c", ipds_admm="#9467bd",
               accent="#ff7f0e", grey="#7f7f7f")


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_headline(art: str, out: str):
    """Claim 1 + 3: the two distinct convergence statements, measured."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    raw = json.load(open(os.path.join(art, "claim1", "claim1_raw.json")))
    ax = axes[0]
    shown = 0
    for key, d in raw.items():
        if "rho_x1" not in key:
            continue
        g = np.asarray(d["gap"], float)
        g = g[np.isfinite(g) & (g > 0)]
        if g.size < 10:
            continue
        k = np.arange(1, g.size + 1)
        lab = key.split("|")[0][:26]
        ax.loglog(k, g, lw=1.2, alpha=0.85, label=lab)
        shown += 1
        if shown >= 6:
            break
    k = np.logspace(0, np.log10(max(1e3, 10)), 50)
    ax.loglog(k, k ** -1.0 * 1e-1, "--", color=PALETTE["grey"], lw=1.4,
              label=r"$\Theta(1/k)$ reference")
    _style(ax, "iteration $k$", r"$|L(x^k,z^k,w^k)-L^\star|$",
           "Theorem 3.1: gap falls strictly faster than $1/k$")
    ax.legend(fontsize=6.5, frameon=False, ncol=1)

    lg = json.load(open(os.path.join(art, "claim3", "claim3_local_gaps.json")))
    ax = axes[1]
    for key, g in list(lg.items()):
        g = np.asarray(g, float)
        g = np.where(g > 0, g, np.nan)
        if np.isfinite(g).sum() < 5:
            continue
        ctrl = key.startswith("CONTROL")
        ax.semilogy(np.arange(1, g.size + 1), g, lw=1.6 if ctrl else 1.1,
                    color=PALETTE["padmm"] if ctrl else None,
                    ls="--" if ctrl else "-", alpha=0.9,
                    label=(key[:30] if not ctrl else "control: $\\|C\\|$ outside eq.(4)"))
    _style(ax, "iteration $k$",
           r"$L(x^k,z^k,w^k)-\min_{B(x^k,z^k;r)}L$",
           "Theorem 3.3: local gap under ACTIVE polyhedral indicators")
    h, l = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b); hh.append(a); ll.append(b)
    ax.legend(hh[:6], ll[:6], fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_dt_scaling(art: str, out: str):
    """Claim 4: measured ||C(dt)|| exponent vs the vacuous circular fit."""
    sc = json.load(open(os.path.join(art, "claim4", "claim4_C_scaling.json")))
    dts = np.asarray(sc["dts"], float)
    nc = np.asarray(sc["norm_C"], float)
    nd = np.asarray(sc["norm_d"], float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    ax = axes[0]
    ax.loglog(dts, nc, "o-", color=PALETTE["admm"], lw=1.6, ms=4,
              label=f"measured $\\|C(\\Delta t)\\|$  (slope {sc['measured_slope_normC_vs_dt']:.3f})")
    ref = nc[0] * (dts / dts[0]) ** 3
    ax.loglog(dts, ref, "--", color=PALETTE["grey"], lw=1.4, label=r"$(\Delta t)^3$")
    ax.loglog(dts, nd, "s-", color=PALETTE["accent"], lw=1.2, ms=3.5,
              label=r"measured $\|d(\Delta t)\|$ (linear part)")
    _style(ax, r"$\Delta t$ [s]", "operator norm",
           "Measured from the assembled eq.(6) operator")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    rows = _read_csv(os.path.join(art, "claim4", "claim4_dt_sweep.csv"))
    for tag, mk in (("no-indicators", "o"), ("with-indicators", "s")):
        sub = [r for r in rows if r["label"].startswith(tag)]
        x = [_f(r["dt"]) for r in sub]
        y = [1.0 if r["linear"] == "True" else 0.0 for r in sub]
        det = [r["determined"] == "True" for r in sub]
        ax.scatter([xi for xi, d in zip(x, det) if d],
                   [yi for yi, d in zip(y, det) if d],
                   marker=mk, s=42, label=f"{tag} (determined)",
                   color=PALETTE["admm"] if tag == "no-indicators" else PALETTE["iadmm"])
    bis = json.load(open(os.path.join(art, "claim4", "claim4_t0_bisection.json")))
    ax.axvline(bis["t0_lower"], color=PALETTE["padmm"], ls="--", lw=1.4,
               label=f"bisected $t_0\\approx${bis['t0_lower']:.3f}s")
    ax.axvline(0.005, color=PALETTE["grey"], ls=":", lw=1.4,
               label=r"paper's suggested $\Delta t=0.005$s")
    ax.set_xscale("log")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not geometric", "geometric"])
    _style(ax, r"$\Delta t$ [s]", "", r"Corollary 4.2: the $\Delta t$ threshold")
    ax.legend(fontsize=7, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_claim2(art: str, out: str):
    """Claim 2: eq.(4) certificate vs the bisected empirical onset in ||C||."""
    rows = [r for r in _read_csv(os.path.join(art, "claim2", "claim2_results.csv"))
            if r["label"].startswith("sweep|")]
    bis = json.load(open(os.path.join(art, "claim2", "claim2_bisection.json")))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    ax = axes[0]
    for seed, col in zip((1, 2, 3), (PALETTE["admm"], PALETTE["iadmm"], PALETTE["accent"])):
        sub = sorted([r for r in rows if f"seed={seed}|" in r["label"]],
                     key=lambda r: _f(r["norm_C"]))
        x = [max(_f(r["norm_C"]), 1e-5) for r in sub]
        ax.loglog(x, [max(_f(r["eq4_margin"]), 1e-6) for r in sub], "o-", ms=3.5,
                  lw=1.3, color=col, label=f"seed {seed}")
    ax.axhline(1.0, color=PALETTE["padmm"], ls="--", lw=1.4,
               label="certificate threshold (margin = 1)")
    _style(ax, r"$\|C\|$", "eq.(4) margin  $\\mu_f\\,/\\,$bound",
           "A-priori certificate, from problem data only")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for seed, col in zip((1, 2, 3), (PALETTE["admm"], PALETTE["iadmm"], PALETTE["accent"])):
        sub = sorted([r for r in rows if f"seed={seed}|" in r["label"]],
                     key=lambda r: _f(r["norm_C"]))
        x = [max(_f(r["norm_C"]), 1e-5) for r in sub]
        y = [_f(r["lam_min_reduced_hessian"]) for r in sub]
        ax.semilogx(x, y, "o-", ms=3.5, lw=1.3, color=col, label=f"seed {seed}")
    for b, col in zip(bis, (PALETTE["admm"], PALETTE["iadmm"], PALETTE["accent"])):
        ax.axvline(b["onset_lo"], color=col, ls=":", lw=1.2)
    ax.axhline(0.0, color=PALETTE["grey"], lw=1.0)
    _style(ax, r"$\|C\|$", r"$\lambda_{\min}(\nabla^2 f(x^\star)+\sum_i w^\star_i C_i)$",
           "Operative condition; dotted = bisected geometric-convergence onset")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_baselines(art: str, out: str):
    """Claim 5: q sufficiency and the Figure-4 baseline comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    qrows = _read_csv(os.path.join(art, "claim5", "claim5_q_sweep.csv"))
    ax = axes[0]
    det = [r for r in qrows if r["determined"] == "True"]
    qs = np.array([_f(r["q"]) for r in det])
    lin = np.array([r["linear"] == "True" for r in det])
    edges = np.unique(qs)
    frac = [lin[qs == q].mean() for q in edges]
    ax.semilogx(edges, frac, "o-", color=PALETTE["admm"], lw=1.5, ms=4.5)
    ax.axvline(10.0, color=PALETTE["padmm"], ls="--", lw=1.5,
               label=r"paper's sufficient $q\geq 10$")
    onsets = json.load(open(os.path.join(art, "claim5", "claim5_q_onset_bisection.json")))
    ax.axvline(float(np.median([o["onset_hi"] for o in onsets])), color=PALETTE["iadmm"],
               ls=":", lw=1.5, label="bisected empirical onset (median)")
    ax.set_ylim(-0.05, 1.08)
    _style(ax, "$q$", "fraction converging geometrically",
           "Figure 2: sufficiency of $q\\geq10$ over 4 $\\mu$-settings, 3 starts, 2 $\\rho$")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    brows = _read_csv(os.path.join(art, "claim5", "claim5_baseline_comparison.csv"))
    ax = axes[1]
    algos = ("admm", "padmm", "iadmm", "ipds_admm")
    labels = {"admm": "ADMM (Alg. 1)", "padmm": "PADMM", "iadmm": "IADMM",
              "ipds_admm": "IPDS-ADMM"}
    probs = ("B1", "B2", "B3")
    width = 0.2
    for i, a in enumerate(algos):
        vals = []
        for p in probs:
            sub = [r for r in brows if r["problem"] == p]
            it = [_f(r[f"{a}_iters_to_tol"]) for r in sub]
            it = [v for v in it if np.isfinite(v)]
            vals.append(np.median(it) if it else np.nan)
        ax.bar(np.arange(len(probs)) + (i - 1.5) * width, vals, width,
               color=PALETTE[a], label=labels[a])
    ax.set_xticks(np.arange(len(probs)))
    ax.set_xticklabels(["B1\n(multi-affine)", "B2\n(linear)", "B3\n(nonconvex, linear)"],
                       fontsize=8)
    _style(ax, "", r"median iterations to $\|A(x)+Qz\|\leq 10^{-8}$",
           "Figure 4: baseline comparison (reimplemented baselines)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_locomotion(art: str, out: str):
    """Claim 6: Figure 3's violation curves and the Figure 5 panels."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

    f3 = json.load(open(os.path.join(art, "claim6", "claim6_fig3_violation_mean_std.json")))
    ax = axes[0]
    for (key, d), col in zip(f3.items(), (PALETTE["admm"], PALETTE["accent"], PALETTE["iadmm"])):
        m = np.asarray(d["mean"], float)
        s = np.asarray(d["std"], float)
        k = np.arange(1, m.size + 1)
        ax.semilogy(k, np.maximum(m, 1e-18), lw=1.6, color=col,
                    label=f"{key}  ({d['n_trials']} trials)")
        ax.fill_between(k, np.maximum(m - s, 1e-18), np.maximum(m + s, 1e-18),
                        color=col, alpha=0.18, linewidth=0)
    _style(ax, "ADMM iteration $k$", r"centroidal dynamics violation $\|A(x)+Qz\|$",
           "Figure 3: humanoid jump, mean $\\pm$ std over randomised trials")
    ax.legend(fontsize=7, frameon=False)

    tr = json.load(open(os.path.join(art, "claim6", "claim6_violation_traces.json")))
    ax = axes[1]
    for key, v in tr.items():
        if not key.startswith("fig5-right"):
            continue
        v = np.asarray(v, float)
        ax.semilogy(np.arange(1, v.size + 1), np.maximum(v, 1e-18), lw=1.0, alpha=0.8)
    _style(ax, "ADMM iteration $k$", r"$\|A(x)+Qz\|$",
           "Figure 5 (right): 2D locomotion, 10 randomised initial configurations")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


ALL = [("headline_convergence.png", fig_headline),
       ("claim2_eq4_certificate.png", fig_claim2),
       ("claim4_dt_scaling.png", fig_dt_scaling),
       ("claim5_baselines.png", fig_baselines),
       ("claim6_locomotion.png", fig_locomotion)]


def build_all(art: str, outdir: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    made = []
    for name, fn in ALL:
        try:
            fn(art, os.path.join(outdir, name))
            made.append(name)
            print(f"  wrote {name}")
        except Exception as exc:  # a missing artifact must not hide the others
            print(f"  SKIPPED {name}: {type(exc).__name__}: {exc}")
    return made


if __name__ == "__main__":
    import sys
    build_all(sys.argv[1], sys.argv[2])
