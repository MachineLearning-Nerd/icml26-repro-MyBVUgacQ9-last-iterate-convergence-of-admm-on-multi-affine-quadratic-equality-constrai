"""Picklable problem specifications, so worker processes rebuild instances
deterministically instead of shipping large matrices between processes."""

from __future__ import annotations

import numpy as np


def build(spec: dict):
    """Return (problem, x0) for a spec dict."""
    kind = spec["kind"]
    if kind == "locomotion":
        from repro.problems.locomotion import build_locomotion, feasible_start
        kw = {k: v for k, v in spec.items() if k not in ("kind", "x0", "x0_seed", "x0_scale")}
        p = build_locomotion(**kw)
        rng = np.random.default_rng(spec["x0_seed"]) if spec.get("x0_seed") is not None else None
        return p, feasible_start(p, rng=rng, scale=spec.get("x0_scale", 1.0))
    if kind == "random":
        from repro.problems.toy import random_maqep
        kw = {k: v for k, v in spec.items() if k not in ("kind", "x0")}
        p = random_maqep(**kw)
        return p, np.zeros(p.n_x)
    if kind == "toy":
        from repro.problems.toy import toy_q, toy_q_ball, toy_q_boxed
        mux, muz = spec.get("mu_x", 1.0), spec.get("mu_z", 1.0)
        if spec.get("box") is not None:
            lo, hi = spec["box"]
            p = toy_q_boxed(spec["q"], lo, hi, mux, muz)
        elif spec.get("ball") is not None:
            p = toy_q_ball(spec["q"], spec["ball"], mux, muz)
        else:
            p = toy_q(spec["q"], mux, muz)
        x0 = np.array(spec.get("x0", [1.0, 0.5, -0.5, 1.0]), float)
        return p, x0
    if kind == "appendix_b1":
        from repro.problems.toy import appendix_b1
        p = appendix_b1(spec.get("mu_x", 1.0), spec.get("mu_z", 1.0))
        return p, np.array(spec.get("x0", [1.0, 0.5, -0.5, 1.0]), float)
    if kind == "appendix_b2":
        from repro.problems.toy import appendix_b2
        p = appendix_b2(spec.get("mu_x", 1.0), spec.get("mu_z", 1.0))
        return p, np.array(spec.get("x0", [1.0, 0.5]), float)
    raise ValueError(f"unknown spec kind {kind!r}")
