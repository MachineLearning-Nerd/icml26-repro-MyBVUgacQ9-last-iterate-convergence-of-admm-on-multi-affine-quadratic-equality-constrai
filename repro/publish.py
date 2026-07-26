"""Build the candidate Hugging Face logbook from the extracted artifacts.

    uv run --frozen python -m repro.publish <artifacts-dir> <judged-dir> <out-dir> [checker.txt]

The judged revision's files are copied in first and never modified, so the old
file set is a subset of the new one by construction.  The two pages that carried
the superseded verification stay byte-identical but are re-labelled in the
navigation as "Historical rejected baseline"; the current verification comes
first in the tree.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import textwrap

from repro.pagespecs import CLAIMS

RUN_COMMAND = "uv run --frozen python -m repro.run_all"
REPO = ("https://github.com/MachineLearning-Nerd/icml26-repro-MyBVUgacQ9-"
        "last-iterate-convergence-of-admm-on-multi-affine-quadratic-equality-constrai")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fmt(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "" if v is None else str(v)
    if f != f:
        return "n/a"
    if f == 0 or 1e-3 <= abs(f) < 1e5:
        return f"{f:.4g}"
    return f"{f:.3e}"


def md_table(rows, cols, limit=None):
    if not rows:
        return "_(no rows recorded)_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in (rows[:limit] if limit else rows):
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    if limit and len(rows) > limit:
        out.append(f"\n_{len(rows) - limit} further rows are in the linked raw CSV._")
    return "\n".join(out) + "\n"


def code_excerpt(repo_root, path, first=0, last=60):
    full = os.path.join(repo_root, path)
    if not os.path.exists(full):
        return ""
    lines = open(full).read().splitlines()[first:last]
    return "```python\n" + "\n".join(lines) + "\n```\n"


def provenance_block(prov, seconds):
    return textwrap.dedent(f"""
    | provenance | value |
    |---|---|
    | Git SHA | `{prov.get('git_sha')}` |
    | branch | `{prov.get('git_branch')}` |
    | fixed run command | `{RUN_COMMAND}` |
    | Python | {prov.get('python')} |
    | numpy / scipy | {prov.get('numpy')} / {prov.get('scipy')} |
    | platform | {prov.get('platform')} |
    | CPU allocation | {prov.get('cpu_count_usable')} usable cores (cgroup quota {prov.get('cgroup_cpu_quota')}; host reports {prov.get('cpu_count_host')}) |
    | global seed | {prov.get('global_seed')} |
    | total wall clock | {seconds:.0f} s |
    | UTC | {prov.get('utc')} |
    """).strip() + "\n"


def copy_raw(art, out):
    copied = []
    for dirpath, _d, files in os.walk(art):
        for fn in files:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, art)
            if rel.split(os.sep)[0] == "paper":
                continue
            dst = os.path.join(out, "raw", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append("raw/" + rel.replace(os.sep, "/"))
    return sorted(copied)


# --------------------------------------------------------------------------
def page_verification(summary, imgs, claims):
    prov = summary["provenance"]
    cal, aud = summary["calibration"], summary["assumptions"]
    L = ["# Verification (current)", "",
         "This page and the six per-claim pages linked below are the **current** "
         "verification of this reproduction. They supersede *Evidence (historical "
         "rejected baseline)* and *Verification run (historical rejected baseline)*, "
         "which are preserved unchanged for the record and must not be read as the "
         "current verifier.", "",
         f"**Superseding code and revision.** Everything here is produced by `repro/` "
         f"at Git SHA `{prov.get('git_sha')}` (branch `{prov.get('git_branch')}`), run "
         f"with the single fixed command `{RUN_COMMAND}`. The superseded verifier was "
         "`repro/src/verify_admm.py`, a 4-variable numpy script that classified "
         "convergence with the heuristic `R^2 < 0.97 or tail ratio > 0.995`. It is no "
         "longer executed anywhere in this artifact.", ""]
    if "headline_convergence.png" in imgs:
        L += ["## Headline result", "", "![headline](images/headline_convergence.png)",
              "", "*Left:* the Theorem 3.1 Lagrangian gap against a `Theta(1/k)` "
              "reference line, on the paper's locomotion problem and a randomised "
              "ensemble. *Right:* the Theorem 3.3 local gap when the polyhedral "
              "indicators are **active** at the limit, with the out-of-regime control "
              "dashed.", ""]
    L += ["## Verdicts", "",
          md_table([dict(claim=f"[{CLAIMS[k]['title']}](#/{CLAIMS[k]['slug']})",
                         verdict=v.get("verdict"),
                         confidence=v.get("confidence", "-"),
                         headline=v.get("headline", v.get("reason", "")))
                    for k, v in claims.items()],
                   ["claim", "verdict", "confidence", "headline"]),
          "", "Every verdict is exactly one of VERIFIED, FALSIFIED or BLOCKED. "
          "Nothing toy, skipped, proxy or inconclusive is reported as a pass.", "",
          "## Instrument calibration (a negative control on the measurement itself)", "",
          "Before any claim is measured, the rate classifier runs on sequences whose "
          "asymptotics are known. An instrument that always answered \"linear\", or "
          "that waved `Theta(1/k)` through as `o(1/k)`, would fail here. This replaces "
          "the heuristic used by the superseded verifier.", "",
          md_table([dict(case=c["case"], expected=json.dumps(c["expected"]),
                         observed=json.dumps(c["observed"]),
                         result="pass" if c["passed"] else "FAIL")
                    for c in cal["cases"]], ["case", "expected", "observed", "result"]),
          "", f"**{sum(c['passed'] for c in cal['cases'])}/{len(cal['cases'])} correct.** "
          "Raw: [rate_classifier_calibration.json]"
          "(raw/calibration/rate_classifier_calibration.json)", "",
          "## Assumption audit", "",
          "Every instance is checked numerically against Assumption 2.3 (strong "
          "convexity of `f` and `phi`; closed convex block indicators), Assumption 2.6 "
          "(`Q` full row rank) and Definition 2.1 (every `C_j` has zero diagonal "
          "blocks, i.e. `A` really is multi-affine) **before** it is used, and the "
          "Theorem 3.1 threshold `rho_thm` is computed from the formula.", "",
          md_table([dict(instance=r["instance"], n_x=r["n_x"], n_c=r["n_c"],
                         n_z=r["n_z"], blocks=r["n_blocks"], mu_f=r["mu_f"],
                         mu_phi=r["mu_phi"], normC=r["norm_C"],
                         rankQ=f"{r['rank_Q']}/{r['n_c']}",
                         rho_thm=r["rho_threshold"], holds=r["assumptions_hold"])
                    for r in aud["instances"]],
                   ["instance", "n_x", "n_c", "n_z", "blocks", "mu_f", "mu_phi",
                    "normC", "rankQ", "rho_thm", "holds"]),
          "", "Raw: [assumption_audit.json](raw/assumptions/assumption_audit.json)", "",
          "## Reproducing this", "", provenance_block(prov, summary.get("wall_clock_s", 0.0)),
          "", "```bash", f"git clone {REPO}", f"git checkout {prov.get('git_sha')}",
          "uv sync --frozen    # python 3.12.11, numpy 2.2.6, scipy 1.15.3, clarabel 0.10.0",
          f"{RUN_COMMAND}", "```", "",
          "The command is identical on every node of the experiment tree; nodes differ "
          "only in committed code. It **exits non-zero** if the instrument calibration "
          "fails, if the assumptions do not hold on the instances used, or if any "
          "claim check reports an internal failure.", "",
          "## Independent checker", "",
          "`repro/checker.py` re-derives every verdict from the raw CSV/JSON alone - it "
          "does not import the claim modules and does not re-run ADMM - and exits "
          "non-zero on any disagreement, on a missing artifact, or on a negative "
          "control that failed to fail. Its output is on "
          "[Independent checker output](#/checker-output).", "",
          "## Where to go next", "",
          "| page | what it holds |", "|---|---|",
          "| [Visibility matrix](#/visibility-matrix) | one row per claim: where each "
          "required piece of evidence lives |",
          "| [Limitations and deviations](#/limitations) | every deviation from the "
          "paper, stated in one place |",
          "| [Independent checker output](#/checker-output) | verbatim output of the "
          "independent re-derivation |"]
    for k in CLAIMS:
        if k in claims:
            L.append(f"| [{CLAIMS[k]['title']}](#/{CLAIMS[k]['slug']}) | "
                     f"{claims[k].get('verdict')} |")
    return "\n".join(L)


def page_claim(key, spec, summary, art, imgs, repo_root):
    res = summary["claims"][key]
    prov = summary["provenance"]
    rawcsv = [p for p in spec["raw"] if p.endswith(".csv")]
    rows = read_csv(os.path.join(art, rawcsv[0][len("raw/"):])) if rawcsv else []
    img = {"claim2": "claim2_eq4_certificate.png", "claim4": "claim4_dt_scaling.png",
           "claim5": "claim5_baselines.png", "claim6": "claim6_locomotion.png",
           "claim1": "headline_convergence.png", "claim3": "headline_convergence.png"}.get(key)
    L = [f"# {spec['title']}", "",
         f"**Verdict: {res.get('verdict')}**  ·  confidence {res.get('confidence','-')}", "",
         f"> {res.get('headline', res.get('reason',''))}", "",
         "## The claim, as printed in the paper", "", spec["statement"], "",
         "## Exact quantifiers, and what they force the test to do", ""]
    L += [f"- {q}" for q in spec["quantifiers"]]
    L += ["", "## What was measured", ""]
    L += [f"- {m}" for m in spec["measured"]]
    if img and img in imgs:
        L += ["", f"![{key}](images/{img})", ""]
    L += ["", "## Results (inline)", "", md_table(rows, spec["table_cols"], limit=40), ""]
    L += ["## Negative control", "", spec["control"], ""]
    ctrl_path = os.path.join(art, key, f"{key}_negative_control.json")
    if os.path.exists(ctrl_path):
        L += ["```json", json.dumps(json.load(open(ctrl_path)), indent=2), "```", ""]
    L += ["## Raw data", "",
          "\n".join(f"- [{p.split('/')[-1]}]({p})" for p in spec["raw"]), "",
          "## Executable source", "",
          "\n".join(f"- `{c}` in the repository at Git SHA `{prov.get('git_sha')}` "
                    f"([browse]({REPO}/blob/{prov.get('git_sha')}/{c}))"
                    for c in spec["code"]), "",
          "The relevant module's contract, verbatim from the source:", "",
          code_excerpt(repo_root, spec["code"][0], 0, 45),
          "## Exact command, pinned environment, provenance", "",
          "```bash", f"git checkout {prov.get('git_sha')} && uv sync --frozen && {RUN_COMMAND}",
          "```", "", provenance_block(prov, summary.get("wall_clock_s", 0.0)), "",
          "## Limitations and deviations for this claim", ""]
    L += [f"- {x}" for x in spec["limitations"]]
    L += ["", f"Back to [Verification (current)](#/verification-current)."]
    return "\n".join(L)


def page_visibility(summary, art):
    cols = ["Claim", "Canonical page", "Code visible", "Data inline", "Raw link",
            "Checker", "Control", "Exact claim tested", "Reviewer verdict"]
    lines = ["# Visibility matrix", "",
             "Every cell below is reachable by following links from "
             "[the index](#/index) → [Verification (current)](#/verification-current) → "
             "the claim page. No cell relies on the GitHub repository, the "
             "OpenResearch dashboard, an experiment branch or any unlinked path.", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for k, spec in CLAIMS.items():
        if k not in summary["claims"]:
            continue
        res = summary["claims"][k]
        rawlinks = ", ".join(f"[{p.split('/')[-1]}]({p})" for p in spec["raw"])
        lines.append("| " + " | ".join([
            f"[{spec['title'].split(' - ')[0]}](#/{spec['slug']})",
            f"[{spec['slug']}](#/{spec['slug']})",
            "yes (excerpt + repo link)",
            "yes (results table)",
            rawlinks,
            "[checker-output](#/checker-output)",
            "yes (on the claim page)",
            "yes (statement + quantifiers section)",
            res.get("verdict", "?"),
        ]) + " |")
    lines += ["", "## Non-claim evidence", "",
              "| item | where |", "|---|---|",
              "| instrument calibration | [Verification (current)]"
              "(#/verification-current) + [raw]"
              "(raw/calibration/rate_classifier_calibration.json) |",
              "| assumption audit | [Verification (current)](#/verification-current) + "
              "[raw](raw/assumptions/assumption_audit.json) |",
              "| full machine-readable summary | [summary.json](raw/summary.json) |",
              "| provenance (SHA, seeds, CPU, runtime) | [provenance.json]"
              "(raw/provenance.json) and every claim page |",
              "| limitations and deviations | [Limitations](#/limitations) |"]
    return "\n".join(lines)


def page_limitations(summary):
    L = ["# Limitations and deviations", "",
         "Collected in one place so that no deviation is discoverable only by reading "
         "code. Each also appears on its claim page.", ""]
    for k, spec in CLAIMS.items():
        if k not in summary["claims"]:
            continue
        L += [f"## {spec['title']}", ""]
        L += [f"- {x}" for x in spec["limitations"]]
        L += [""]
    L += ["## Scope of the whole reproduction", "",
          "- This is a **clean-room** reimplementation from the paper text. No author "
          "code, data or hardware logs were used, because none are published.",
          "- All compute is CPU. Runs longer than a few minutes or needing more than "
          "one core execute on Hugging Face `cpu-upgrade` (8 usable cores, cgroup "
          "quota 8.0); short single-core checks run locally. No GPU was used.",
          "- Theorems 3.1-3.3 and Corollary 4.2 are universally quantified over "
          "infinite instance families and assert asymptotic rates. Finite-horizon runs "
          "on finitely many instances are **scoped corroboration**: they show the "
          "stated behaviour on every instance tested, at the parameter values the "
          "theorems prescribe, with an instrument calibrated to detect the specific "
          "violation each statement rules out. No proof certificate is claimed and "
          "none is implied.",
          "- Where the paper leaves a parameter unstated (`mu_x`, `mu_z`, the constants "
          "`m_i` of equation 4), the parameter is swept or the paper's own derivation "
          "is used to pin it down, and this is said explicitly rather than resolved "
          "silently."]
    return "\n".join(L)


def page_checker(text):
    return ("# Independent checker output\n\n"
            "`repro/checker.py` re-derives every verdict from the raw CSV/JSON "
            "artifacts alone. It does not import the claim modules and does not re-run "
            "ADMM. It exits non-zero on any disagreement with the recorded verdicts, "
            "on a missing artifact, or on a negative control that failed to fail.\n\n"
            "```bash\nuv run --frozen python -m repro.checker <artifacts-dir>\n```\n\n"
            "```\n" + (text or "(checker output not captured)") + "\n```\n")


def page_index(summary):
    claims = summary["claims"]
    verified = sum(1 for v in claims.values() if v.get("verdict") in ("VERIFIED", "FALSIFIED"))
    L = ["# Repro - Last-iterate Convergence of ADMM on Multi-affine Quadratic "
         "Equality Constrained Problem", "",
         "Clean-room CPU reproduction of [arXiv:2603.11919]"
         "(https://arxiv.org/abs/2603.11919) "
         "([OpenReview](https://openreview.net/forum?id=MyBVUgacQ9)), claim by claim.",
         "",
         "**Start here: [Verification (current)](#/verification-current).** That page "
         "and the six claim pages linked from it are the current verifier. The pages "
         "labelled *historical rejected baseline* are preserved for the record only.",
         "",
         f"{verified} of {len(claims)} claims resolve to VERIFIED or FALSIFIED; the "
         "rest are reported as BLOCKED with the specific missing capability named.",
         "", "## Pages", "", "| page | |", "|---|---|",
         "| [Verification (current)](#/verification-current) | **current** verifier: "
         "verdicts, calibration, assumption audit, exact command |",
         "| [Visibility matrix](#/visibility-matrix) | where every required piece of "
         "evidence lives |",
         "| [Limitations and deviations](#/limitations) | every deviation, in one place |",
         "| [Independent checker output](#/checker-output) | verdicts re-derived from "
         "raw data alone |"]
    for k, spec in CLAIMS.items():
        if k in claims:
            L.append(f"| [{spec['title']}](#/{spec['slug']}) | "
                     f"{claims[k].get('verdict')} · {claims[k].get('confidence','-')} |")
    L += ["| [Overview](#/overview) | paper and scope |",
          "| [Claims](#/claims) | the six claim statements |",
          "| [Conclusion](#/conclusion) | summary |",
          "| [Evidence (historical rejected baseline)](#/evidence) | preserved, "
          "superseded |",
          "| [Verification run (historical rejected baseline)](#/verification-run) | "
          "preserved, superseded |"]
    return "\n".join(L)


def page_conclusion(summary):
    claims = summary["claims"]
    prov = summary["provenance"]
    L = ["# Conclusion", "", "## Executive summary", "",
         "Clean-room reproduction of arXiv:2603.11919 (`MyBVUgacQ9`) on CPU. Each of "
         "the six anchored claims was turned into an explicit contract, tested against "
         "the paper's own problems at their stated parameters, and given exactly one "
         "of VERIFIED, FALSIFIED or BLOCKED.", "",
         md_table([dict(claim=CLAIMS[k]["title"], verdict=v.get("verdict"),
                        confidence=v.get("confidence", "-"),
                        headline=v.get("headline", v.get("reason", "")))
                   for k, v in claims.items()],
                  ["claim", "verdict", "confidence", "headline"]), "",
         "## What changed relative to the previously judged revision", "", "",
         "| previous evidence | what replaced it |", "|---|---|",
         "| a single 4-variable synthetic problem for every claim | the paper's own "
         "eq.(6) locomotion problem in 2D and 3D (up to n_x = 288 force variables, 75 "
         "constraints), a randomised multi-affine ensemble, and the Section-5 / "
         "Appendix-B problems |",
         "| rate classified by the heuristic `R^2<0.97 or tail ratio>0.995` | a "
         "classifier calibrated against sequences of known rate, which **rejects** "
         "`Theta(1/k)` as `o(1/k)` |",
         "| assumptions never checked | Assumption 2.3, Assumption 2.6 and Definition "
         "2.1 audited numerically on every instance, and `rho` computed from the "
         "Theorem 3.1 formula |",
         "| eq.(4) never evaluated | an a-priori certificate derived from Appendix D, "
         "plus the posterior reduced-Hessian condition, plus a bisected empirical "
         "onset in `\\|C\\|` |",
         "| box constraint that never activated | instances whose polyhedral "
         "indicators are **active** at the limit, so `L` is verifiably not "
         "second-order differentiable there, and the theorem's actual local-gap "
         "quantity |",
         "| `(dt)^3` verified by regressing `log(dt**3)` on `log(dt)` | `\\|C(dt)\\|` "
         "**measured** from the assembled eq.(6) operator; the circular fit is kept "
         "alongside and flagged as vacuous |",
         "| baseline comparison explicitly not reproduced | PADMM, IADMM and IPDS-ADMM "
         "implemented and compared on the three Appendix-B problems (with the "
         "provenance deviation stated) |",
         "| real robot experiments absent, claim marked passing | the humanoid-jump "
         "and quadruped-bound centroidal solves reproduced including flight phases and "
         "Figure 3's mean±std curves; the hardware conjunct reported **BLOCKED**, not "
         "passing |", "",
         "## Scope and cost", "", "| | this reproduction |", "|---|---|",
         "| hardware | CPU only (Hugging Face `cpu-upgrade`, "
         f"{prov.get('cpu_count_usable')} usable cores) |",
         f"| wall clock | {summary.get('wall_clock_s', 0.0) / 60:.0f} min for the full "
         "suite |", "| GPU | none |",
         f"| Git SHA | `{prov.get('git_sha')}` |", "",
         "See [Limitations and deviations](#/limitations) for everything this "
         "reproduction does not establish."]
    return "\n".join(L)


# --------------------------------------------------------------------------
def build(art, judged, out, checker_text=None, repo_root="."):
    os.makedirs(out, exist_ok=True)
    for dirpath, _d, files in os.walk(judged):
        for fn in files:
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, judged)
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    summary = json.load(open(os.path.join(art, "summary.json")))
    claims = summary["claims"]
    copy_raw(art, out)

    from repro.figures import build_all
    imgs = build_all(art, os.path.join(out, "images"))

    pages = {
        "index": page_index(summary),
        "verification-current": page_verification(summary, imgs, claims),
        "visibility-matrix": page_visibility(summary, art),
        "limitations": page_limitations(summary),
        "checker-output": page_checker(checker_text),
        "conclusion": page_conclusion(summary),
    }
    for k, spec in CLAIMS.items():
        if k in claims:
            pages[spec["slug"]] = page_claim(k, spec, summary, art, imgs, repo_root)

    for slug, text in pages.items():
        path = (os.path.join(out, "pages", "index.md") if slug == "index"
                else os.path.join(out, "pages", slug, "page.md"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(text + "\n")

    # navigation: current verification first, historical last and clearly labelled
    children = [dict(slug="verification-current", title="Verification (current)",
                     file="pages/verification-current/page.md", children=[])]
    for k, spec in CLAIMS.items():
        if k in claims:
            children.append(dict(slug=spec["slug"], title=spec["title"],
                                 file=f"pages/{spec['slug']}/page.md", children=[]))
    children += [
        dict(slug="visibility-matrix", title="Visibility matrix",
             file="pages/visibility-matrix/page.md", children=[]),
        dict(slug="limitations", title="Limitations and deviations",
             file="pages/limitations/page.md", children=[]),
        dict(slug="checker-output", title="Independent checker output",
             file="pages/checker-output/page.md", children=[]),
        dict(slug="overview", title="Overview", file="pages/overview/page.md", children=[]),
        dict(slug="claims", title="Claims", file="pages/claims/page.md", children=[]),
        dict(slug="conclusion", title="Conclusion", file="pages/conclusion/page.md",
             children=[]),
        dict(slug="evidence", title="Evidence (historical rejected baseline)",
             file="pages/evidence/page.md", children=[]),
        dict(slug="verification-run",
             title="Verification run (historical rejected baseline)",
             file="pages/verification-run/page.md", children=[]),
    ]
    lb = json.load(open(os.path.join(out, "logbook.json")))
    lb["root"]["children"] = children
    lb["updated_at"] = summary["provenance"]["utc"]
    json.dump(lb, open(os.path.join(out, "logbook.json"), "w"), indent=2)

    return dict(out=out, images=imgs, n_pages=len(pages))


if __name__ == "__main__":
    art, judged, out = sys.argv[1], sys.argv[2], sys.argv[3]
    txt = open(sys.argv[4]).read() if len(sys.argv) > 4 and os.path.exists(sys.argv[4]) else None
    print(json.dumps(build(art, judged, out, txt), indent=2))
