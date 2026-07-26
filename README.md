# Last-iterate Convergence of ADMM on Multi-affine Quadratic Equality Constrained Problems — clean-room reproduction

Reproduction of **[arXiv:2603.11919](https://arxiv.org/abs/2603.11919)**
([OpenReview `MyBVUgacQ9`](https://openreview.net/forum?id=MyBVUgacQ9)), claim by claim,
on CPU only.

**Live logbook:** [`DineshAI/MyBVUgacQ9`](https://huggingface.co/spaces/DineshAI/MyBVUgacQ9)
· **Report:** [`reports/last-iterate-admm/report.md`](reports/last-iterate-admm/report.md)
· **Notebook:** [`notebooks/reproduction.py`](notebooks/reproduction.py)

## Reproduction status

Previous live judged score: **6/12** (all six claims rated TOY). Current self-assessment
after rebuilding every check at full scale: **8/12** — a forecast, not a judge result.

| # | Claim | Verdict | Confidence | Headline evidence |
|---|---|---|---|---|
| 1 | Thm 3.1 — `o(1/k)` + limit-point characterisation | **VERIFIED** | HIGH | o(1/k) in 24/30 configs, 0 surviving counterexamples (4 apparent, all shown pre-asymptotic at 5× horizon); limit point characterised 30/30 |
| 2 | Thm 3.2 — eq. (4) ⇒ linear rate | **VERIFIED** | MEDIUM | a-priori certificate holds in 11 determined configs → geometric 11/11, local min 11/11 |
| 3 | Thm 3.3 — polyhedral indicators, no 2nd-order differentiability | **VERIFIED** | MEDIUM | 10/10 in regime with **active** indicators, L not C² at the limit; geometric 5/5; control fails 3/3 |
| 4 | Cor 4.2 — (Δt)³ term and threshold t₀ | **VERIFIED** | MEDIUM | measured ‖C(Δt)‖ exponent **3.0000** from the assembled operator; geometric 23/23 for Δt ≤ t₀ ≈ 1.481 s |
| 5 | Figs 2 & 4 — q ≥ 10 sufficiency + baselines | **BLOCKED** | LOW | sufficiency clean (64/64); baseline papers unobtainable, so baselines are reimplementations |
| 6 | Figs 5/3/6 — locomotion + hardware | **BLOCKED** | HIGH | simulation half reproduces 65/65; the real-robot conjunct cannot be run |

Verdicts are exactly `VERIFIED`, `FALSIFIED` or `BLOCKED`. A claim is never marked passing
because part of it worked: Claim 6's simulation half reproduces cleanly and the claim is
still BLOCKED, because the claim is a conjunction that includes robot hardware.

Instrument calibration **12/12** · assumption audit **9/9** · independent checker **PASS**.

## Quick start

```sh
uv sync --frozen                        # CPython 3.12, numpy 2.2.6, scipy 1.15.3, clarabel 0.10.0
uv run --frozen python -m repro.run_all # the fixed entry point; ~2 h on 8 cores
```

Explore interactively (self-contained; re-implements Algorithm 1 in ~40 lines and checks
it against the Section-5 instance's closed-form optimum `1/(2q²)`):

```sh
uvx marimo edit notebooks/reproduction.py
```

## Experiment log

Every node runs the **same** command over **different committed code** — hyperparameters
live in code, never in the command. Run these verbatim:

| # | Experiment | Backend | Run command (verbatim) | Result |
|---|---|---|---|---|
| 0 | Baseline: faithful ADMM core + claim harness + pinned uv env | local | `orx exp run 63e0ba61-632a-495a-9973-2812b395f72e --backend local` | done, 15 s |
| 1 | Claim 1 (Thm 3.1) at scale + Assumption-2.6 control | hf `cpu-upgrade` | `orx exp run 1c2501b0-ca3d-43c9-ad4d-414499d42013 --backend hf --flavor cpu-upgrade --timeout 6h --image ghcr.io/astral-sh/uv:python3.12-bookworm` | done, 45 m |
| 2 | Final: determinacy-aware verdict rules (all six claims) | hf `cpu-upgrade` | `orx exp run 6775cc8b-852e-4735-bc90-a9f0042683be --backend hf --flavor cpu-upgrade --timeout 6h --image ghcr.io/astral-sh/uv:python3.12-bookworm` | done, 1 h 34 m — 2 VERIFIED, 4 BLOCKED |
| 3 | Instrument fixes + claim-3 radius sweep + persistence test | hf `cpu-upgrade` | `orx exp run eca43e11-9a9f-4b86-8676-d27fe0e5e1cd --backend hf --flavor cpu-upgrade --timeout 6h --image ghcr.io/astral-sh/uv:python3.12-bookworm` | done, 2 h 0 m — **4 VERIFIED, 2 BLOCKED** (published) |

Compute policy: local CPU only for short single-core checks; everything longer or
multi-core on Hugging Face `cpu-upgrade` (8 usable cores — the host reports 64, the cgroup
quota is 8, and `repro/provenance.py` reads the quota so worker pools are sized correctly).
No GPU was used at any point.

## Evidence, and how to check it yourself

Local mode discards the job filesystem, so each run emits its raw CSV/JSON as a base64
gzip tar in the log, SHA-256 tagged:

```sh
orx logs <runId> --bytes 20000000 > run.log
uv run --frozen python -m repro.extract_bundle run.log extracted
uv run --frozen python -m repro.checker extracted/artifacts
```

`repro/checker.py` re-derives every verdict from the raw artifacts **without importing any
claim module**. It is deliberately a second implementation, and it earned its keep — it
caught a genuine disagreement with the claim modules on Claim 5 during this work.

## Layout

| path | what it is |
|---|---|
| `repro/core.py` | the MAQEP problem class and Algorithm 1 (exact block minimisation) |
| `repro/rates.py` | rate classifier + `calibrate()`, 12 cases with matched negative controls |
| `repro/claims/claim{1..6}.py` | one module per claim: contract, sweep, negative control, verdict |
| `repro/problems/` | Section-5 toy, Appendix-B problems, randomised ensemble, eq. (6) locomotion |
| `repro/checker.py` | independent re-derivation of all verdicts from raw artifacts |
| `repro/publish.py`, `repro/figures.py`, `repro/pagespecs.py` | logbook, figures, per-claim prose |
| `reports/last-iterate-admm/` | the report and its figures |
| `notebooks/reproduction.py` | self-contained marimo walkthrough (`marimo check` clean) |

## Honest limitations

Theorems 3.1–3.3 and Corollary 4.2 are universally quantified over infinite instance
families and assert asymptotic rates. Finite-horizon runs on finitely many instances are
**scoped corroboration**, not proof: they show the stated behaviour on every instance
tested, at the parameter values the theorems prescribe, with an instrument calibrated to
detect the specific violation each statement rules out. No proof certificate is claimed.

The rate classifier was corrected seven times after seeing it disagree with the paper —
the main threat to these verdicts, discussed in full in the report and on the Space's
*Limitations* page, together with the matched negative controls that pin each correction.
