"""Process-level parallelism across independent configurations.

Each configuration is an independent deterministic ADMM run, so the results do
not depend on the number of workers -- only the wall clock does.  BLAS threading
is pinned to one thread per worker so that the total core usage equals the number
of workers and is reportable.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import concurrent.futures as _cf  # noqa: E402

from repro.provenance import cpu_count  # noqa: E402


def workers() -> int:
    return max(1, min(cpu_count(), int(os.environ.get("REPRO_MAX_WORKERS", "64"))))


def pmap(fn, items, desc: str = ""):
    """Run `fn` over `items` in a process pool, returning results in input order."""
    n = workers()
    if n <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    with _cf.ProcessPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        out = [None] * len(items)
        done = 0
        for fut in _cf.as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
            done += 1
            if desc:
                print(f"    [{desc}] {done}/{len(items)} done", flush=True)
    return out
