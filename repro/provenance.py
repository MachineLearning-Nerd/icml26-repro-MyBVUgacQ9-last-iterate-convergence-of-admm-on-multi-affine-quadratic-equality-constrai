"""Run provenance: git SHA, environment, CPU, seeds. Printed by every run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

GLOBAL_SEED = 20260726
ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".openresearch", "artifacts")


def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def cgroup_quota() -> float | None:
    """CPU quota enforced by the container, if any (cgroup v2 then v1)."""
    try:
        txt = open("/sys/fs/cgroup/cpu.max").read().split()
        if txt[0] != "max":
            return float(txt[0]) / float(txt[1])
    except Exception:
        pass
    try:
        q = float(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read().strip())
        p = float(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read().strip())
        if q > 0:
            return q / p
    except Exception:
        pass
    return None


def cpu_count() -> int:
    """Usable cores: the container quota when one is set, else affinity.

    On Hugging Face `cpu-upgrade` the host reports 64 cores while the container
    is limited to far fewer; launching one worker per host core would only
    oversubscribe.
    """
    quota = cgroup_quota()
    try:
        avail = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        avail = os.cpu_count() or 1
    if quota is not None:
        return max(1, min(avail, int(quota)))
    return avail


def provenance() -> dict:
    import numpy as np
    import scipy
    return dict(
        git_sha=_sh("git rev-parse HEAD"),
        git_branch=_sh("git rev-parse --abbrev-ref HEAD"),
        git_dirty=bool(_sh("git status --porcelain")),
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count_usable=cpu_count(),
        cpu_count_host=os.cpu_count(),
        cgroup_cpu_quota=cgroup_quota(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        global_seed=GLOBAL_SEED,
        run_command="uv run --frozen python -m repro.run_all",
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def write_artifact(relpath: str, payload) -> str:
    path = os.path.join(ARTIFACTS, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(payload, (dict, list)):
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=_jsonable)
    else:
        with open(path, "w") as fh:
            fh.write(str(payload))
    return path


def _jsonable(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def write_csv(relpath: str, rows: list[dict], fieldnames=None) -> str:
    import csv
    path = os.path.join(ARTIFACTS, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").write("")
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: _csvval(r.get(k)) for k in fieldnames})
    return path


def _csvval(v):
    import numpy as np
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, default=_jsonable)
    return v
