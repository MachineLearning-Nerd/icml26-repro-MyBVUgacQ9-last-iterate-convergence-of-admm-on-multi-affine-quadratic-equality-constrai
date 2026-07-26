"""Extract the artifact tarball that `repro.run_all` writes into the run log.

    orx logs <runId> --bytes 100000000 > run.log
    uv run --frozen python -m repro.extract_bundle run.log <outdir>
"""

from __future__ import annotations

import base64
import hashlib
import io
import sys
import tarfile

MARKER = "ORX-ARTIFACT-BUNDLE"


def extract(log_path: str, outdir: str) -> None:
    lines = open(log_path, errors="replace").read().splitlines()
    try:
        i = next(n for n, ln in enumerate(lines) if ln.startswith(f"{MARKER}-BEGIN"))
        j = next(n for n, ln in enumerate(lines) if ln.startswith(f"{MARKER}-END"))
    except StopIteration:
        raise SystemExit("no artifact bundle found in the log")
    header = lines[i]
    blob = "".join(ln.strip() for ln in lines[i + 1:j])
    declared = dict(kv.split("=", 1) for kv in header.split()[1:])
    got = hashlib.sha256(blob.encode()).hexdigest()
    if declared.get("sha256") and got != declared["sha256"]:
        raise SystemExit(f"bundle sha256 mismatch: {got} != {declared['sha256']}")
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz") as tar:
        tar.extractall(outdir)
    print(f"extracted to {outdir} (sha256 {got}, {len(blob)} b64 chars)")


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "extracted")
