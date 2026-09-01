# SPDX-License-Identifier: MIT
"""Generate one mesh from one image, no JSON required.

Run with the repository's virtual environment:

    .venv\\Scripts\\python.exe tools\\run_single.py --image assets\\sample.png --out C:\\out
    .venv\\Scripts\\python.exe tools\\run_single.py --image assets\\sample.png --out C:\\out ^
        --ss_steps 12 --slat_steps 6 --seed 1

This is a thin wrapper over the runner protocol (one JSON object per line on
stdin/stdout). It starts the runner as a child process, streams its progress to
the console, and prints the result as JSON at the end.

The first run is much slower: MIOpen tunes convolution kernels once (measured
which can add minutes on the very first mesh). Benchmark from the second run onward.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="input image (png/jpg)")
    parser.add_argument("--out", required=True, help="output directory (created if missing)")
    parser.add_argument("--ss_steps", type=int, default=None)
    parser.add_argument("--slat_steps", type=int, default=None)
    parser.add_argument("--ss_guidance", type=float, default=None)
    parser.add_argument("--slat_guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    params: dict[str, object] = {
        "image_path": str(Path(args.image).resolve()),
        "out_dir": str(Path(args.out).resolve()),
    }
    for key in ("ss_steps", "slat_steps", "ss_guidance", "slat_guidance", "seed"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    proc = subprocess.Popen(
        [sys.executable, "-m", "runners.trellis"],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None

    request = {"id": 1, "method": "image_to_mesh", "params": params}
    started = time.perf_counter()
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()

    exit_code = 2
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            print(f"[non-protocol line] {line}", file=sys.stderr, flush=True)
            continue
        elapsed = time.perf_counter() - started
        kind = event.get("event")
        if kind == "progress":
            print(f"[{elapsed:7.1f}s] {event.get('stage')}: {event.get('message')}", flush=True)
        elif kind == "result":
            print(f"[{elapsed:7.1f}s] result:", flush=True)
            print(json.dumps(event["result"], indent=2))
            exit_code = 0
            break
        elif kind == "error":
            print(f"[{elapsed:7.1f}s] error: {json.dumps(event['error'])}", flush=True)
            break

    proc.stdin.write(json.dumps({"id": 2, "method": "shutdown"}) + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

