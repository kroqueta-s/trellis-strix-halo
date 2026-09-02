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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


BAR_WIDTH = 24


class ProgressLine:
    """Draw the runner's progress: a bar on a terminal, plain lines elsewhere.

    The runner reports `step`, and `total` as well whenever the length of the
    loop is known. **A percentage is shown only when a total arrived** - nothing
    here estimates, and there is deliberately no ETA (on this hardware the same
    loop ran 167 s per step on its first run and 14.7 s afterwards).

    Bars are drawn with `#` and `-` on purpose: the console code page on a
    Japanese Windows install cannot encode the block-drawing characters, and a
    progress bar must never be the thing that raises.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._tty = bool(getattr(stream, "isatty", lambda: False)())
        self._active = ""
        self._last_key: tuple[str, int] | None = None

    def update(
        self,
        elapsed: float,
        stage: str,
        message: str,
        step: int | None,
        total: int | None,
    ) -> None:
        """Show one progress event."""
        if step is None:
            self.note(f"[{elapsed:7.1f}s] {stage}: {message}")
            return
        if total:
            percent = min(100, int(100 * step / total))
            filled = round(BAR_WIDTH * step / total)
            bar = "#" * filled + "-" * (BAR_WIDTH - filled)
            line = f"[{elapsed:7.1f}s] {stage:<10} [{bar}] {percent:3d}%  ({step}/{total})"
            key = (stage, percent // 5)
        else:
            # No total: report the count and nothing more. **Never guess one.**
            line = f"[{elapsed:7.1f}s] {stage:<10} step {step}"
            key = (stage, step // 10)
        if self._tty:
            self._draw(line)
            if total and step >= total:
                self._end_line()
            return
        # Redirected output (a log, or CI): one line per bucket, so a long loop
        # cannot bury everything else.
        if key != self._last_key:
            self._last_key = key
            print(line, file=self._stream, flush=True)

    def note(self, text: str) -> None:
        """Print a line that is not a bar, without leaving a half-drawn bar behind."""
        if self._tty and self._active:
            self._clear()
        print(text, file=self._stream, flush=True)
        if self._tty and self._active:
            self._draw(self._active)

    def close(self) -> None:
        """Finish the current bar, if one is on screen."""
        if self._tty and self._active:
            self._end_line()

    def _draw(self, line: str) -> None:
        pad = max(0, len(self._active) - len(line))
        self._stream.write("\r" + line + " " * pad)
        self._stream.flush()
        self._active = line

    def _clear(self) -> None:
        self._stream.write("\r" + " " * len(self._active) + "\r")
        self._stream.flush()

    def _end_line(self) -> None:
        self._stream.write("\n")
        self._stream.flush()
        self._active = ""
        self._last_key = None


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
    printer = ProgressLine(sys.stdout)
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
            printer.note(f"[non-protocol line] {line}")
            continue
        elapsed = time.perf_counter() - started
        kind = event.get("event")
        if kind == "progress":
            printer.update(
                elapsed,
                str(event.get("stage", "")),
                str(event.get("message", "")),
                event.get("step"),
                event.get("total"),
            )
        elif kind == "result":
            printer.close()
            print(f"[{elapsed:7.1f}s] result:", flush=True)
            print(json.dumps(event["result"], indent=2))
            exit_code = 0
            break
        elif kind == "error":
            printer.close()
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
