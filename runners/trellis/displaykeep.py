# SPDX-License-Identifier: MIT
"""Display keepalive: hold the console display awake while generating.

**Why it exists** (measured 2026-09-02, method and data in gfx1151-gemm's
``docs/displayoff.md``): when the console display turns off — lid closed, or
the display-off timeout expiring, locked or not — the AMD Windows driver pins
the GPU near 600 MHz, and generation runs about 4x slower until the display
comes back. A render loop does not help in that state; holding
``SetThreadExecutionState(ES_DISPLAY_REQUIRED)``, the same API media players
use, prevents it completely.

**It prevents; it cannot rescue.** If the display is already off when
generation starts, the clock stays pinned — the hold only stops the display
from turning off later.

**Default off.** Keeping the display lit during every generation is a real
trade (an OLED panel, a laptop battery), and an operator who has disabled
display-off timeouts machine-wide needs nothing from this switch. Turn it on
(`TRELLIS_DISPLAY_KEEPALIVE`) when generations run unattended on a machine
whose display is allowed to sleep. Whether the hold was in effect is
reported in `metrics.display_keepalive`.

**Start and stop must run on the same thread** — the execution state is
per-thread and continuous until that thread clears it. The runner calls both
from its request loop.

This file is identical in all three runners, which do not share a module
because each ships as its own repository: **fix one and fix the others.**
"""

from __future__ import annotations

import ctypes
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class DisplayKeep:
    """Holds the display-required execution state. **Never raises when it fails.**"""

    def __init__(self) -> None:
        self._held = False

    def start(self) -> None:
        """Ask Windows to keep the display awake. Generation continues if this fails."""
        if sys.platform != "win32":
            return
        try:
            rc = ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED
            )
            self._held = bool(rc)
        except OSError as exc:
            print(f"[displaykeep] not held (generation continues): {exc}", file=sys.stderr)
            self._held = False

    def is_held(self) -> bool:
        """Whether the hold took effect. Recorded in metrics."""
        return self._held

    def stop(self) -> None:
        """Release the hold. Must run on the thread that called `start`."""
        if not self._held:
            return
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except OSError:
            pass
        self._held = False
