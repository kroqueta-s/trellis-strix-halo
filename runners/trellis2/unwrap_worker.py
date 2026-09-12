# SPDX-License-Identifier: MIT
"""The unwrap, in a process of its own. **So that the heartbeat keeps beating.**

`xatlas.parametrize` is a C extension that holds the GIL for its whole run, and
that run is minutes on a mesh this size. Every python thread in the process
stops with it - including the one that emits `heartbeat`, which is what the
caller watches for liveness. Measured 2026-09-12: ten minutes of complete
silence, which `tests/harness.py` ends at sixty seconds as a stall.

So the unwrap happens somewhere else. **This module imports numpy and xatlas and
nothing else**: it is what the child process loads, and pulling torch in there
would cost seconds and memory for no reason.
"""

from __future__ import annotations

import numpy as np


def unwrap(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut the surface open and lay it flat. Returns `(vmapping, indices, uvs)`.

    **xatlas's own defaults.** They were compared against a relaxed chart search
    (`max_iterations=1`, `max_cost=8`, the shape weights at zero, no brute-force
    packing) on 100,000 faces: 13.8 s against 14.8 s, for the same charts. The
    cost is not in the settings, so there is nothing here to tune.
    """
    import xatlas

    return xatlas.parametrize(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
    )
