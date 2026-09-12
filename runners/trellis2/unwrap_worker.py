# SPDX-License-Identifier: MIT
"""The packing, in a process of its own. **So that the heartbeat keeps beating.**

xatlas is a C extension that holds the GIL for its whole run, and
that run is minutes on a mesh this size. Every python thread in the process
stops with it - including the one that emits `heartbeat`, which is what the
caller watches for liveness. Measured 2026-09-12: ten minutes of complete
silence, which `tests/harness.py` ends at sixty seconds as a stall.

So xatlas runs somewhere else. **This module imports numpy and xatlas and
nothing else**: it is what the child process loads, and pulling torch in there
would cost seconds and memory for no reason.
"""

from __future__ import annotations

import numpy as np


def unwrap(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def pack(
    uvs: np.ndarray, faces: np.ndarray, resolution: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay charts that are already flat into one atlas. Returns `(vmapping, indices, uvs)`.

    The charts come from `charts.project_charts` in **world units**, so every
    one of them has the same texel density and xatlas only has to place them.
    That is the cheap half: measured 2.0 s where `parametrize` on the same
    200,000 faces took 31.9 s and produced charts of five triangles.

    **`texels_per_unit` is deliberately not set.** Left alone, xatlas sizes the
    atlas to fit and returns UVs in 0..1 - which is what `texture.bake`
    rasterizes. Setting it pins the atlas to `resolution` and then leaves most
    of it empty: measured on a box, utilization 0.030 against 0.727.
    `resolution` is passed as the hint it is.
    """
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_uv_mesh(
        np.ascontiguousarray(uvs, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
    )
    options = xatlas.PackOptions()
    options.resolution = int(resolution)
    # One texel of gutter, so the dilation in `texture._dilate` has somewhere to
    # spread without walking into the next chart.
    options.padding = 2
    atlas.generate(pack_options=options)
    vmapping, indices, packed = atlas[0]
    return vmapping, indices, packed
