# SPDX-License-Identifier: MIT
"""Confirm that `runners/trellis/fill_holes.py` **produces the same result as upstream**.

It is a transcription, so **comparing it against upstream is the whole point.**
Upstream's `_fill_holes` is slow because of the Python-side inefficiency, but
**a small mesh takes only seconds**, which is where the one-to-one comparison
happens.

Run it with this repository's virtual environment (torch and the upstream clone
are required).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import config  # noqa: E402

if config.FAST_ATTENTION:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

from runners.trellis import fill_holes as ours  # noqa: E402
from runners.trellis import raster, shims  # noqa: E402

VIEWS = 30
RESOLUTION = 256


def _prepare() -> None:
    """Make upstream importable and install the rasterizer replacement."""
    shims.install(head_chunk=config.ATTN_HEAD_CHUNK)
    shims.install_absent_nvdiffrast()
    raster.install()
    ours.ensure_upstream_on_path(str(config.TRELLIS_REPO))


def _sample_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    """Build a small mesh that **has an interior shell**.

    With nothing to cut there would be nothing to compare.
    """
    outer = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    inner = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    verts = np.concatenate([outer.vertices, inner.vertices], axis=0)
    faces = np.concatenate([outer.faces, inner.faces + len(outer.vertices)], axis=0)
    return (
        torch.tensor(verts, dtype=torch.float32, device="cuda"),
        torch.tensor(faces, dtype=torch.int32, device="cuda"),
    )


def test_visibility_matches_upstream_definition() -> None:
    """**The interior shell has visibility ratio 0.**

    That is the condition upstream feeds to the min-cut source.
    """
    _prepare()
    verts, faces = _sample_mesh()
    vis = ours.visibility(verts, faces, RESOLUTION, VIEWS)
    n_outer = 1280  # face count of icosphere(3)
    assert (vis[:n_outer] > 0).all(), "some faces of the outer sphere are never visible"
    assert (
        vis[n_outer:] == 0
    ).all(), f"the interior shell was visible: {vis[n_outer:].max().item()}"


def test_same_result_as_upstream() -> None:
    """**The same vertices and faces come out as from upstream's `_fill_holes`.**

    It is a transcription, so a disagreement here would defeat the purpose.
    """
    _prepare()
    from trellis.utils import postprocessing_utils

    verts, faces = _sample_mesh()
    theirs_v, theirs_f = postprocessing_utils._fill_holes(
        verts.clone(),
        faces.clone(),
        max_hole_size=0.04,
        max_hole_nbe=250,
        resolution=RESOLUTION,
        num_views=VIEWS,
    )
    ours_v, ours_f = ours.fill_holes(
        verts.clone(),
        faces.clone(),
        max_hole_size=0.04,
        max_hole_nbe=250,
        resolution=RESOLUTION,
        num_views=VIEWS,
    )
    assert (
        ours_v.shape == theirs_v.shape
    ), f"vertex counts differ: {ours_v.shape} != {theirs_v.shape}"
    assert ours_f.shape == theirs_f.shape, f"face counts differ: {ours_f.shape} != {theirs_f.shape}"
    assert torch.allclose(ours_v, theirs_v, atol=1e-5), "vertices do not match"
    assert torch.equal(ours_f, theirs_f), "faces do not match"


def test_faster_than_upstream() -> None:
    """**Faster than upstream** (transcribing it was pointless otherwise)."""
    import time

    _prepare()
    from trellis.utils import postprocessing_utils

    verts, faces = _sample_mesh()
    t0 = time.perf_counter()
    postprocessing_utils._fill_holes(
        verts.clone(), faces.clone(), resolution=RESOLUTION, num_views=VIEWS, max_hole_nbe=250
    )
    upstream = time.perf_counter() - t0
    t0 = time.perf_counter()
    ours.fill_holes(
        verts.clone(), faces.clone(), resolution=RESOLUTION, num_views=VIEWS, max_hole_nbe=250
    )
    mine = time.perf_counter() - t0
    print(f"       upstream {upstream:.2f}s / ours {mine:.2f}s")
    assert mine <= upstream * 1.1, f"not faster: upstream {upstream:.2f}s / ours {mine:.2f}s"


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
