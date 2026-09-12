# SPDX-License-Identifier: MIT
"""Verify the compiled mesh -> dual grid conversion, **when it has been built**.

`native/o_voxel_cpu/` is optional, so a missing module is reported and passes:
the point of this script is that a module which *is* there does what upstream's
does. The check is a round trip - a box and a sphere go through the conversion
and back through the runner's own extraction - which lands within a cell of
the input when the two agree on where cells and vertices are.

Run it with the TRELLIS.2 virtual environment; it reads TRELLIS2_NATIVE_DIR
from `.env`, or `O_VOXEL_CPU_DIR` from the environment.
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

from runners.trellis import shims  # noqa: E402


def _where() -> str:
    override = os.environ.get("O_VOXEL_CPU_DIR")
    if override:
        return override
    try:
        from runners.trellis2 import config

        return config.NATIVE_DIR
    except Exception:  # noqa: BLE001 - a missing .env is the "not built" case too
        return ""


def _round_trip(mesh: trimesh.Trimesh, grid: int) -> tuple[int, int, float]:
    """(voxels, crossings, centroid offset in cells) through conversion and extraction."""
    convert = sys.modules["o_voxel._C"].mesh_to_flexible_dual_grid_cpu
    v = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    f = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int32)
    # Upstream's python wrapper pads the box by half a cell and shifts the
    # vertices to its origin; the same is done here.
    lo, hi = v.min(0).values, v.max(0).values
    size = torch.tensor([grid] * 3, dtype=torch.int32)
    padding = (hi - lo) / (size.float() - 1)
    lo, hi = lo - padding * 0.5, hi + padding * 0.5
    voxel = (hi - lo) / size.float()
    grid_range = torch.stack([torch.zeros_like(size), size]).int()
    coords, dual, flags = convert(v - lo, f, voxel, grid_range, 1.0, 1.0, 0.1, False)
    verts, faces = shims.flexible_dual_grid_to_mesh(
        coords.long(), dual.float(), flags.bool(), aabb=torch.stack([lo, hi]), grid_size=grid
    )
    out = trimesh.Trimesh(verts.numpy(), faces.numpy(), process=False)
    offset = (out.vertices.mean(0) - np.asarray(mesh.vertices).mean(0)) / voxel.numpy()
    return int(len(coords)), int(flags.sum()), float(np.abs(offset).max())


def main() -> int:
    directory = _where()
    if not shims.install_o_voxel_cpu(directory):
        print(f"  SKIP o_voxel_cpu is not built (looked in {directory!r})")
        print("\n0/0 passed (nothing to test)")
        return 0

    failures = 0
    for label, mesh, grid, tolerance in (
        ("box", trimesh.creation.box(), 32, 0.1),
        ("sphere", trimesh.creation.icosphere(subdivisions=3), 48, 1.0),
    ):
        voxels, crossings, offset = _round_trip(mesh, grid)
        ok = voxels > 0 and crossings > 0 and offset < tolerance
        failures += 0 if ok else 1
        print(
            f"  {'OK  ' if ok else 'FAIL'} {label}: {voxels} voxels, {crossings} crossings, "
            f"centroid offset {offset:.3f} cells (limit {tolerance})"
        )
    print(f"\n{2 - failures}/2 passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
