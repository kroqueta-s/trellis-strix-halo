# SPDX-License-Identifier: MIT
"""Verify the UV bake **without a GPU, weights or a model**.

The bake has one part that can be wrong quietly: the map from a texel to the
triangle under it and the barycentric weights inside it. Everything downstream
takes that on trust, and a texture that is subtly wrong looks like a texture.

**It has an exact reference.** Give the mesh vertices whose 3D positions *are*
their UV coordinates, and the position interpolated for a texel must come back
as that texel's own centre - to floating-point. Nothing about the geometry,
the model or the colours enters into it.

Run it with either virtual environment; it falls back to CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis2 import texture  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _unit_quad() -> tuple[torch.Tensor, torch.Tensor]:
    """Two triangles covering the whole atlas, wound the same way."""
    uvs = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32, device=DEVICE
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=DEVICE)
    return uvs, faces


def test_a_full_quad_covers_every_texel_once() -> None:
    """**A gap is a hole in the texture and an overlap is a seam**, so count both."""
    size = 16
    uvs, faces = _unit_quad()
    texel, _triangle, _weights = texture._cover(uvs, faces, size)
    counts = torch.bincount(texel, minlength=size * size)
    assert int(counts.min()) >= 1, f"{int((counts == 0).sum())} texels uncovered"
    assert int(counts.max()) <= 2, f"a texel was claimed {int(counts.max())} times"
    # Only the shared diagonal may be claimed twice.
    assert int((counts > 1).sum()) <= size, int((counts > 1).sum())


def test_the_interpolated_position_is_the_texel_centre() -> None:
    """The exact reference: positions that are the UVs must reproduce the grid.

    This is the whole chain - bounding box, inside test, barycentric weights and
    the weighted sum - checked against arithmetic that cannot drift.
    """
    size = 32
    uvs, faces = _unit_quad()
    # z is left at zero; x and y are the UVs themselves.
    vertices = torch.cat([uvs, torch.zeros_like(uvs[:, :1])], dim=1)

    texel, triangle, weights = texture._cover(uvs, faces, size)
    corners = vertices[faces[triangle]]
    point = (corners * weights.unsqueeze(-1)).sum(dim=1)

    x = (texel % size).float()
    y = (texel // size).float()
    want = torch.stack([(x + 0.5) / size, (y + 0.5) / size], dim=1)
    error = (point[:, :2] - want).abs().max().item()
    assert error < 1e-5, f"max difference {error}"
    assert float(point[:, 2].abs().max()) < 1e-6, "z drifted away from the plane"


def test_the_weights_are_a_partition() -> None:
    """Barycentric weights sum to one and are never negative inside the triangle."""
    size = 24
    uvs, faces = _unit_quad()
    _texel, _triangle, weights = texture._cover(uvs, faces, size)
    assert float(weights.min()) >= -1e-6, float(weights.min())
    assert float((weights.sum(dim=1) - 1).abs().max()) < 1e-5


def test_a_degenerate_triangle_claims_nothing() -> None:
    """**A triangle with no area in the atlas must not divide by it.**"""
    uvs = torch.tensor(
        [[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], dtype=torch.float32, device=DEVICE
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=DEVICE)
    texel, _triangle, weights = texture._cover(uvs, faces, 16)
    assert texel.numel() == 0, f"a flat triangle claimed {texel.numel()} texels"
    assert not bool(torch.isnan(weights).any())


def test_a_chart_that_misses_the_atlas_is_dropped() -> None:
    """A triangle outside the square has an empty box and must not be indexed."""
    uvs = torch.tensor(
        [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0]], dtype=torch.float32, device=DEVICE
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=DEVICE)
    texel, _triangle, _weights = texture._cover(uvs, faces, 8)
    assert texel.numel() == 0, texel.numel()


def test_dilation_spreads_outward_and_stops() -> None:
    """**Without it every seam draws a black line**; with too much it floods."""
    size = 9
    colour = torch.zeros(size, size, 3, device=DEVICE)
    filled = torch.zeros(size, size, dtype=torch.bool, device=DEVICE)
    colour[4, 4] = torch.tensor([1.0, 0.5, 0.25], device=DEVICE)
    filled[4, 4] = True

    grown, rounds = texture._dilate(colour, filled, 2)
    assert rounds == 2, rounds
    # One round reaches the eight neighbours, two reaches the 5x5 block.
    assert torch.allclose(grown[3, 3], torch.tensor([1.0, 0.5, 0.25], device=DEVICE), atol=1e-5)
    assert float(grown[2, 2].sum()) > 0, "the second round did not reach"
    assert float(grown[0, 0].sum()) == 0.0, "it spread further than it was asked to"

    # It stops early rather than looping once everything is filled.
    _full, done = texture._dilate(
        torch.ones(4, 4, 3, device=DEVICE), torch.ones(4, 4, dtype=torch.bool, device=DEVICE), 5
    )
    assert done == 0, done


def test_the_unwrap_keeps_the_surface() -> None:
    """xatlas cuts and duplicates vertices; **it must not drop faces**."""
    try:
        import trimesh
        import xatlas  # noqa: F401
    except ImportError as exc:
        print(f"  skipped (no {exc.name})")
        return
    mesh = trimesh.creation.icosphere(subdivisions=2)
    vertices, faces, uvs, report = texture.unwrap(mesh, in_process=True)
    assert len(faces) == len(mesh.faces), (len(faces), len(mesh.faces))
    assert len(vertices) == len(uvs) >= len(mesh.vertices)
    assert report.vertices_before == len(mesh.vertices)
    assert float(np.asarray(uvs).min()) >= -1e-6 and float(np.asarray(uvs).max()) <= 1 + 1e-6


def main() -> int:
    """Run every test."""
    print(f"device: {DEVICE}")
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
