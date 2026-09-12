# SPDX-License-Identifier: MIT
"""Verify the shell on shapes whose solid is known in advance.

A sphere's surface thickened by a wall is a spherical shell of known volume; a
closed sphere with cavity filling is a ball; an open surface - a sphere with a
patch missing, a Möbius strip - still comes back as a closed, consistently
wound solid, because only *where* the surface is matters.

No GPU, no weights. Small lattices keep it under a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis.split_manifold import count_non_manifold, make_manifold  # noqa: E402
from runners.trellis2.shell import thicken  # noqa: E402


def _sphere(radius: float = 1.0) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=4, radius=radius)


def _closed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Separate the sheets surface nets leave touching, and close the seams."""
    made, report = make_manifold(mesh)
    assert report["boundary_edges"] == 0, report
    assert report["non_manifold_edges"] == 0, report
    assert report["winding_consistent"], report
    return made


def test_a_hollow_sphere_becomes_a_shell_of_the_asked_thickness() -> None:
    sphere = _sphere()
    grid = 96
    shell, report = thicken(sphere, grid=grid, thickness=0.2, fill_cavities=False)
    assert report.half_cells == round(0.2 * grid / 2)
    boundary, _ = count_non_manifold(shell)
    assert boundary == 0, "surface nets over a closed occupancy leave no boundary"
    made = _closed(shell)
    cell = sphere.extents.max() / (grid - 1)
    half = report.half_cells * cell
    want = 4 / 3 * np.pi * ((1 + half) ** 3 - (1 - half) ** 3)
    assert abs(made.volume - want) / want < 0.15, (made.volume, want)
    assert made.volume > 0


def test_a_closed_sphere_is_filled_when_asked() -> None:
    sphere = _sphere()
    grid = 96
    shell, report = thicken(sphere, grid=grid, thickness=0.1, fill_cavities=True)
    assert report.cavities_filled >= 1, report
    made = _closed(shell)
    cell = sphere.extents.max() / (grid - 1)
    half = report.half_cells * cell
    want = 4 / 3 * np.pi * (1 + half) ** 3
    assert abs(made.volume - want) / want < 0.1, (made.volume, want)


def test_an_open_surface_still_comes_back_closed() -> None:
    """A sphere with its top removed: the wall wraps the cut edge and closes."""
    sphere = _sphere()
    keep = sphere.triangles_center[:, 2] < 0.7
    punched = trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces[keep], process=False)
    shell, _report = thicken(punched, grid=96, thickness=0.1, fill_cavities=False)
    boundary, _ = count_non_manifold(shell)
    assert boundary == 0
    made = _closed(shell)
    assert made.volume > 0


def test_orientation_does_not_matter() -> None:
    """Half the faces flipped at random give the same shell as the clean sphere."""
    sphere = _sphere()
    rng = np.random.default_rng(0)
    faces = sphere.faces.copy()
    flip = rng.random(len(faces)) < 0.5
    faces[flip] = faces[flip][:, [0, 2, 1]]
    messy = trimesh.Trimesh(vertices=sphere.vertices, faces=faces, process=False)
    clean_shell, _ = thicken(sphere, grid=64, thickness=0.1)
    messy_shell, _ = thicken(messy, grid=64, thickness=0.1)
    assert len(clean_shell.faces) == len(messy_shell.faces)
    assert np.allclose(clean_shell.vertices, messy_shell.vertices)
    assert np.array_equal(clean_shell.faces, messy_shell.faces)


def test_the_wall_is_never_thinner_than_asked() -> None:
    """A flat sheet becomes a slab at least the wall thick."""
    sheet = trimesh.creation.box(extents=(1.0, 1.0, 0.001))
    shell, report = thicken(sheet, grid=64, thickness=0.2, fill_cavities=True)
    made = _closed(shell)
    cell = 1.0 / (64 - 1)
    assert made.extents[2] >= 2 * report.half_cells * cell * 0.9, (made.extents, report)


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
