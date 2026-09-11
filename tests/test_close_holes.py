# SPDX-License-Identifier: MIT
"""Verify hole closing on shapes whose answer is known in advance.

A closed mesh must stay closed, a mesh with one hole punched in it must come
back closed, and **the patch must be wound the same way as the face it joins** -
which shows up as a volume that stays positive rather than cancelling out.

No GPU, no weights: these are small meshes built here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis.close_holes import _directed_boundary_edges, close_holes  # noqa: E402


def _boundary(mesh: trimesh.Trimesh) -> int:
    return len(_directed_boundary_edges(mesh))


def test_a_closed_mesh_is_left_alone() -> None:
    """Nothing to close means nothing added."""
    box = trimesh.creation.box()
    closed, stats = close_holes(box, max_extent=0)
    assert stats.as_dict()["faces_added"] == 0, stats.as_dict()
    assert len(closed.faces) == len(box.faces)


def test_one_punched_hole_closes() -> None:
    """Remove a face from a box and it comes back watertight."""
    box = trimesh.creation.box()
    punched = trimesh.Trimesh(vertices=box.vertices.copy(), faces=box.faces[1:].copy(), process=False)
    assert _boundary(punched) == 3

    closed, stats = close_holes(punched, max_extent=0)
    assert _boundary(closed) == 0, stats.as_dict()
    assert closed.is_watertight, stats.as_dict()
    assert stats.loops == 1


def test_many_holes_close_and_keep_the_volume() -> None:
    """A sphere with scattered faces removed comes back closed, with its volume.

    **Watertight is not asserted here.** Where two holes share a vertex the fan
    leaves a non-manifold edge behind, which is a real limit of this function
    and is measured rather than hidden: what it promises is that the boundary
    is gone.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(0)
    drop = rng.choice(len(sphere.faces), size=40, replace=False)
    keep = np.setdiff1d(np.arange(len(sphere.faces)), drop)
    punched = trimesh.Trimesh(vertices=sphere.vertices.copy(), faces=sphere.faces[keep], process=False)

    closed, stats = close_holes(punched, max_extent=0)
    assert _boundary(closed) == 0, stats.as_dict()
    # The patches are tiny, so the volume must land back on the sphere's.
    assert closed.volume > 0, closed.volume
    assert abs(closed.volume - sphere.volume) / sphere.volume < 0.02, (
        closed.volume,
        sphere.volume,
    )


def test_a_patch_is_wound_like_its_neighbour() -> None:
    """The closed box keeps a positive volume, which a flipped patch would not.

    A patch wound the wrong way subtracts the volume it should add, so the sign
    and size of the volume is the test: **the wrong winding cannot pass it.**
    """
    box = trimesh.creation.box()
    for dropped in (0, 3, 7):
        keep = np.setdiff1d(np.arange(len(box.faces)), [dropped])
        punched = trimesh.Trimesh(
            vertices=box.vertices.copy(), faces=box.faces[keep], process=False
        )
        closed, stats = close_holes(punched, max_extent=0)
        assert _boundary(closed) == 0, stats.as_dict()
        # The apex sits on the hole, so the volume changes by less than a face.
        assert closed.volume > 0.9 * box.volume, (dropped, closed.volume, box.volume)


def test_a_wide_loop_is_left_open() -> None:
    """**A fan over a wide loop is a sail, not a repair**, so it is refused.

    Measured at 512: closing every loop covers 13.4% of the surface area with
    patch. The cap is what keeps that at 3.3%, and it has to be the loop's
    extent rather than its vertex count - the worst offenders had only a few
    hundred vertices and spanned a third of the model.
    """
    # Fine enough that one missing triangle really is a pinhole against the
    # whole model - which is the case the cap has to keep closing.
    sphere = trimesh.creation.icosphere(subdivisions=4)
    centre = sphere.triangles_center
    wide = centre[:, 2] > 0.9 * sphere.vertices[:, 2].max()
    drop = np.flatnonzero(wide)
    keep = np.setdiff1d(np.arange(len(sphere.faces)), np.append(drop, 3000))
    punched = trimesh.Trimesh(vertices=sphere.vertices.copy(), faces=sphere.faces[keep], process=False)

    closed, stats = close_holes(punched, max_extent=0.05)
    assert stats.loops_left_open >= 1, stats.as_dict()
    assert _boundary(closed) > 0, "the wide loop should still be open"
    # The narrow one is closed all the same.
    assert stats.loops >= 1, stats.as_dict()
    assert stats.fan_area_fraction < 0.01, stats.as_dict()

    everything, all_stats = close_holes(punched, max_extent=0)
    assert _boundary(everything) == 0, all_stats.as_dict()
    assert all_stats.fan_area_fraction > stats.fan_area_fraction, all_stats.as_dict()


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
