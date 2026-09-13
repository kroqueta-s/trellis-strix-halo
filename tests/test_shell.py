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
from runners.trellis2.shell import solidify, thicken  # noqa: E402

# The rays run on the CPU here: the lattices are small, and the tests must
# not need a GPU.
CPU = "cpu"


def _sphere(radius: float = 1.0) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=4, radius=radius)


def test_carving_keeps_the_surface_and_fills_the_inside() -> None:
    """A sphere carved by visibility is a ball of the sphere's own radius."""
    sphere = _sphere()
    grid = 64
    solid, report = solidify(sphere, grid=grid, mode="carve", device=CPU)
    assert report.mode == "carve"
    boundary, _ = count_non_manifold(solid)
    assert boundary == 0
    made = _closed(solid)
    want = 4 / 3 * np.pi
    assert abs(made.volume - want) / want < 0.08, (made.volume, want)
    # The surface stays where it was: no growth outward beyond a cell.
    cell = sphere.extents.max() / (grid - 1)
    radius = np.linalg.norm(made.vertices - made.vertices.mean(axis=0), axis=1)
    assert radius.max() < 1.0 + 1.5 * cell, (radius.max(), cell)
    assert np.median(radius) < 1.0 + 0.75 * cell, (np.median(radius), cell)


def test_carving_fills_a_hollow_with_a_gap() -> None:
    """A double-walled shell with a hole in it comes back as one solid ball.

    Two concentric spheres with a patch cut out of each: a flood from outside
    would pour through the gaps and call the whole inside air; the rays do not.
    """
    outer = _sphere(1.0)
    inner = _sphere(0.85)
    keep_o = outer.triangles_center[:, 2] < 0.9
    keep_i = inner.triangles_center[:, 0] < 0.8
    soup = trimesh.util.concatenate(
        [
            trimesh.Trimesh(outer.vertices, outer.faces[keep_o], process=False),
            trimesh.Trimesh(inner.vertices, inner.faces[keep_i], process=False),
        ]
    )
    solid, report = solidify(soup, grid=64, mode="carve", device=CPU)
    made = _closed(solid)
    want = 4 / 3 * np.pi
    assert abs(made.volume - want) / want < 0.1, (made.volume, want, report.as_dict())


def test_carving_leaves_a_concavity_open() -> None:
    """A cup's bowl sees the sky, so it stays air; only what is behind the wall fills."""
    cup = trimesh.creation.annulus(r_min=0.6, r_max=1.0, height=1.0)
    # Add a bottom so that the cup encloses a solid ring plus a floor.
    bottom = trimesh.creation.cylinder(radius=1.0, height=0.2)
    bottom.apply_translation((0, 0, -0.6))
    soup = trimesh.util.concatenate([cup, bottom])
    solid, _report = solidify(soup, grid=64, mode="carve", device=CPU)
    made = _closed(solid)
    # Ring wall (pi (1 - 0.36) * 1.0) plus the floor (pi * 0.2): the bowl is empty.
    want = np.pi * (1.0 - 0.36) * 1.0 + np.pi * 0.2
    assert abs(made.volume - want) / want < 0.15, (made.volume, want)


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
    assert report.pockets_filled >= 1, report
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


def test_islands_smaller_than_the_threshold_become_air() -> None:
    """A speck of solid detached from the body is dropped; the body is kept whatever its size."""
    from scipy import ndimage

    from runners.trellis2.shell import _drop_islands

    six = ndimage.generate_binary_structure(3, 1)
    solid = np.zeros((20, 20, 20), dtype=bool)
    solid[2:8, 2:8, 2:8] = True  # the body, 216 corners
    solid[15, 15, 15] = True  # a single stray corner
    solid[12:14, 2:4, 2:4] = True  # a 2x2x2 speck, 8 corners
    solid[15:18, 2:6, 2:6] = True  # 48 corners, only 6-connected to nothing
    out, dropped, corners = _drop_islands(solid, 64, six)
    assert (dropped, corners) == (3, 1 + 8 + 48), (dropped, corners)
    assert out.sum() == 216, out.sum()
    # Nothing is dropped at 0, and the largest piece survives any threshold.
    same, dropped, _ = _drop_islands(solid, 0, six)
    assert dropped == 0 and same.sum() == solid.sum()
    only_body, dropped, _ = _drop_islands(solid, 10_000, six)
    assert dropped == 3 and only_body.sum() == 216


def test_a_speck_in_the_air_is_not_part_of_the_carved_solid() -> None:
    """A tetrahedron smaller than a cell, floating beside a sphere, leaves no part behind.

    Its samples mark one corner of the lattice, which the carve would keep as
    a one-corner solid; the report counts it, and the mesh has one part.
    """
    sphere = _sphere()
    speck = trimesh.creation.icosphere(subdivisions=0, radius=0.005)
    speck.apply_translation((1.4, 0.0, 0.0))
    soup = trimesh.util.concatenate([sphere, speck])
    grid = 64
    kept, report_kept = solidify(soup, grid=grid, mode="carve", device=CPU, island_corners=0)
    assert report_kept.islands_dropped == 0
    assert len(trimesh.graph.connected_components(kept.face_adjacency)) == 2
    solid, report = solidify(soup, grid=grid, mode="carve", device=CPU)
    assert report.islands_dropped == 1, report.as_dict()
    assert report.island_corners_dropped >= 1, report.as_dict()
    assert len(trimesh.graph.connected_components(solid.face_adjacency)) == 1
    assert report.solid_corners == report_kept.solid_corners - report.island_corners_dropped


def test_the_smoothing_takes_the_terraces_off_and_leaves_the_volume() -> None:
    """Taubin's two passes remove the lattice's steps without shrinking the solid.

    A sphere is the case that shows both halves: every step on it is the
    lattice's, so roughness has to fall a long way, and a Laplacian alone would
    pull it in. The volume is what says the negative pass is doing its job.
    """
    ball = trimesh.creation.icosphere(subdivisions=4)
    rough, report = solidify(ball, grid=64, mode="carve", smooth=0)
    smooth, smooth_report = solidify(ball, grid=64, mode="carve", smooth=5)

    assert report.smooth_rounds == 0, report.smooth_rounds
    assert smooth_report.smooth_rounds == 5, smooth_report.smooth_rounds
    assert len(smooth.vertices) == len(rough.vertices), "smoothing must not retopologize"
    assert len(smooth.faces) == len(rough.faces)

    def roughness(mesh: trimesh.Trimesh) -> float:
        """Mean distance from a vertex to the average of its neighbours."""
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e = np.concatenate([e, e[:, ::-1]])
        total = np.zeros_like(v)
        np.add.at(total, e[:, 0], v[e[:, 1]])
        count = np.bincount(e[:, 0], minlength=len(v)).astype(np.float64)
        live = count > 0
        return float(np.linalg.norm(total[live] / count[live, None] - v[live], axis=1).mean())

    before, after = roughness(rough), roughness(smooth)
    assert after < 0.5 * before, (before, after)
    assert 0.97 < smooth.volume / rough.volume < 1.03, smooth.volume / rough.volume
    # **The report has to say how far it moved**, because that is the only
    # number telling an operator whether the surface they asked to keep is
    # still where they left it.
    assert 0.0 < smooth_report.smooth_moved_cells < 1.0, smooth_report.smooth_moved_cells


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
