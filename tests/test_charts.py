# SPDX-License-Identifier: MIT
"""Verify the axis-projection charts on shapes whose answer is known.

A box is six charts, one per face direction, with nothing folded; a sphere
is a handful of charts covering nearly every face; and every face of every
chart projects with a positive area, because a fold would overlap.

No GPU, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis2.charts import project_charts  # noqa: E402


def test_a_box_is_six_charts() -> None:
    box = trimesh.creation.box()
    vertices, faces, uvs, report = project_charts(box, smoothing_rounds=0, min_faces=1)
    assert report.charts == 6, report.as_dict()
    assert report.folded_faces == 0, report.as_dict()
    # Every face keeps its three corners, and a corner shared by three charts
    # exists three times.
    assert len(faces) == len(box.faces)
    assert report.vertices_after == 24, report.as_dict()
    # The projection is orthographic: within a chart, u and v are two of the
    # box's own coordinates, so a unit box gives unit-wide charts.
    assert uvs.shape == (24, 2)
    assert np.allclose(np.ptp(uvs, axis=0), 1.0)


def test_a_sphere_is_a_few_large_charts_with_no_folds() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=4)
    _v, _f, _uv, report = project_charts(sphere, smoothing_rounds=3, min_faces=50)
    assert report.charts <= 12, report.as_dict()
    assert report.faces_in_large_charts > 0.99, report.as_dict()
    assert report.folded_faces == 0, report.as_dict()


def test_small_charts_are_merged_into_compatible_neighbours() -> None:
    """A box with one face dented: the dent's tilted faces join the face they sit in."""
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    box = box.subdivide().subdivide()
    vertices = box.vertices.copy()
    top = np.isclose(vertices[:, 2], 1.0)
    inner = top & (np.abs(vertices[:, 0]) < 0.6) & (np.abs(vertices[:, 1]) < 0.6)
    vertices[inner, 2] -= 0.15
    dented = trimesh.Trimesh(vertices=vertices, faces=box.faces, process=False)
    _v, _f, _uv, report = project_charts(dented, smoothing_rounds=2, min_faces=20)
    assert report.charts == 6, report.as_dict()
    assert report.folded_faces == 0, report.as_dict()


def test_an_inconsistently_wound_mesh_shatters() -> None:
    """Half the faces wound the other way: each points the wrong way and is its own chart.

    The direction is signed, so a flipped face lands in the opposite class
    from its neighbours and no chart can grow across it - which is the
    failure the charts must be protected from by orienting the mesh first.
    The count of charts says so; the folds do not, because a flipped face
    projects consistently with its own (wrong) normal.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3)
    faces = sphere.faces.copy()
    faces[::2] = faces[::2][:, [0, 2, 1]]
    messy = trimesh.Trimesh(vertices=sphere.vertices, faces=faces, process=False)
    _v, _f, _uv, report = project_charts(messy, smoothing_rounds=0, min_faces=1)
    _v, _f, _uv, clean = project_charts(sphere, smoothing_rounds=0, min_faces=1)
    assert report.charts > len(faces) / 4, report.as_dict()
    assert report.charts > 10 * clean.charts, (report.charts, clean.charts)


def test_folded_faces_are_moved_or_given_a_chart() -> None:
    """A face pointing away from its chart's axis goes to a neighbour it fits, or gets its own.

    Four faces in a row, all in a +Z chart but face 2, which is +X. Face 1
    points -Z and borders face 2 while pointing +X too, so it moves there;
    face 3 points -Z, borders only +Z faces, and is large, so it gets a chart
    of its own along -Z; face 0 is a -Z sliver with nowhere to go, and stays.
    """
    from runners.trellis2.charts import _AXIS_VECTORS, _unfold

    chart = np.array([0, 0, 1, 0])
    chart_direction = np.array([4, 0])  # +Z, +X
    normals = np.array(
        [
            [0.0, 0.0, -1.0],  # sliver, folded, no neighbour to take it
            [0.6, 0.0, -0.8],  # folded in +Z, fits +X
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],  # folded, large, only +Z neighbours
        ]
    )
    area = np.array([0.1, 1.0, 1.0, 5.0])
    left = np.array([0, 1, 1, 2])
    right = np.array([1, 2, 3, 3])
    out, directions, moved, own = _unfold(chart, chart_direction, normals, area, left, right, 2.0)
    assert (moved, own) == (1, 1), (moved, own)
    assert out[1] == out[2], out
    assert out[0] != out[3] and out[3] != out[1], out
    assert len(directions) == 3
    assert np.array_equal(_AXIS_VECTORS[directions[out[3]]], [0, 0, -1]), directions
    # Nothing folds any more except the sliver that was left.
    dot = (normals * _AXIS_VECTORS[directions[out]]).sum(axis=1)
    assert np.array_equal(dot <= 0, [True, False, False, False]), dot


def test_a_folded_triangle_in_a_flat_grid_gets_its_own_chart() -> None:
    """One vertex dragged past an opposite edge turns one triangle over; it leaves the chart.

    Its smoothed normal still says +Z, so the chart takes it, and its own
    says -Z, so it folds. Kept folded when the area threshold is out of
    reach; given a chart of its own when the threshold is zero.
    """
    n = 9
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            faces += [[a, b, c], [a, c, d]]
    vertices[4 * n + 4, 0] += 1.2
    vertices[4 * n + 4, 1] += 0.3
    grid = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    _v, _f, _uv, kept = project_charts(grid, smoothing_rounds=10, min_faces=4, fold_area=1e9)
    assert kept.folded_faces == 1, kept.as_dict()
    assert kept.folds_own_chart == 0, kept.as_dict()
    _v, _f, _uv, report = project_charts(grid, smoothing_rounds=10, min_faces=4, fold_area=0.0)
    assert report.folded_faces == 0, report.as_dict()
    assert report.folds_own_chart == kept.folded_faces, report.as_dict()
    assert report.charts == kept.charts + kept.folded_faces, (report.charts, kept.charts)


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
