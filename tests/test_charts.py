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
