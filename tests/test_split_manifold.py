# SPDX-License-Identifier: MIT
"""Verify sheet separation and orientation on shapes whose answer is known.

**Nothing may be deleted and nothing may move**: separation duplicates vertices
and reassigns faces, so the face count is invariant and, with the default
separation of 0, every position is one that was already there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis.split_manifold import (  # noqa: E402
    count_non_manifold,
    orient_faces,
    split_non_manifold,
)


def _two_boxes_touching_at_a_vertex() -> trimesh.Trimesh:
    """Two cubes meeting at one corner: **a non-manifold vertex, manifold edges**."""
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([1.0, 1.0, 1.0])
    vertices = np.vstack([first.vertices, second.vertices])
    faces = np.vstack([first.faces, second.faces + len(first.vertices)])
    merged = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return merged


def test_a_clean_mesh_is_left_alone() -> None:
    """A box has nothing to separate, so it comes back with the same counts."""
    box = trimesh.creation.box()
    split, stats = split_non_manifold(box)
    assert len(split.faces) == len(box.faces)
    assert stats.non_manifold_edges_after == 0
    assert stats.boundary_edges_after == 0
    assert split.is_watertight


def test_faces_are_never_deleted() -> None:
    """Separation reassigns faces; it does not drop any."""
    mesh = _two_boxes_touching_at_a_vertex()
    split, _stats = split_non_manifold(mesh)
    assert len(split.faces) == len(mesh.faces)
    assert len(split.vertices) >= len(mesh.vertices)


def test_a_shared_edge_is_separated() -> None:
    """Three triangles on one edge: the edge must stop being shared by three."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    fan = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _boundary, non_manifold = count_non_manifold(fan)
    assert non_manifold == 1, non_manifold

    split, stats = split_non_manifold(fan)
    assert stats.non_manifold_edges_after == 0, stats.as_dict()
    assert len(split.faces) == 3


def test_positions_do_not_move_by_default() -> None:
    """Every vertex of the separated mesh sits on one the input already had."""
    mesh = _two_boxes_touching_at_a_vertex()
    split, _stats = split_non_manifold(mesh)
    tree = trimesh.proximity.ProximityQuery(mesh)
    distance = np.abs(tree.vertex(split.vertices)[0])
    assert float(distance.max()) < 1e-9, float(distance.max())


def test_orientation_makes_a_flipped_box_consistent() -> None:
    """One reversed face on a box is put back, and the volume comes out positive."""
    box = trimesh.creation.box()
    faces = box.faces.copy()
    faces[3] = faces[3][[0, 2, 1]]
    broken = trimesh.Trimesh(vertices=box.vertices.copy(), faces=faces, process=False)
    assert not broken.is_winding_consistent

    oriented, stats = orient_faces(broken)
    assert stats["conflicts"] == 0, stats
    assert oriented.is_winding_consistent, stats
    assert oriented.volume > 0, oriented.volume
    assert abs(oriented.volume - box.volume) < 1e-9


def test_an_inside_out_box_is_turned_out() -> None:
    """A consistently inward box is flipped whole, not face by face."""
    box = trimesh.creation.box()
    inward = trimesh.Trimesh(
        vertices=box.vertices.copy(), faces=box.faces[:, [0, 2, 1]].copy(), process=False
    )
    assert inward.volume < 0

    oriented, stats = orient_faces(inward)
    assert stats["flipped_whole"] == 1, stats
    assert oriented.volume > 0, oriented.volume


def test_make_manifold_produces_a_manifold() -> None:
    """Two boxes joined at a vertex come back closed, oriented and manifold."""
    from runners.trellis.split_manifold import make_manifold

    mesh = _two_boxes_touching_at_a_vertex()
    made, report = make_manifold(mesh)
    assert report["boundary_edges"] == 0, report
    assert report["non_manifold_edges"] == 0, report
    assert report["watertight"], report
    assert report["winding_consistent"], report
    assert made.volume > 0, made.volume


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
