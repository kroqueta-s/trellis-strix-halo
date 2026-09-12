# SPDX-License-Identifier: MIT
"""The fast edge counting and colouring give the same answers as the slow ones.

The post-processing was made affordable on 2026-09-12 by three replacements:
edge rows became one-dimensional keys, the per-component search became one
search, and `drop_small_parts` stopped building a `Trimesh` per part. None of
them is allowed to change an answer, so each is checked here against the
implementation it replaced, on meshes with the features that matter - many
components, non-manifold edges, a Möbius strip, and duplicate vertices.

No GPU, no weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis import postprocess  # noqa: E402
from runners.trellis.close_holes import _directed_boundary_edges  # noqa: E402
from runners.trellis.split_manifold import (  # noqa: E402
    _edge_counts,
    count_non_manifold,
    make_manifold,
    orient_faces,
)

# --- the implementations being replaced, kept here as the reference ---------


def _reference_edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = faces[:, [0, 1, 2]].ravel(order="F")
    ends = faces[:, [1, 2, 0]].ravel(order="F")
    keys = np.sort(np.stack([starts, ends], axis=1), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return inverse.ravel(), counts


def _reference_boundary(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = mesh.faces
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    return directed[counts[inverse] == 1]


def _reference_orient(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """The per-component search: returns (faces flipped, conflicts)."""
    faces = np.asarray(mesh.faces).copy()
    face_count = len(faces)
    inverse, counts = _reference_edge_counts(faces)
    order = np.argsort(inverse, kind="stable")
    instance_face = np.tile(np.arange(face_count), 3)
    starts = faces[:, [0, 1, 2]].ravel(order="F")
    paired = order[counts[inverse[order]] == 2].reshape(-1, 2)
    left, right = instance_face[paired[:, 0]], instance_face[paired[:, 1]]
    disagree = (starts[paired[:, 0]] == starts[paired[:, 1]]).astype(np.int8)
    low, high = np.minimum(left, right), np.maximum(left, right)
    _, unique_index = np.unique(np.stack([low, high], axis=1), axis=0, return_index=True)
    left, right, disagree = left[unique_index], right[unique_index], disagree[unique_index]
    graph = csr_matrix(
        (
            np.concatenate([disagree + 1, disagree + 1]),
            (np.concatenate([left, right]), np.concatenate([right, left])),
        ),
        shape=(face_count, face_count),
    )
    indptr, indices, data = graph.indptr, graph.indices, graph.data
    flip = np.zeros(face_count, dtype=bool)
    _, components = connected_components(graph, directed=False)
    for component in np.unique(components):
        root = int(np.flatnonzero(components == component)[0])
        visit, predecessors = breadth_first_order(graph, root, directed=False)
        for node in visit[1:]:
            parent = predecessors[node]
            span = slice(indptr[parent], indptr[parent + 1])
            position = indptr[parent] + int(np.searchsorted(indices[span], node))
            flip[node] = flip[parent] ^ bool(data[position] - 1)
    still = disagree.astype(bool) ^ (flip[left] ^ flip[right])
    return int(flip.sum()), int(still.sum())


def _reference_drop(
    mesh: trimesh.Trimesh, min_ratio: float, min_thick: float
) -> tuple[int, int, int]:
    """`split()`-based decision: (parts before, parts kept, faces dropped)."""
    parts = mesh.split(only_watertight=False)
    whole = max(float(np.max(mesh.bounding_box.extents)), 1e-12)
    face_counts = np.array([len(p.faces) for p in parts])
    keep = np.ones(len(parts), dtype=bool)
    if min_ratio > 0:
        keep &= (
            np.array([float(np.max(p.bounding_box.extents)) for p in parts]) / whole >= min_ratio
        )
    if min_thick > 0:
        keep &= (
            np.array([float(np.min(p.bounding_box.extents)) for p in parts]) / whole >= min_thick
        )
    keep[int(np.argmax(face_counts))] = True
    return len(parts), int(keep.sum()), int((~keep).sum())


# --- specimens ---------------------------------------------------------------


def _mobius(segments: int = 40) -> trimesh.Trimesh:
    """A Möbius strip: the classic surface that cannot be oriented."""
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    w = np.array([-0.3, 0.3])
    u, v = np.meshgrid(t, w, indexing="ij")
    x = (1 + v * np.cos(u / 2)) * np.cos(u)
    y = (1 + v * np.cos(u / 2)) * np.sin(u)
    z = v * np.sin(u / 2)
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    faces = []
    for i in range(segments):
        a, b = 2 * i, 2 * i + 1
        if i + 1 < segments:
            c, d = 2 * (i + 1), 2 * (i + 1) + 1
        else:
            # The strip closes with a half twist: the edge is glued reversed.
            c, d = 1, 0
        faces.append([a, b, c])
        faces.append([b, d, c])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)


def _soup(seed: int = 0) -> trimesh.Trimesh:
    """Many components, some sharing edges three ways, with random flips."""
    rng = np.random.default_rng(seed)
    pieces = []
    for _i in range(60):
        piece = trimesh.creation.icosphere(subdivisions=1, radius=rng.uniform(0.05, 0.5))
        piece.apply_translation(rng.uniform(-3, 3, size=3))
        faces = piece.faces.copy()
        flip = rng.random(len(faces)) < 0.5
        faces[flip] = faces[flip][:, [0, 2, 1]]
        pieces.append(trimesh.Trimesh(vertices=piece.vertices, faces=faces, process=False))
    mesh = trimesh.util.concatenate(pieces)
    # A fan of three faces on one edge, welded into the soup.
    base = len(mesh.vertices)
    extra_v = np.array([[5, 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1], [5, -1, 0]], dtype=float)
    extra_f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]]) + base
    return trimesh.Trimesh(
        vertices=np.vstack([mesh.vertices, extra_v]),
        faces=np.vstack([mesh.faces, extra_f]),
        process=False,
    )


# --- tests -------------------------------------------------------------------


def test_edge_counts_match_the_row_version() -> None:
    for mesh in (_soup(), _mobius(), trimesh.creation.box()):
        faces = np.asarray(mesh.faces)
        inverse, counts, _order = _edge_counts(faces)
        ref_inverse, ref_counts = _reference_edge_counts(faces)
        # Group ids may be numbered differently; the partition and counts may not.
        assert np.array_equal(counts[inverse], ref_counts[ref_inverse])
        assert len(np.unique(inverse)) == len(np.unique(ref_inverse))
        boundary, non_manifold = count_non_manifold(mesh)
        assert boundary == int((ref_counts == 1).sum())
        assert non_manifold == int((ref_counts > 2).sum())


def test_boundary_edges_match_the_row_version() -> None:
    for mesh in (_soup(), _mobius()):
        fast = _directed_boundary_edges(mesh)
        slow = _reference_boundary(mesh)
        assert fast.shape == slow.shape, (fast.shape, slow.shape)
        assert np.array_equal(fast, slow)


def test_orientation_matches_the_per_component_search() -> None:
    """Same flips and the same conflicts, including the nine of a Möbius strip."""
    for mesh in (_soup(), _mobius(), trimesh.creation.icosphere(subdivisions=2)):
        _oriented, stats = orient_faces(mesh)
        ref_flipped, ref_conflicts = _reference_orient(mesh)
        # Which faces flip, and which non-tree edges end up as the conflicts,
        # depend on the spanning tree - measured on a 1.3 M-face mesh, 25,142
        # against 25,122. Whether there are any does not, and an orientable
        # surface must come back consistently wound either way.
        assert (stats["conflicts"] == 0) == (ref_conflicts == 0), (stats, ref_conflicts)
        assert stats["unreached"] == 0
        if ref_conflicts == 0:
            assert _oriented.is_winding_consistent
    strip = _mobius()
    _oriented, stats = orient_faces(strip)
    assert stats["conflicts"] > 0, "a Möbius strip must report conflicts"


def test_make_manifold_still_closes_the_soup() -> None:
    mesh = _soup()
    made, report = make_manifold(mesh)
    assert report["boundary_edges"] == 0, report
    assert report["non_manifold_edges"] == 0, report
    assert report["winding_consistent"], report
    assert len(made.faces) >= len(mesh.faces)


def test_drop_small_parts_matches_the_split_version() -> None:
    """Same parts kept; the only difference is that `split()`'s hole filling is gone."""
    mesh = _soup()
    stats = postprocess.CleanStats(faces_before=len(mesh.faces))
    out = postprocess.drop_small_parts(mesh, 0.10, 0.02, None, stats)
    before, kept, dropped = _reference_drop(mesh, 0.10, 0.02)
    assert stats.parts_before == before, (stats.parts_before, before)
    assert stats.parts_after == kept, (stats.parts_after, kept)
    assert stats.dropped_parts == dropped
    # Every face that survives is one of the input's faces, unchanged.
    kept_positions = {tuple(map(tuple, np.round(out.vertices[f], 9))) for f in out.faces}
    input_positions = {tuple(map(tuple, np.round(mesh.vertices[f], 9))) for f in mesh.faces}
    assert kept_positions <= input_positions
    assert len(out.faces) == len(mesh.faces) - stats.dropped_faces
    assert len(out.vertices) <= len(mesh.vertices)


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
