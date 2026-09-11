# SPDX-License-Identifier: MIT
"""Separate the sheets that touch, by duplicating vertices rather than deleting faces.

**This is what stands between TRELLIS.2's meshes and a watertight manifold.**
Measured 2026-09-12 on the construction-mecha specimen at 1024, decimated to 1.4
M faces: **16,933 edges are shared by more than two faces**, `manifold3d`
refuses the mesh outright, and forge's `repair_manifold` comes back with
"repairing did not make this watertight".

**Deleting the offending faces does not work.** Cutting them out and closing the
resulting holes was measured over three rounds and diverged: 16,933 non-manifold
edges became 32,377, then 74,413, then 206,207, because each fan patch joins
loops that share vertices.

The fix is the standard one, and it removes nothing. Two faces belong to the
same surface sheet only if they meet across an edge that **exactly two** faces
use. Group each vertex's corners by that relation, and give every group its own
copy of the vertex. An edge used by four faces then becomes two edges between
two pairs of copies - **edge-manifold and vertex-manifold, with every face kept
and every position unchanged**. Where sheets came apart, a boundary appears, and
closing those is `close_holes`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components


@dataclass
class SplitStats:
    """What separating did."""

    vertices_before: int
    vertices_after: int
    non_manifold_edges_before: int
    non_manifold_edges_after: int
    boundary_edges_before: int
    boundary_edges_after: int

    def as_dict(self) -> dict[str, int]:
        return {
            "vertices_before": self.vertices_before,
            "vertices_after": self.vertices_after,
            "non_manifold_edges_before": self.non_manifold_edges_before,
            "non_manifold_edges_after": self.non_manifold_edges_after,
            "boundary_edges_before": self.boundary_edges_before,
            "boundary_edges_after": self.boundary_edges_after,
        }


def _edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group the 3F edge instances. Returns (inverse, counts, order by key)."""
    starts = faces[:, [0, 1, 2]].ravel(order="F")
    ends = faces[:, [1, 2, 0]].ravel(order="F")
    keys = np.sort(np.stack([starts, ends], axis=1), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return inverse.ravel(), counts, np.argsort(inverse.ravel(), kind="stable")


def count_non_manifold(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """(boundary edges, non-manifold edges), counted the same way everywhere here."""
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def split_non_manifold(
    mesh: trimesh.Trimesh, separation: float = 0.0
) -> tuple[trimesh.Trimesh, SplitStats]:
    """Duplicate the vertices where surface sheets meet, keeping every face.

    Args:
        mesh: The mesh to separate. **It is not modified.**
        separation: How far to pull each copy towards its own sheet, as a
            fraction of the longest side. **0 leaves every copy exactly where
            the original was**, which keeps the geometry untouched but leaves
            the copies coincident - and a library that welds by position, as
            `manifold3d` does, then puts the junction straight back.

    Returns:
        The separated mesh and what it took.
    """
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)
    face_count = len(faces)
    boundary_before, non_manifold_before = count_non_manifold(mesh)

    inverse, counts, order = _edge_counts(faces)
    instance_face = np.tile(np.arange(face_count), 3)
    instance_slot = np.repeat(np.arange(3), face_count)
    # The corner holding the edge's first vertex, and the one holding its second.
    corner_start = instance_face * 3 + instance_slot
    corner_end = instance_face * 3 + (instance_slot + 1) % 3
    starts = faces[:, [0, 1, 2]].ravel(order="F")

    # **Only edges used by exactly two faces join sheets.** Their instances sit
    # next to each other once sorted by key, so the pairs come out of a reshape.
    per_key_count = counts[inverse[order]]
    paired = order[per_key_count == 2].reshape(-1, 2)
    first, second = paired[:, 0], paired[:, 1]

    # The two faces may traverse the edge in either direction, so the corner
    # that holds a given vertex has to be matched by vertex, not by position.
    same_direction = starts[first] == starts[second]
    partner_start = np.where(same_direction, corner_start[second], corner_end[second])
    partner_end = np.where(same_direction, corner_end[second], corner_start[second])

    rows = np.concatenate([corner_start[first], corner_end[first]])
    cols = np.concatenate([partner_start, partner_end])
    corners = face_count * 3
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(corners, corners)
    )
    _, labels = connected_components(graph, directed=False)

    # Every group becomes one vertex, sitting where its original sits.
    corner_vertex = faces.ravel()  # corner c = face c//3, slot c%3
    group_vertex = np.zeros(labels.max() + 1, dtype=np.int64)
    group_vertex[labels] = corner_vertex
    new_vertices = vertices[group_vertex]
    new_faces = labels.reshape(face_count, 3)
    if separation > 0:
        # Pull each copy a little towards the middle of the faces that kept it,
        # so that copies of one original stop sharing a position.
        scale = separation * float(np.ptp(vertices, axis=0).max())
        centres = np.zeros_like(new_vertices)
        weights = np.zeros(len(new_vertices))
        for corner in range(3):
            np.add.at(centres, new_faces[:, corner], new_vertices[new_faces].mean(axis=1))
            np.add.at(weights, new_faces[:, corner], 1.0)
        direction = centres / np.maximum(weights, 1)[:, None] - new_vertices
        length = np.linalg.norm(direction, axis=1, keepdims=True)
        new_vertices = new_vertices + scale * direction / np.maximum(length, 1e-12)

    split = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    boundary_after, non_manifold_after = count_non_manifold(split)
    return split, SplitStats(
        vertices_before=len(vertices),
        vertices_after=len(split.vertices),
        non_manifold_edges_before=non_manifold_before,
        non_manifold_edges_after=non_manifold_after,
        boundary_edges_before=boundary_before,
        boundary_edges_after=boundary_after,
    )


def orient_faces(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, int]]:
    """Wind every face the same way round, and point them outwards.

    **Only meaningful once the mesh is edge-manifold**, which is what
    `split_non_manifold` produces: then every edge has exactly two faces, and
    agreeing on an orientation is a two-colouring of the face graph - flip a
    face when it traverses a shared edge in the same direction as its
    neighbour. Where that two-colouring fails the surface is not orientable,
    and the conflicts are counted rather than hidden.

    `trimesh.repair.fix_winding` is the obvious alternative and **it does not
    finish the job here**: measured 2026-09-12 on this mesh, 88 s and the
    winding still inconsistent afterwards.

    The sign of the volume decides which way is out, so a closed surface that
    came out inside-in is flipped whole at the end.
    """
    faces = np.asarray(mesh.faces).copy()
    face_count = len(faces)
    inverse, counts, order = _edge_counts(faces)
    instance_face = np.tile(np.arange(face_count), 3)
    starts = faces[:, [0, 1, 2]].ravel(order="F")

    paired = order[counts[inverse[order]] == 2].reshape(-1, 2)
    left, right = instance_face[paired[:, 0]], instance_face[paired[:, 1]]
    # **Same direction means they disagree**: two faces sharing an edge are
    # consistently wound when they traverse it in opposite directions.
    disagree = (starts[paired[:, 0]] == starts[paired[:, 1]]).astype(np.int8)

    # **Deduplicate the pairs first.** `csr_matrix` sums the data of repeated
    # coordinates, and two faces can share more than one edge - which would
    # turn a parity of 1 into 2 and corrupt the colouring. Measured before this
    # was handled: 39,553 phantom conflicts on a surface that has none.
    low = np.minimum(left, right)
    high = np.maximum(left, right)
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
    seen = np.zeros(face_count, dtype=bool)
    conflicts = 0
    _, components = connected_components(graph, directed=False)
    for component in np.unique(components):
        root = int(np.flatnonzero(components == component)[0])
        visit, predecessors = breadth_first_order(graph, root, directed=False)
        seen[visit] = True
        for node in visit[1:]:
            parent = predecessors[node]
            span = slice(indptr[parent], indptr[parent + 1])
            position = indptr[parent] + int(np.searchsorted(indices[span], node))
            flip[node] = flip[parent] ^ bool(data[position] - 1)
    # A conflict is an edge whose two faces still disagree once flipped.
    still = (disagree.astype(bool)) ^ (flip[left] ^ flip[right])
    conflicts = int(still.sum())

    faces[flip] = faces[flip][:, [0, 2, 1]]
    oriented = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=faces, process=False)
    flipped_whole = False
    if oriented.volume < 0:
        # **Closed and inside-in**: one flip of everything, not a search.
        faces = faces[:, [0, 2, 1]]
        oriented = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=faces, process=False)
        flipped_whole = True

    return oriented, {
        "faces_flipped": int(flip.sum()),
        "conflicts": conflicts,
        "flipped_whole": int(flipped_whole),
        "unreached": int((~seen).sum()),
    }
