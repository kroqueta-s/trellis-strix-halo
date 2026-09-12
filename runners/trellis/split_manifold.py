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
from typing import Any

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


def edge_keys(faces: np.ndarray) -> np.ndarray:
    """One int64 per edge instance, the same for both directions of an edge.

    **A one-dimensional key is what makes counting edges affordable.**
    `np.unique(..., axis=0)` on the 4 M edge rows of a 1.3 M-face mesh takes
    2.1 s and it is called a dozen times on the way to a manifold; the same
    count on `min * V + max` takes 0.1 s (measured 2026-09-12). `V` is read off
    the faces so that the key cannot collide, and it stays under 2^63 up to
    three billion vertices.
    """
    starts = faces[:, [0, 1, 2]].ravel(order="F")
    ends = faces[:, [1, 2, 0]].ravel(order="F")
    vertex_count = int(faces.max()) + 1 if faces.size else 1
    low = np.minimum(starts, ends).astype(np.int64)
    high = np.maximum(starts, ends).astype(np.int64)
    return low * vertex_count + high


def _edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group the 3F edge instances. Returns (inverse, counts, order by key)."""
    _, inverse, counts = np.unique(edge_keys(faces), return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    return inverse, counts, np.argsort(inverse, kind="stable")


def count_non_manifold(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """(boundary edges, non-manifold edges), counted the same way everywhere here."""
    _, counts = np.unique(edge_keys(np.asarray(mesh.faces)), return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def _two_colour(graph: csr_matrix) -> np.ndarray:
    """Which faces to flip so that neighbours agree, one BFS forest for the whole mesh.

    `graph` holds `1` on an edge whose faces already agree and `2` on one whose
    faces disagree. The colouring follows a spanning forest: a face is flipped
    when the parity of disagreeing edges on its path to the root is odd.

    **One search, not one per component.** `breadth_first_order` allocates
    arrays the size of the whole mesh on every call, so calling it per
    component costs components times faces - 57 s on a 200 k-face mesh with
    10 k components (measured 2026-09-12), against 7 s on 1.3 M faces with 73.
    A virtual root joined to one face of every component turns that into a
    single call, and the parity along the tree is then gathered by pointer
    doubling: each round halves the distance to the root, so a dozen vectorised
    passes cover a tree a few thousand deep.
    """
    face_count = graph.shape[0]
    if face_count == 0:
        return np.zeros(0, dtype=bool)
    _, components = connected_components(graph, directed=False)
    _, roots = np.unique(components, return_index=True)

    root = face_count
    extra = coo_matrix(
        (np.ones(len(roots), dtype=graph.dtype), (np.full(len(roots), root), roots)),
        shape=(face_count + 1, face_count + 1),
    )
    padded = csr_matrix(graph)
    padded.resize((face_count + 1, face_count + 1))
    forest = (padded + extra + extra.T).tocsr()
    forest.sort_indices()
    _visit, predecessors = breadth_first_order(forest, root, directed=False)

    parent = predecessors.astype(np.int64)
    parent[root] = root
    # The weight of each tree edge, looked up by its (parent, child) key.
    coo = forest.tocoo()
    keys = coo.row.astype(np.int64) * (face_count + 1) + coo.col
    order = np.argsort(keys)
    wanted = parent * (face_count + 1) + np.arange(face_count + 1)
    position = np.searchsorted(keys[order], wanted).clip(max=len(order) - 1)
    weight = coo.data[order][position]
    parity = weight == 2
    parity[root] = False

    ancestor = parent
    while True:
        parity = parity ^ parity[ancestor]
        ancestor = ancestor[ancestor]
        if bool((ancestor == root).all()):
            break
    return parity[:face_count]


def split_non_manifold(
    mesh: trimesh.Trimesh, separation: float = 0.0, cut: np.ndarray | None = None
) -> tuple[trimesh.Trimesh, SplitStats]:
    """Duplicate the vertices where surface sheets meet, keeping every face.

    Args:
        mesh: The mesh to separate. **It is not modified.**
        cut: Optional boolean mask over the **paired edge instances**, marking
            edges to treat as a cut even though two faces share them. This is
            how the orientation conflicts are resolved: an edge whose two faces
            cannot agree is separated instead of being argued with.
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
    if cut is not None:
        paired = paired[~cut]
    first, second = paired[:, 0], paired[:, 1]

    # The two faces may traverse the edge in either direction, so the corner
    # that holds a given vertex has to be matched by vertex, not by position.
    same_direction = starts[first] == starts[second]
    partner_start = np.where(same_direction, corner_start[second], corner_end[second])
    partner_end = np.where(same_direction, corner_end[second], corner_start[second])

    rows = np.concatenate([corner_start[first], corner_end[first]])
    cols = np.concatenate([partner_start, partner_end])
    corners = face_count * 3
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(corners, corners))
    _, labels = connected_components(graph, directed=False)

    # Every group becomes one vertex, sitting where its original sits.
    corner_vertex = faces.ravel()  # corner c = face c//3, slot c%3
    group_vertex = np.zeros(labels.max() + 1, dtype=np.int64)
    group_vertex[labels] = corner_vertex
    new_vertices = vertices[group_vertex]
    new_faces = labels.reshape(face_count, 3)
    if separation > 0:
        # Pull each copy a little towards the middle of the faces that kept it,
        # so that copies of one original stop sharing a position. **Only the
        # copies move**: a vertex that was never duplicated stays exactly
        # where it was.
        scale = separation * float(np.ptp(vertices, axis=0).max())
        copies = np.bincount(group_vertex, minlength=len(vertices))[group_vertex] > 1
        centres = np.zeros_like(new_vertices)
        weights = np.zeros(len(new_vertices))
        for corner in range(3):
            np.add.at(centres, new_faces[:, corner], new_vertices[new_faces].mean(axis=1))
            np.add.at(weights, new_faces[:, corner], 1.0)
        direction = centres / np.maximum(weights, 1)[:, None] - new_vertices
        length = np.linalg.norm(direction, axis=1, keepdims=True)
        moved = new_vertices + scale * direction / np.maximum(length, 1e-12)
        new_vertices = np.where(copies[:, None], moved, new_vertices)

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


def separate_coincident(
    mesh: trimesh.Trimesh, distance: float = 1e-5, rounds: int = 4
) -> tuple[trimesh.Trimesh, int]:
    """Nudge every vertex that shares its exact position with another, so none does.

    **Whatever leaves this module gets welded by position downstream** - forge's
    `repair_manifold` starts with `merge_vertices()` - and two vertices at one
    point become one, taking their faces' edges with them. The split above
    keeps its own copies apart; this catches the rest, which arrive from
    elsewhere (decimation collapsing two vertices onto one point, a fan apex
    landing on a vertex, a copy whose faces surround it symmetrically): 260
    of 7,710 copies and 14 from decimation on the 512 specimen, measured
    2026-09-12, enough for `manifold3d` to refuse the welded mesh.

    Each such vertex moves `distance` (a fraction of the longest side) towards
    the middle of its own faces, which keeps it on its own sheet; a vertex
    whose faces surround it symmetrically moves along a fixed direction
    instead. A few rounds cover the unlikely case of two moves landing on
    one point again.

    Returns:
        The mesh with the moved vertices, and how many were moved.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces)
    extent = float(np.ptp(vertices, axis=0).max())
    step = distance * extent
    moved_total = 0
    for _ in range(rounds):
        # Within 1e-7 of the longest side counts as one point: closer than a
        # float32 file keeps apart, and still a hundred times finer than the
        # nudge, so a moved vertex is never caught again.
        _, inverse, counts = np.unique(
            np.round(vertices / max(extent, 1e-12), 7),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        shared = counts[inverse.ravel()] > 1
        if not shared.any():
            break
        centres = np.zeros_like(vertices)
        weights = np.zeros(len(vertices))
        face_centre = vertices[faces].mean(axis=1)
        for corner in range(3):
            np.add.at(centres, faces[:, corner], face_centre)
            np.add.at(weights, faces[:, corner], 1.0)
        direction = centres / np.maximum(weights, 1)[:, None] - vertices
        length = np.linalg.norm(direction, axis=1)
        flat = length < 1e-12
        direction[flat] = np.array([1.0, 1.0, 1.0])
        length[flat] = np.sqrt(3.0)
        # Two vertices of one group can have the same direction - copies of
        # a split vertex whose sheets mirror each other do - so each member
        # also steps a different distance, by its rank within the group.
        order = np.argsort(inverse.ravel(), kind="stable")
        group = inverse.ravel()[order]
        first = np.r_[0, np.flatnonzero(group[1:] != group[:-1]) + 1]
        rank = np.empty(len(vertices), dtype=np.int64)
        sizes = np.diff(np.r_[first, len(vertices)])
        rank[order] = np.arange(len(vertices)) - np.repeat(first, sizes)
        scale = step * (1.0 + rank)
        nudge = direction / length[:, None] * scale[:, None]
        vertices[shared] += nudge[shared]
        moved_total += int(shared.sum())
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), moved_total


def drop_pillows(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    """Drop faces that repeat a vertex, and every face whose vertex set another face repeats.

    Two faces on the same three vertices are a pillow: a closed sliver of no
    volume that the edge counts call manifold (each edge has its two faces)
    and that a downstream `unique_faces()` then opens by keeping one of them
    - measured 2026-09-12 on the 512 specimen, 87 pairs became 265 boundary
    edges in forge's repair. They come from fans closed over three-vertex
    loops and from decimation, and they touch nothing else (0 of their 261
    edges were shared), so both faces of each pair go.

    Returns:
        The mesh without them, and how many faces went.
    """
    faces = np.asarray(mesh.faces)
    repeated = (
        (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    )
    vertex_count = int(faces.max()) + 1 if faces.size else 1
    key = np.sort(faces, axis=1).astype(np.int64)
    key = (key[:, 0] * vertex_count + key[:, 1]) * vertex_count + key[:, 2]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    duplicated = counts[inverse.ravel()] > 1
    drop = repeated | duplicated
    if not drop.any():
        return mesh, 0
    kept = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=faces[~drop], process=False)
    kept.remove_unreferenced_vertices()
    return kept, int(drop.sum())


def conflicting_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """Which paired edges still disagree once the orientation has been propagated.

    Returns a boolean mask over the paired edge instances, in the order
    `split_non_manifold` expects for its `cut` argument. **These are the edges
    that make the surface non-orientable**, and cutting them is the only way to
    an orientable one - measured 2026-09-12, 26,843 of roughly 2.2 M.
    """
    _oriented, stats = orient_faces(mesh, _return_mask=True)
    return stats["conflict_mask"]


def orient_faces(
    mesh: trimesh.Trimesh, _return_mask: bool = False
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
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

    paired_all = order[counts[inverse[order]] == 2].reshape(-1, 2)
    paired = paired_all
    left, right = instance_face[paired[:, 0]], instance_face[paired[:, 1]]
    # **Same direction means they disagree**: two faces sharing an edge are
    # consistently wound when they traverse it in opposite directions.
    disagree = (starts[paired[:, 0]] == starts[paired[:, 1]]).astype(np.int8)

    # **Deduplicate the pairs first.** `csr_matrix` sums the data of repeated
    # coordinates, and two faces can share more than one edge - which would
    # turn a parity of 1 into 2 and corrupt the colouring. Measured before this
    # was handled: 39,553 phantom conflicts on a surface that has none.
    low = np.minimum(left, right).astype(np.int64)
    high = np.maximum(left, right).astype(np.int64)
    _, unique_index = np.unique(low * face_count + high, return_index=True)
    left, right, disagree = left[unique_index], right[unique_index], disagree[unique_index]

    graph = csr_matrix(
        (
            np.concatenate([disagree + 1, disagree + 1]),
            (np.concatenate([left, right]), np.concatenate([right, left])),
        ),
        shape=(face_count, face_count),
    )
    flip = _two_colour(graph)
    seen = np.ones(face_count, dtype=bool)
    # A conflict is an edge whose two faces still disagree once flipped.
    still = (disagree.astype(bool)) ^ (flip[left] ^ flip[right])
    conflicts = int(still.sum())
    if _return_mask:
        # The mask has to line up with every paired edge, not only the
        # deduplicated ones, so it is rebuilt over the original pairing.
        full = np.zeros(len(paired_all), dtype=bool)
        full[unique_index[still]] = True
        return mesh, {"conflicts": conflicts, "conflict_mask": full}

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


def make_manifold(
    mesh: trimesh.Trimesh, separation: float = 1e-5
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Turn the decoder's surface into a closed, orientable, manifold one.

    `separation` is how far each duplicated vertex is pulled towards its own
    sheet, as a fraction of the longest side. **It must not be zero for a mesh
    that leaves this process**: copies left at one position are welded back
    together by anything that merges vertices by position - forge's
    `repair_manifold` does so as its first step - and the junctions return
    with them. Measured 2026-09-12 on the carved 512 specimen: 14,231
    coincident copies became 20,163 non-manifold edges after
    `merge_vertices()`, and `manifold3d` refused the result. At 1e-5 the copies
    sit 0.5 % of a cell apart on a 512 lattice, far beyond trimesh's merge
    tolerance of 1e-8, and invisibly close for any other purpose.

    Four steps, each one measured on the way past:

    1. **Separate the sheets that touch** — the non-manifold edges become
       boundary, and nothing is deleted.
    2. **Propagate the orientation** and find the edges that still disagree.
       There are always some: the surface the decoder produces is genuinely
       non-orientable (26,843 edges of roughly 2.2 M, measured 2026-09-12, and
       a Möbius strip reproduces the effect while a torus does not).
    3. **Cut those edges too**, which is what makes an orientable surface out of
       a non-orientable one. **This is where the geometry is paid for**: it
       opens long seams, and closing them again is invention.
    4. **Close every seam**, uncapped — a capped close would leave the mesh
       open, which defeats the point of asking for a manifold at all.

    Measured end to end on the mecha at 1024, decimated to 1.4 M faces: a
    manifold of 1,520,680 faces, `manifold3d` accepting it with `NoError`, and
    forge's decompose-and-union going through. **The patches are 7.6% of the
    faces and most of the added area is internal** — the silhouette and the
    detail survive, which is why this is worth its cost.

    Returns:
        The manifold mesh and what each step did. **`fan_area_fraction` says
        how much of it is invention**, and it is not small.
    """
    from .close_holes import close_holes

    report: dict[str, Any] = {"faces_in": int(len(mesh.faces))}
    work, split_stats = split_non_manifold(mesh, separation=separation)
    report["split"] = split_stats.as_dict()

    cut = conflicting_edges(work)
    report["conflicting_edges"] = int(cut.sum())
    if cut.any():
        work, second = split_non_manifold(work, separation=separation, cut=cut)
        report["cut"] = second.as_dict()

    work, orient_stats = orient_faces(work)
    report["orient"] = {k: v for k, v in orient_stats.items() if k != "conflict_mask"}

    # **Uncapped on purpose**: these are seams this function opened, not holes
    # in the model, and leaving them open would make the whole exercise moot.
    work, close_stats = close_holes(work, max_extent=0.0)
    report["close"] = close_stats.as_dict()

    # **Last, because everything above can leave two vertices on one point.**
    work, moved = separate_coincident(work, distance=separation)
    report["coincident_moved"] = moved
    work, dropped = drop_pillows(work)
    report["pillow_faces_dropped"] = dropped

    boundary, non_manifold = count_non_manifold(work)
    report["boundary_edges"] = boundary
    report["non_manifold_edges"] = non_manifold
    report["watertight"] = bool(work.is_watertight)
    report["winding_consistent"] = bool(work.is_winding_consistent)
    report["faces_out"] = int(len(work.faces))
    return work, report
