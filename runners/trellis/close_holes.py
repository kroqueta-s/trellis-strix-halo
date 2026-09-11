# SPDX-License-Identifier: MIT
"""Close every boundary loop in a mesh, however many there are.

**TRELLIS.2's geometry does not arrive closed.** The flags the decoder predicts
say which grid edges the surface crosses, and **about 5% of the primal faces
carry an odd number of crossings** (4.99% at 512, 5.02% at 1024, measured
2026-09-12 as the share of quad edges with an odd number of quads on them). A
dual-grid surface closes only where that count is even, and no inside/outside
labelling can explain a field that is 5% frustrated - recovering one was tried
and left 6.8% of its constraints violated, which is worse than the 5% it set
out to fix. Upstream reaches a closed mesh the way anything else would: it
fills the holes afterwards, with `CuMesh.fill_holes`, which has no
Windows + ROCm build.

What the holes actually look like decides how to close them. Measured on the
same specimen, decimated to 700k faces: **2,978 loops over 27,305 vertices,
median 3 vertices, 95.2% under 32, none over 1000.** They are thousands of
pinholes, not a few gaping openings - which is why `trimesh.repair.fill_holes`
only reached a third of them (it closes three- and four-edge holes and nothing
else).

So each loop is closed with a fan from its own centroid. That needs no ordering
of the loop, works whatever shape it is, and is one pass over the boundary
edges. **The orientation comes from the face the boundary edge already belongs
to**, so the patch is wound to match its neighbour rather than by guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class CloseStats:
    """What closing did - **including how much of the surface it invented**."""

    loops: int
    boundary_edges_before: int
    boundary_edges_after: int
    faces_added: int
    vertices_added: int
    largest_loop: int
    loops_left_open: int = 0
    fan_area_fraction: float = 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loops": self.loops,
            "boundary_edges_before": self.boundary_edges_before,
            "boundary_edges_after": self.boundary_edges_after,
            "faces_added": self.faces_added,
            "vertices_added": self.vertices_added,
            "largest_loop": self.largest_loop,
            # **Loops too wide to be a hole**, left open rather than covered over.
            "loops_left_open": self.loops_left_open,
            # **How much of the delivered surface is patch, measured by area.**
            # The face count understates it badly: the patches were 7.6% of the
            # faces and **13.4% of the area** (measured 2026-09-12 at 512).
            "fan_area_fraction": round(self.fan_area_fraction, 4),
        }


def _directed_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """The boundary edges, **in the direction the existing face uses them**.

    A fill triangle has to run the other way round to be wound consistently with
    its neighbour, so the direction is the whole point of returning them this
    way rather than sorted.
    """
    faces = mesh.faces
    directed = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    return directed[counts[inverse] == 1]


def _loop_labels(edges: np.ndarray, vertex_count: int) -> np.ndarray:
    """Label each boundary edge with the connected loop it belongs to.

    Union-find over the boundary vertices. **Loops are not assumed to be simple**
    - a figure-of-eight through a pinched vertex stays one label and is closed
    as one fan, which is the conservative choice.
    """
    parent = np.arange(vertex_count)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges.tolist():
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    roots = np.array([find(int(v)) for v in edges[:, 0]])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def close_holes(
    mesh: trimesh.Trimesh, passes: int = 1, max_extent: float = 0.05
) -> tuple[trimesh.Trimesh, CloseStats]:
    """Fill boundary loops with a fan from each loop's centroid, until none are left.

    **One pass does not always finish.** A fan closes a loop that is a proper
    cycle; where the boundary pinches through a shared vertex, the component is
    not a cycle and the fan leaves a shorter boundary behind. Measured on the
    700k specimen: 26,959 boundary edges fall to 3,123 in one pass, and the
    remainder are loops of that kind.

    **Repeating does not help, so the default is one pass.** Measured on the
    same specimen: four passes take 13.2s and finish on 3,224 boundary edges,
    where one pass takes 2.5s and finishes on 3,123. What is left is not a hole
    - it is the boundary that **non-manifold edges** produce, and closing that
    needs the surfaces separated first. Raise `passes` only with a measurement
    that says it earned it.

    **A loop wider than `max_extent` is not a hole and is left open.** A fan
    over a wide loop is a sail, not a repair: measured at 512, closing every
    loop covers **13.4% of the surface area** with patch, and the loops
    responsible span up to 32% of the model. Capping at 0.05 of the longest
    side closes 15,135 of 15,352 loops for **3.3%** of the area, which is where
    the size distribution has its knee (the 99th percentile extent is 0.0658).
    Set it to 0 to close everything, as before.

    Args:
        mesh: The mesh to close. **It is not modified.**
        passes: Give up after this many rounds rather than looping forever.
        max_extent: Widest loop to close, as a fraction of the longest side.
            0 closes every loop.

    Returns:
        The closed mesh and what it took. **`fan_area_fraction` says how much
        of the result is invention.**
    """
    total_faces = 0
    total_vertices = 0
    total_loops = 0
    largest = 0
    first_boundary = -1
    work = mesh
    left_open = 0
    fan_area = 0.0
    for _ in range(passes):
        work, report = _close_once(work, max_extent)
        left_open = report.loops_left_open
        fan_area += report.fan_area_fraction
        if first_boundary < 0:
            first_boundary = report.boundary_edges_before
        total_faces += report.faces_added
        total_vertices += report.vertices_added
        total_loops += report.loops
        largest = max(largest, report.largest_loop)
        if report.boundary_edges_after == 0:
            break
    return work, CloseStats(
        loops=total_loops,
        boundary_edges_before=max(first_boundary, 0),
        boundary_edges_after=len(_directed_boundary_edges(work)),
        faces_added=total_faces,
        vertices_added=total_vertices,
        largest_loop=largest,
        loops_left_open=left_open,
        fan_area_fraction=fan_area,
    )


def _close_once(
    mesh: trimesh.Trimesh, max_extent: float = 0.0
) -> tuple[trimesh.Trimesh, CloseStats]:
    """One fan pass over every boundary loop narrow enough to be a hole."""
    edges = _directed_boundary_edges(mesh)
    before = len(edges)
    if before == 0:
        return mesh, CloseStats(0, 0, 0, 0, 0, 0)

    labels = _loop_labels(edges, len(mesh.vertices))
    loop_count = int(labels.max()) + 1

    # One new vertex per loop, at the mean of the vertices on it.
    sums = np.zeros((loop_count, 3))
    weights = np.zeros(loop_count)
    np.add.at(sums, labels, mesh.vertices[edges[:, 0]])
    np.add.at(weights, labels, 1.0)
    centroids = sums / weights[:, None]

    left_open = 0
    if max_extent > 0:
        # A loop's extent is the box its vertices span; anything wider than the
        # threshold is an opening, not a hole.
        lows = np.full((loop_count, 3), np.inf)
        highs = np.full((loop_count, 3), -np.inf)
        np.minimum.at(lows, labels, mesh.vertices[edges[:, 0]])
        np.maximum.at(highs, labels, mesh.vertices[edges[:, 0]])
        extent = np.linalg.norm(highs - lows, axis=1)
        narrow = extent <= max_extent * float(mesh.extents.max())
        left_open = int((~narrow).sum())
        keep_edge = narrow[labels]
        if not keep_edge.any():
            return mesh, CloseStats(0, before, before, 0, 0, 0, left_open, 0.0)
        edges = edges[keep_edge]
        labels = labels[keep_edge]

    vertices = np.concatenate([mesh.vertices, centroids], axis=0)
    apex = len(mesh.vertices) + labels
    # **How much surface this invents**, measured before it is added.
    arms_a = mesh.vertices[edges[:, 0]] - centroids[labels]
    arms_b = mesh.vertices[edges[:, 1]] - centroids[labels]
    fan_area = float(0.5 * np.linalg.norm(np.cross(arms_a, arms_b), axis=1).sum())
    # The existing face uses (u, v); the patch uses (v, u, apex).
    patches = np.stack([edges[:, 1], edges[:, 0], apex], axis=1)
    faces = np.concatenate([mesh.faces, patches], axis=0)

    closed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    after = len(_directed_boundary_edges(closed))
    sizes = np.bincount(labels, minlength=loop_count)
    area = float(mesh.area)
    return closed, CloseStats(
        loops=int((sizes > 0).sum()),
        boundary_edges_before=before,
        boundary_edges_after=after,
        faces_added=len(patches),
        vertices_added=int((sizes > 0).sum()),
        largest_loop=int(sizes.max()),
        loops_left_open=left_open,
        fan_area_fraction=fan_area / area if area > 0 else 0.0,
    )
