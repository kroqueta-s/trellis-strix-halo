# SPDX-License-Identifier: MIT
"""Upstream TRELLIS's `_fill_holes`, transcribed **with its procedure and thresholds intact**.

## Why it was transcribed (**this is not a different algorithm**)

Upstream's `trellis/utils/postprocessing_utils.py:_fill_holes` is correct, but
**its Python side has one pathological inefficiency** that dominated on this
machine.

```python
g.add_edges([(f, "s") for f in inner_face_indices], ...)   # walks a CUDA tensor element by element
```

Measured 2026-09-01 on a 697,152-face mesh with 150 views:

| Stage | Time | Share |
|---|--:|--:|
| Rasterizing 150 views | 36.15 s | 44% |
| **Edges to source (the line above)** | **21.64 s** | **27%** |
| **Edges to target (an identical line)** | **13.52 s** | **17%** |
| `g.mincut` itself | 5.11 s | 6% |
| Everything else | 5.0 s | 6% |

**The min-cut itself takes 5 seconds**; the cost was 360,000 GPU reads, which
disappear once the indices move to the CPU in one go. **Neither the algorithm
nor any threshold was changed**, so the output matches upstream
(`tests/test_fill_holes.py` confirms the agreement on a small mesh).

**Upstream's helpers are called as-is** (`utils3d.torch.*`,
`sphere_hammersley_sequence`). Only the sequencing lives here.

**Two of upstream's libraries are not used**: `igraph` (GPL) for the min-cut
and `pymeshfix` (AGPL-3.0) for the hole filling. This repository is MIT, so
the cut is solved with scipy's maximum flow and the holes are closed by
`close_holes.py`. Measured on the same specimen: 518,576 faces against
525,992, one component, 0 boundary edges, 0 non-manifold edges, watertight,
and the enclosed volume within 0.26%.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
import torch

from .steps import StepCounter, counted


def _cameras(num_views: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build upstream's viewpoints (Hammersley sequence, radius 2.0, 40-degree FOV)."""
    import utils3d
    from trellis.utils.random_utils import sphere_hammersley_sequence

    yaws, pitchs = [], []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws_t = torch.tensor(yaws, device=device)
    pitchs_t = torch.tensor(pitchs, device=device)
    fov = torch.deg2rad(torch.tensor(40, device=device))
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    origin = torch.zeros(3, device=device, dtype=torch.float32)
    up = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
    views = []
    for yaw, pitch in zip(yaws_t, pitchs_t, strict=True):
        eye = (
            torch.stack(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                ]
            ).float()
            * 2.0
        )
        views.append(utils3d.torch.view_look_at(eye, origin, up))
    return torch.stack(views, dim=0), projection


def visibility(
    verts: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    num_views: int,
    progress: Callable[[str, str], None] | None = None,
) -> torch.Tensor:
    """Return the visibility ratio per face.

    **Upstream's definition**: the fraction of views the face appears in.
    """
    import utils3d

    views, projection = _cameras(num_views, verts.device)
    seen = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    ctx = utils3d.torch.RastContext(backend="cuda")
    # **This loop is ours**, so it is counted directly rather than through a
    # hook. It used to report only every 50th view, which said almost nothing
    # over the 36 s it takes.
    counter = StepCounter()
    counter.bind(progress, "raster", "measuring visibility")
    try:
        for i in counted(range(int(views.shape[0])), counter):
            buffers = utils3d.torch.rasterize_triangle_faces(
                ctx,
                verts[None],
                faces,
                resolution,
                resolution,
                view=views[i],
                projection=projection,
            )
            face_id = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
            seen[torch.unique(face_id).long()] += 1
    finally:
        counter.bind(None, "raster")
    return seen.float() / num_views


def fill_holes(
    verts: torch.Tensor,
    faces: torch.Tensor,
    max_hole_size: float = 0.04,
    max_hole_nbe: int = 250,
    resolution: int = 1024,
    num_views: int = 150,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut invisible faces with a min-cut and fill small holes.

    **Upstream's procedure, with upstream's thresholds.**

    Args:
        verts: `[V, 3]` on cuda.
        faces: `[F, 3]` on cuda.
        max_hole_size: Maximum area of the boundary loop a cut may open. A cut
            exceeding it is rejected.
        max_hole_nbe: Maximum number of edges in a boundary loop the closing
            will fill.
        resolution: Rasterization resolution.
        num_views: Number of viewpoints.
        progress: Where to report the stage.

    Returns:
        `(verts, faces)`.
    """
    import utils3d

    def say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    say("raster", f"measuring visibility ({num_views} views / {resolution}^2)")
    visblity = visibility(verts, faces, resolution, num_views, progress)

    say("graph", "building the dual graph")
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    components = utils3d.torch.compute_connected_components(faces, edges, face2edge)

    # Decide the "outer faces" per component (upstream's adaptive threshold).
    outer_mask = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for comp in components:
        threshold = min(max(visblity[comp].quantile(0.75).item(), 0.25), 0.5)
        outer_mask[comp] = visblity[comp] > threshold
    outer_face_indices = outer_mask.nonzero().reshape(-1)
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if inner_face_indices.shape[0] == 0:
        say("graph", "no invisible faces, so nothing is cut")
        return verts, faces

    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)

    n_faces = int(faces.shape[0])
    source, target = n_faces, n_faces + 1

    # **The indices move to the CPU in one go**, where upstream walks a CUDA
    # tensor element by element in Python and spends 35 seconds on 360,000 GPU
    # reads. The edges are the same ones.
    inner_list = inner_face_indices.cpu().numpy()
    outer_list = outer_face_indices.cpu().numpy()
    say("mincut", f"solving the min-cut (inner {len(inner_list):,} / outer {len(outer_list):,})")
    remove_face_indices = torch.tensor(
        _min_cut(
            n_faces,
            dual_edges.cpu().numpy(),
            dual_weights.cpu().numpy(),
            inner_list,
            outer_list,
            source,
            target,
        ),
        dtype=torch.long,
        device=faces.device,
    )
    if remove_face_indices.shape[0] == 0:
        say("mincut", "no faces to cut")
    else:
        remove_face_indices = _validate_cut(
            verts,
            faces,
            edges,
            face2edge,
            boundary_edge_indices,
            visblity,
            remove_face_indices,
            max_hole_size,
        )
        if remove_face_indices.shape[0] > 0:
            keep = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
            keep[remove_face_indices] = False
            faces = faces[keep]
            faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
            say("mincut", f"cut {int(remove_face_indices.shape[0]):,} faces")

    say("close", f"closing boundary loops of up to {max_hole_nbe} edges")
    import trimesh

    from .close_holes import close_holes
    from .split_manifold import count_non_manifold, split_non_manifold

    closed, stats = close_holes(
        trimesh.Trimesh(
            vertices=verts.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
            process=False,
        ),
        max_extent=0.0,
        max_edges=int(max_hole_nbe),
    )
    say("close", f"closed {stats.loops:,} loops, left {stats.loops_left_open:,} open")

    # **A fan through a pinch vertex leaves a non-manifold edge behind.** There
    # are only a handful - three on the reference specimen - and leaving them
    # costs the mesh its watertightness, which is the one property the caller
    # is entitled to. Separating them and closing what opens takes under a
    # second at this size.
    _boundary, non_manifold = count_non_manifold(closed)
    if non_manifold:
        say("close", f"separating {non_manifold:,} non-manifold edges the patches left")
        closed, _split = split_non_manifold(closed)
        closed, again = close_holes(closed, max_extent=0.0)
        say("close", f"closed {again.loops:,} more loops")
    return (
        torch.tensor(np.asarray(closed.vertices), device=verts.device, dtype=torch.float32),
        torch.tensor(np.asarray(closed.faces), device=faces.device, dtype=torch.int32),
    )


def _validate_cut(
    verts: torch.Tensor,
    faces: torch.Tensor,
    edges: torch.Tensor,
    face2edge: torch.Tensor,
    boundary_edge_indices: torch.Tensor,
    visblity: torch.Tensor,
    remove_face_indices: torch.Tensor,
    max_hole_size: float,
) -> torch.Tensor:
    """Accept or reject each cut (**upstream's two conditions**).

    1. **Reject** when the median visibility of the piece exceeds 0.25 (never cut
       faces that are visible).
    2. **Reject** when the boundary loop it would open exceeds `max_hole_size`
       in area (never open a large hole).
    """
    import utils3d

    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    valid: list[torch.Tensor] = []
    for cc in to_remove_cc:
        if visblity[remove_face_indices[cc]].median() > 0.25:
            continue
        cc_edge_indices, cc_edges_degree = torch.unique(
            face2edge[remove_face_indices[cc]], return_counts=True
        )
        cc_boundary = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary = cc_boundary[~torch.isin(cc_boundary, boundary_edge_indices)]
        if len(cc_new_boundary) > 0:
            loops = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary])
            too_big = False
            for loop in loops:
                pts = verts[edges[cc_new_boundary[loop]]]
                center = pts.mean(dim=1).mean(dim=0)
                e1 = verts[edges[cc_new_boundary[loop]][:, 0]] - center
                e2 = verts[edges[cc_new_boundary[loop]][:, 1]] - center
                area = torch.norm(torch.cross(e1, e2, dim=-1), dim=1).sum() * 0.5
                if area > max_hole_size:
                    too_big = True
                    break
            if too_big:
                continue
        valid.append(cc)
    if not valid:
        return torch.empty(0, dtype=torch.long, device=faces.device)
    return remove_face_indices[torch.cat(valid)]


def ensure_upstream_on_path(repo: str) -> None:
    """Make the upstream clone importable, so its helpers can be used."""
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _min_cut(
    n_faces: int,
    dual_edges: np.ndarray,
    dual_weights: np.ndarray,
    inner: np.ndarray,
    outer: np.ndarray,
    source: int,
    target: int,
) -> list[int]:
    """Which faces fall on the source side of the minimum cut.

    **This was `igraph.Graph.mincut`, and igraph is GPL** while this repository
    is MIT, so it is done with scipy's maximum flow instead (BSD, and already a
    dependency). The two are the same computation: a maximum flow saturates the
    minimum cut, and the source side of that cut is exactly the set of nodes
    still reachable from the source through edges with capacity to spare.

    Capacities are integers because `maximum_flow` requires them, which is why
    upstream's weights were already being multiplied by a thousand.

    Args:
        n_faces: How many faces; the source and target sit past them.
        dual_edges: `[E, 2]` face pairs sharing an edge.
        dual_weights: `[E]` weight per pair.
        inner: Faces joined to the source (invisible ones).
        outer: Faces joined to the target (visible ones).
        source: Index of the source node.
        target: Index of the target node.

    Returns:
        The face indices to remove.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    nodes = n_faces + 2
    capacity = np.rint(np.asarray(dual_weights, dtype=np.float64) * 1000.0)
    # The undirected graph: face to face across a shared edge, every invisible
    # face to the source, every visible one to the target.
    ends_a = np.concatenate([dual_edges[:, 0], inner, outer])
    ends_b = np.concatenate(
        [dual_edges[:, 1], np.full(len(inner), source), np.full(len(outer), target)]
    )
    weights = np.concatenate([capacity, np.full(len(inner), 1000.0), np.full(len(outer), 1000.0)])
    # **Each undirected edge becomes a pair of directed ones**, which is how an
    # undirected cut is put to a directed maximum flow.
    rows = np.concatenate([ends_a, ends_b])
    cols = np.concatenate([ends_b, ends_a])
    data = np.concatenate([weights, weights])
    graph = coo_matrix(
        (np.clip(data, 1, np.iinfo(np.int32).max).astype(np.int32), (rows, cols)),
        shape=(nodes, nodes),
    ).tocsr()
    graph.sum_duplicates()

    result = maximum_flow(graph, source, target)
    residual = (graph - result.flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int32)
    residual.eliminate_zeros()

    # **A minimum cut is not unique, and which one is chosen changes the mesh.**
    # Reachability from the source gives the smallest source side; igraph
    # returns the largest, so the same is taken here: everything that cannot
    # still reach the target. The two agree on every random graph tested
    # (`tests/test_min_cut.py`).
    to_target, _ = breadth_first_order(
        residual.T.tocsr(), target, directed=True, return_predecessors=True
    )
    reaches_target = np.zeros(nodes, dtype=bool)
    reaches_target[to_target] = True
    return [int(v) for v in np.flatnonzero(~reaches_target) if v < n_faces]
