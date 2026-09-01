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
`sphere_hammersley_sequence`, `pymeshfix`). Only the sequencing lives here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
import torch


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
    for i in range(views.shape[0]):
        buffers = utils3d.torch.rasterize_triangle_faces(
            ctx, verts[None], faces, resolution, resolution, view=views[i], projection=projection
        )
        face_id = buffers["face_id"][0][buffers["mask"][0] > 0.95] - 1
        seen[torch.unique(face_id).long()] += 1
        if progress is not None and (i + 1) % 50 == 0:
            progress("raster", f"measuring visibility ({i + 1}/{views.shape[0]} views)")
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
        max_hole_nbe: Maximum number of edges in a boundary loop `pymeshfix`
            will fill.
        resolution: Rasterization resolution.
        num_views: Number of viewpoints.
        progress: Where to report the stage.

    Returns:
        `(verts, faces)`.
    """
    import igraph
    import utils3d
    from pymeshfix import _meshfix

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
    g = igraph.Graph()
    g.add_vertices(n_faces + 2)  # the last two are source and target
    source, target = n_faces, n_faces + 1
    g.add_edges(dual_edges.cpu().numpy())
    weights = dual_weights.cpu().numpy().tolist()

    # **This is the only difference from upstream.** The indices move to the CPU
    # in one go before the edges are built. Upstream walks a CUDA tensor element
    # by element in Python, spending 35 seconds on 360,000 GPU reads.
    # **The edges added are the same.**
    inner_list = inner_face_indices.cpu().numpy().tolist()
    outer_list = outer_face_indices.cpu().numpy().tolist()
    g.add_edges([(f, source) for f in inner_list])
    g.add_edges([(f, target) for f in outer_list])
    weights.extend([1.0] * (len(inner_list) + len(outer_list)))

    say("mincut", f"solving the min-cut (inner {len(inner_list):,} / outer {len(outer_list):,})")
    capacities = (np.asarray(weights, dtype=np.float64) * 1000).tolist()
    cut = g.mincut(source, target, capacities)
    remove_face_indices = torch.tensor(
        [v for v in cut.partition[0] if v < n_faces], dtype=torch.long, device=faces.device
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

    say("meshfix", f"filling small holes (up to {max_hole_nbe} edges)")
    fixer = _meshfix.PyTMesh()
    fixer.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    fixer.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    new_verts, new_faces = fixer.return_arrays()
    return (
        torch.tensor(new_verts, device=verts.device, dtype=torch.float32),
        torch.tensor(new_faces, device=faces.device, dtype=torch.int32),
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
