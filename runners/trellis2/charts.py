# SPDX-License-Identifier: MIT
"""Cut a surface into charts by where it faces, not by what its triangles do.

**Why xatlas alone does not work here.** On this model's surfaces the atlas
shattered into charts of five triangles: 50,052 charts from 200,000 faces,
vertices tripled, coverage 44 % (measured 2026-09-12). Two causes, both in
the input rather than the unwrapper. Half the faces were wound the other
way, and xatlas treats an edge whose two faces disagree as a boundary, so
no chart could cross it. And the surface is rough at the scale of a
triangle - the dihedral angle between neighbours has a median of 10° and a
quarter of them exceed 30° - so any chart growth that follows face normals
stops after a few faces. Relaxing every chart option changed nothing.

**So the charts are decided by direction in space.** Face normals are
smoothed over the adjacency until they describe the surface rather than
its triangles, each face is assigned to the one of six axis directions its
smoothed normal points along, and a chart is a connected run of faces
with the same direction. Charts too small to carry a picture are given to
the neighbour they share the most edges with. Each chart is then laid flat
by orthographic projection along its axis - a bounded distortion (at most
the cosine of the angle to the axis, which is what the six directions
limit) and no seam inside a chart. Blender's *Smart UV Project* is the
same idea. Measured on the carved 512 mecha at 200,000 faces: smoothing
alone moved the share of faces in charts of fifty or more from 50 % to
73 %, and merging the rest takes it further (see `tests/test_charts.py`
and `docs/trellis2.md`). A face whose own normal points away from its
chart's axis would fold over in projection; those are moved to a chart
they do not fold in (`_unfold`), and the slivers that cannot be are left
to read the atlas without writing it.

What comes out is one vertex per (vertex, chart) pair with its projected
coordinates in **world units**, so texel density is the same in every chart
when a packer scales them together. **Packing is not done here**: xatlas
does it in a fraction of a second (`Atlas.add_uv_mesh` then `generate`),
and it holds the GIL, so it belongs in `unwrap_worker`.

The mesh must be consistently wound: the direction is signed. The carved
solid (`shell.py`) is; the decoder's raw surface is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from runners.trellis.split_manifold import _edge_counts

# The two axes a chart keeps when projected along the third, in the cyclic
# order whose cross product is the projection axis, so that a chart seen from
# its own side is not mirrored.
_OTHER_AXES = ((1, 2), (2, 0), (0, 1))


@dataclass
class ChartReport:
    """What the charting did, counted."""

    faces: int = 0
    smoothing_rounds: int = 0
    charts_before_merge: int = 0
    charts: int = 0
    merge_rounds: int = 0
    faces_in_large_charts: float = 0.0
    median_faces_per_chart: float = 0.0
    folds_moved: int = 0
    folds_own_chart: int = 0
    folded_faces: int = 0
    folded_area_fraction: float = 0.0
    vertices_before: int = 0
    vertices_after: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of faces across every edge that exactly two faces use."""
    inverse, counts, order = _edge_counts(faces)
    instance_face = np.tile(np.arange(len(faces)), 3)
    paired = order[counts[inverse[order]] == 2].reshape(-1, 2)
    return instance_face[paired[:, 0]], instance_face[paired[:, 1]]


def smooth_normals(
    normals: np.ndarray, left: np.ndarray, right: np.ndarray, rounds: int
) -> np.ndarray:
    """Average each face's normal with its neighbours', `rounds` times, renormalizing."""
    count = len(normals)
    graph = csr_matrix(
        (np.ones(2 * len(left)), (np.concatenate([left, right]), np.concatenate([right, left]))),
        shape=(count, count),
    )
    degree = np.asarray(graph.sum(axis=1)).ravel()
    out = normals.copy()
    for _ in range(rounds):
        out = out + (graph @ out) / np.maximum(degree, 1)[:, None]
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-20)
    return out


_AXIS_VECTORS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
)


def _chart_directions(
    chart: np.ndarray, normals: np.ndarray, area: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each chart's projection direction (0..5) and its area-weighted mean normal."""
    summed = np.zeros((count, 3))
    np.add.at(summed, chart, normals * area[:, None])
    mean = summed / np.maximum(np.linalg.norm(summed, axis=1, keepdims=True), 1e-20)
    return (mean @ _AXIS_VECTORS.T).argmax(axis=1), mean


def _merge_small(
    chart: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    normals: np.ndarray,
    area: np.ndarray,
    min_faces: int,
    max_rounds: int,
    min_cosine: float = 0.34,
) -> tuple[np.ndarray, int]:
    """Give every chart under `min_faces` to a neighbour it can be projected with.

    A neighbour qualifies when the small chart's mean normal points within
    about 70° of the neighbour's projection direction (`min_cosine`):
    merged into a chart that faces the other way, its faces would fold over
    in projection and overlap - measured without this rule, 4 % of the faces
    on the 200 k specimen. Among the qualifying neighbours, the one sharing
    the most edges wins. Rounds repeat because a merge can leave a chart that
    is still small, or make a small chart's best neighbour larger; a small
    chart with no qualifying neighbour stays as it is.
    """
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        count = int(chart.max()) + 1
        sizes = np.bincount(chart, minlength=count)
        small = sizes < min_faces
        across = chart[left] != chart[right]
        if not (small[chart].any() and across.any()):
            rounds -= 1
            break
        a, b = left[across], right[across]
        source = np.concatenate([chart[a], chart[b]])
        target = np.concatenate([chart[b], chart[a]])
        keep = small[source]
        source, target = source[keep], target[keep]
        key = source.astype(np.int64) * count + target
        unique_key, shared = np.unique(key, return_counts=True)
        src = unique_key // count
        dst = unique_key % count
        direction, mean = _chart_directions(chart, normals, area, count)
        compatible = (mean[src] * _AXIS_VECTORS[direction[dst]]).sum(axis=1) >= min_cosine
        # Prefer a neighbour that is already large enough to keep its axis.
        rank = shared * np.where(small[dst], 1, 1000)
        src, dst, rank = src[compatible], dst[compatible], rank[compatible]
        if len(src) == 0:
            rounds -= 1
            break
        order = np.lexsort((-rank, src))
        first = np.r_[True, src[order][1:] != src[order][:-1]]
        relabel = np.arange(count)
        relabel[src[order][first]] = dst[order][first]
        # Follow chains (a small chart merged into a small chart merged into ...).
        for _ in range(8):
            relabel = relabel[relabel]
        merged = relabel[chart]
        if np.array_equal(merged, chart):
            rounds -= 1
            break
        chart = merged
    _, chart = np.unique(chart, return_inverse=True)
    return chart.ravel(), rounds


def _unfold(
    chart: np.ndarray,
    chart_direction: np.ndarray,
    normals: np.ndarray,
    area: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    min_area: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Move each face that folds under its chart's projection to a chart it does not fold in.

    A face folds when its own normal points away from its chart's axis: its
    projection turns over and lands on its neighbours. **The surface is rough
    at the scale of a triangle**, so the smoothed normal that chose the chart
    and the face's own can disagree by more than a right angle - measured on
    the 200 k specimen, 4,250 faces (2.1 %), 2,488 of them against their own
    smoothed direction. Among the charts across the face's edges, the one its
    normal points along best takes it (a cosine above 0.1, so that it does not
    fold there either; 1,215 of the 4,250). A face with no such neighbour gets
    a chart of its own along its own normal **when it is at least `min_area`**,
    and is otherwise left where it is: the rest are slivers (median area a
    third of the median face), and a chart each would have more than doubled
    the chart count. A sliver left folded reads the atlas where it lands and
    writes nothing (`texture.bake`).

    Returns the charts, their directions (extended by the new ones), and how
    many faces moved and how many got a chart of their own.
    """
    dot = (normals * _AXIS_VECTORS[chart_direction[chart]]).sum(axis=1)
    folded = np.flatnonzero(dot <= 0)
    if len(folded) == 0:
        return chart, chart_direction, 0, 0
    is_folded = np.zeros(len(chart), dtype=bool)
    is_folded[folded] = True
    a = np.concatenate([left, right])
    b = np.concatenate([right, left])
    keep = is_folded[a]
    src, dst = a[keep], chart[b[keep]]
    gain = (normals[src] * _AXIS_VECTORS[chart_direction[dst]]).sum(axis=1)
    ok = gain > 0.1
    src, dst, gain = src[ok], dst[ok], gain[ok]
    chart = chart.copy()
    moved = np.zeros(0, dtype=np.int64)
    if len(src):
        order = np.lexsort((-gain, src))
        first = np.r_[True, src[order][1:] != src[order][:-1]]
        moved = src[order][first]
        chart[moved] = dst[order][first]
    rest = folded[~np.isin(folded, moved)]
    own = rest[area[rest] >= min_area]
    if len(own):
        axis = np.abs(normals[own]).argmax(axis=1)
        negative = normals[own, axis] < 0
        chart[own] = len(chart_direction) + np.arange(len(own))
        chart_direction = np.concatenate([chart_direction, axis * 2 + negative.astype(np.int64)])
    labels, chart = np.unique(chart, return_inverse=True)
    return chart.ravel(), chart_direction[labels], int(len(moved)), int(len(own))


def project_charts(
    mesh: trimesh.Trimesh,
    smoothing_rounds: int = 10,
    min_faces: int = 50,
    max_merge_rounds: int = 30,
    fold_area: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ChartReport]:
    """Cut `mesh` into axis-projected charts.

    Args:
        mesh: A consistently wound triangle mesh (`is_winding_consistent`).
        smoothing_rounds: How many times the normals are averaged over the
            adjacency before the direction is read off them. **Measured 10**:
            on the 200 k specimen 3 rounds put 64 % of the faces in charts of
            fifty or more, 10 put 69 %, 30 put 73 %, with diminishing returns
            and a growing blur of genuinely different directions.
        min_faces: Charts smaller than this are merged into a neighbour.
        max_merge_rounds: A bound on the merging, which normally converges in
            a few rounds.
        fold_area: A face that folds under its chart's projection and has no
            neighbouring chart to take it gets a chart of its own when its
            area is at least this many times the median face's (`_unfold`).
            **Measured 2**: on the 200 k specimen 73 faces qualify and hold
            0.08 % of the area; the 2,962 slivers below it hold 0.4 %.

    Returns:
        `(vertices, faces, uvs, report)`: one vertex per (vertex, chart) pair,
        faces over those, and each vertex's projected coordinates in world
        units. Pack them (xatlas, `add_uv_mesh`) before baking.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    report = ChartReport(faces=int(len(faces)), smoothing_rounds=int(smoothing_rounds))
    if len(faces) == 0:
        raise ValueError("cannot chart a mesh with no faces")

    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(area, 1e-20)[:, None]
    left, right = _adjacency(faces)
    smoothed = (
        smooth_normals(normals, left, right, smoothing_rounds) if smoothing_rounds > 0 else normals
    )

    axis = np.abs(smoothed).argmax(axis=1)
    negative = smoothed[np.arange(len(faces)), axis] < 0
    direction = axis * 2 + negative.astype(np.int64)

    same = direction[left] == direction[right]
    graph = coo_matrix(
        (np.ones(int(same.sum())), (left[same], right[same])), shape=(len(faces), len(faces))
    )
    count, chart = connected_components(graph, directed=False)
    report.charts_before_merge = int(count)

    chart, merge_rounds = _merge_small(
        chart, left, right, normals, area, min_faces, max_merge_rounds
    )
    report.merge_rounds = int(merge_rounds)
    count = int(chart.max()) + 1

    # A merged chart projects along the direction its area faces on average;
    # the faces that would fold under it are then moved out of it.
    chart_direction, _mean = _chart_directions(chart, normals, area, count)
    chart, chart_direction, report.folds_moved, report.folds_own_chart = _unfold(
        chart, chart_direction, normals, area, left, right, fold_area * float(np.median(area))
    )
    count = int(chart.max()) + 1
    report.charts = count
    sizes = np.bincount(chart, minlength=count)
    report.faces_in_large_charts = round(float(sizes[sizes >= min_faces].sum() / len(faces)), 4)
    report.median_faces_per_chart = float(np.median(sizes))
    chart_axis = chart_direction // 2
    chart_negative = chart_direction % 2 == 1

    # One vertex per (vertex, chart): a vertex on a chart boundary has a
    # place in each chart it touches.
    corner_vertex = faces.ravel()
    corner_chart = np.repeat(chart, 3)
    key = corner_vertex * count + corner_chart
    unique_key, first, new_index = np.unique(key, return_index=True, return_inverse=True)
    split_vertex = corner_vertex[first]
    split_chart = corner_chart[first]
    split_faces = new_index.reshape(-1, 3)

    others = np.array(_OTHER_AXES)[chart_axis[split_chart]]  # [N, 2]
    position = vertices[split_vertex]
    uvs = np.stack(
        [
            position[np.arange(len(position)), others[:, 0]],
            position[np.arange(len(position)), others[:, 1]],
        ],
        axis=1,
    )
    # Seen from its own side, a chart facing a negative direction is mirrored
    # by the cyclic projection; mirroring u again puts it right.
    uvs[chart_negative[split_chart], 0] *= -1.0

    # The faces still folded after `_unfold` are the slivers it left: counted,
    # with the area they hold, because they overlap their chart in the atlas.
    p = uvs[split_faces]
    signed = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (
        p[:, 2, 0] - p[:, 0, 0]
    )
    folded = signed < 0
    report.folded_faces = int(folded.sum())
    report.folded_area_fraction = round(float(area[folded].sum() / max(area.sum(), 1e-20)), 5)
    report.vertices_before = int(len(vertices))
    report.vertices_after = int(len(split_vertex))
    return position, split_faces, uvs.astype(np.float32), report
