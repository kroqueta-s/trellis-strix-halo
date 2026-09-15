# SPDX-License-Identifier: MIT
"""Make a printable solid out of the surface. **The model does not produce one.**

What the decoder emits is a thin, double-walled skin - an outer and an inner
surface two or three cells apart at 512 - and it is incomplete: outlines in a
cross-section are open arcs as often as closed loops, so the space inside is
connected to the space outside through gaps at every scale (measured
2026-09-12 on the construction-mecha specimen: sealing the floor and closing
openings up to 24 cells wide both left the interior reachable). A manifold
sewn out of that surface encloses a tenth of what the silhouette suggests,
with a quarter of it wound inside-out, and no flood fill can say what is
inside.

**What can say it is visibility.** A point of the exterior can be seen from
far away in many directions; a point in the hollow behind the skin is seen,
if at all, only through a gap, from a few. So the exterior is carved out by
casting rays in 98 directions across the lattice and keeping, as air, every
corner that escapes unblocked in at least `visibility` of them - the space
carving of a visual hull, with the surface itself as the occluder. Everything
else is solid: the walls, the hollow behind them, every pocket. The outer
surface stays exactly where the model put it, nothing grows outward, and the
interior is filled - which is what a slicer wants, and what `forge.hollow`
takes apart again downstream when a print should be hollow. On the 512
specimen the count of visible directions is sharply bimodal (3.2 M corners
see none, a plateau from twelve to sixteen, the exterior at ninety and more);
the threshold is 4 because a bowl twice as deep as it is wide keeps its
hollow up to 4 and starts to fill at 6 (measured on an annulus with a floor),
and the mecha's solid moves by 0.002 between 2 and 4. The rays take a few
seconds on the GPU.

The older construction is kept as `mode="band"`: every point within half a
wall of the surface is solid, which guarantees a wall thickness but grows
the silhouette by half of it and rounds off detail narrower than the wall.

**No dependency beyond numpy, scipy, trimesh and torch**; the rays run on the
GPU when there is one and on the CPU otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import trimesh
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

# The four cells around a lattice edge, listed so that the quad's normal points
# along the edge's axis (right-hand rule: for the edge axis a, the other two
# axes are taken in the cyclic order (a+1, a+2) mod 3, whose cross product is a).
_OTHER_AXES = ((1, 2), (2, 0), (0, 1))
_QUAD_OFFSETS = ((0, 0), (1, 0), (1, 1), (0, 1))


def _primitive_directions(reach: int) -> tuple[tuple[int, int, int], ...]:
    """Every integer direction with components in `-reach..reach`, each once.

    A vector that is a multiple of a shorter one is the same ray, so it is
    dropped. `reach` 1 gives the 26 neighbours; 2 gives 98 directions, fine
    enough that the floor of a bowl twice as deep as it is wide still sees
    the sky in several of them.
    """
    from math import gcd

    out = []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                if gcd(gcd(abs(dx), abs(dy)), abs(dz)) != 1:
                    continue
                out.append((dx, dy, dz))
    return tuple(out)


_DIRECTIONS = _primitive_directions(2)


@dataclass
class ShellReport:
    """What making the solid did, counted."""

    mode: str = "carve"
    grid: int = 0
    lattice: int = 0
    thickness: float = 0.0
    half_cells: int = 0
    visibility: int = 0
    surface_corners: int = 0
    solid_corners: int = 0
    exterior_fraction: float = 0.0
    pockets_filled: int = 0
    islands_dropped: int = 0
    island_corners_dropped: int = 0
    sheet_corners: int = 0
    crossing_edges: int = 0
    faces: int = 0
    vertices: int = 0
    smooth_rounds: int = 0
    smooth_moved_cells: float = 0.0
    smooth_moved_p99_cells: float = 0.0
    snap_reach_cells: float = 0.0
    snapped: float = 0.0
    snap_moved_cells: float = 0.0
    snap_moved_p99_cells: float = 0.0
    snap_held_back: int = 0
    snap_sec: float = 0.0
    rasterize_sec: float = 0.0
    occupancy_sec: float = 0.0
    extract_sec: float = 0.0
    smooth_sec: float = 0.0
    device: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        cell = 1.0 / max(self.grid, 1)
        # As a fraction of the cube spanned by the longest side, the same unit
        # `volume` in `metrics.topology` is in after normalization.
        out["solid_volume"] = round(self.solid_corners * cell**3, 5)
        return out


def _sample_surface(vertices: np.ndarray, faces: np.ndarray, max_edge: float = 0.0) -> np.ndarray:
    """Points on every triangle, spaced closely enough to mark every corner it touches.

    Thirteen fixed points cover a triangle up to about a cell across. **A wider
    one gets a barycentric grid instead**, spaced at half a cell: the flat
    panels decimation leaves, or a coarse test shape, would otherwise have
    corners unmarked between their samples, and a ray then slips through a
    wall that is there. Measured on a 256-face annulus at grid 64: with
    thirteen points alone the rays reached the whole inside and the solid
    came out at 5 % of its volume. (Subdividing the mesh instead was measured
    at 24 s on 1.3 M faces; this is a fraction of a second.)
    """
    tri = vertices[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    if max_edge > 0:
        longest = np.maximum(
            np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1)),
            np.linalg.norm(a - c, axis=1),
        )
        steps = np.ceil(2.0 * longest / max_edge).astype(np.int64)
    else:
        steps = np.zeros(len(faces), dtype=np.int64)
    small = steps <= 2
    m = tri.mean(axis=1)
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    parts = [
        a[small],
        b[small],
        c[small],
        m[small],
        ab[small],
        bc[small],
        ca[small],
        ((a + m) / 2)[small],
        ((b + m) / 2)[small],
        ((c + m) / 2)[small],
        ((ab + m) / 2)[small],
        ((bc + m) / 2)[small],
        ((ca + m) / 2)[small],
    ]
    # The wide triangles, grouped by how many steps their longest edge needs,
    # each group sampled on one barycentric grid of that spacing.
    for n in np.unique(steps[~small]):
        chosen = steps == n
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        u = (i[keep] / n)[None, :, None]
        v = (j[keep] / n)[None, :, None]
        w = 1.0 - u - v
        grid = w * a[chosen][:, None, :] + u * b[chosen][:, None, :] + v * c[chosen][:, None, :]
        parts.append(grid.reshape(-1, 3))
    return np.concatenate(parts)


def _shifted(vis: torch.Tensor, direction: tuple[int, int, int], k: int) -> torch.Tensor:
    """`vis[p + k * direction]`, with everything outside the lattice counted as visible."""
    out = torch.ones_like(vis)
    size = vis.shape[0]
    src: list[slice] = [slice(None)] * 3
    dst: list[slice] = [slice(None)] * 3
    for axis, step in enumerate(direction):
        s = step * k
        if abs(s) >= size:
            # A jump that leaves the lattice altogether sees only outside.
            return out
        if s > 0:
            src[axis] = slice(s, None)
            dst[axis] = slice(0, size - s)
        elif s < 0:
            src[axis] = slice(0, size + s)
            dst[axis] = slice(-s, None)
    out[tuple(dst)] = vis[tuple(src)]
    return out


def visible_directions(barrier: np.ndarray, device: str | None = None) -> np.ndarray:
    """How many of the 98 directions each corner can see the outside in, past `barrier`.

    A ray escapes when every corner it passes is free, and the lattice's edge
    counts as escaped. The test along a ray is folded by doubling - `k` steps
    of visibility become `2k` in one shifted AND - so each direction costs
    `log2(L)` passes over the lattice rather than `L`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    free = torch.as_tensor(~barrier, device=device)
    count = torch.zeros(free.shape, dtype=torch.uint8, device=device)
    size = free.shape[0]
    for direction in _DIRECTIONS:
        vis = free.clone()
        k = 1
        while k < size:
            vis &= _shifted(vis, direction, k)
            k *= 2
        count += vis.to(torch.uint8)
    return count.cpu().numpy()


def _surface_nets(
    solid: np.ndarray, distance: np.ndarray, pad: int, scale: float, origin: np.ndarray
) -> tuple[trimesh.Trimesh, int]:
    """The boundary of `solid`, one vertex per boundary cell, wound outward.

    A cell's vertex is the mean of the points where its edges cross the
    boundary, and a crossing sits where `distance` to the surface is least
    along the edge - the surface nets rule, which keeps the result smooth
    rather than stepped.
    """
    quads: list[np.ndarray] = []
    inside_low: list[np.ndarray] = []
    points: list[np.ndarray] = []
    for axis in range(3):
        low: list[slice] = [slice(None)] * 3
        low[axis] = slice(0, -1)
        high: list[slice] = [slice(None)] * 3
        high[axis] = slice(1, None)
        crossing = solid[tuple(low)] != solid[tuple(high)]
        start = np.argwhere(crossing)
        if len(start) == 0:
            continue
        end = start.copy()
        end[:, axis] += 1
        d_low = distance[start[:, 0], start[:, 1], start[:, 2]]
        d_high = distance[end[:, 0], end[:, 1], end[:, 2]]
        total = d_low + d_high
        safe = np.where(total > 1e-9, total, 1.0)
        t = np.where(total > 1e-9, d_low / safe, 0.5).clip(0.0, 1.0)
        point = start.astype(np.float64)
        point[:, axis] += t
        points.append(point)
        inside_low.append(solid[start[:, 0], start[:, 1], start[:, 2]])
        # The four cells sharing this edge: one step back along the other two
        # axes and forward by the offsets (cell (i, j, k) spans corners i..i+1).
        b, c = _OTHER_AXES[axis]
        quad = np.empty((len(start), 4, 3), dtype=np.int64)
        for slot, (ob, oc) in enumerate(_QUAD_OFFSETS):
            cell = start.copy()
            cell[:, b] += ob - 1
            cell[:, c] += oc - 1
            quad[:, slot] = cell
        quads.append(quad)
    if not quads:
        raise ValueError("the solid has no surface: the mesh may be smaller than one cell")

    quad = np.concatenate(quads)
    point = np.concatenate(points)
    flip = ~np.concatenate(inside_low)
    cells = solid.shape[0] - 1
    linear = (quad[:, :, 0] * cells + quad[:, :, 1]) * cells + quad[:, :, 2]
    unique_cells, index = np.unique(linear.ravel(), return_inverse=True)
    index = index.reshape(-1, 4)
    position = np.zeros((len(unique_cells), 3), dtype=np.float64)
    weight = np.zeros(len(unique_cells), dtype=np.float64)
    for slot in range(4):
        np.add.at(position, index[:, slot], point)
        np.add.at(weight, index[:, slot], 1.0)
    position /= weight[:, None]
    position = (position - pad) / scale + origin

    # Outward is from the solid side to the empty side.
    index[flip] = index[flip][:, ::-1]
    p = position[index]
    diagonal_a = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    diagonal_b = np.linalg.norm(p[:, 1] - p[:, 3], axis=1)
    use_a = (diagonal_a <= diagonal_b)[:, None]
    split_a = index[:, [0, 1, 2, 0, 2, 3]]
    split_b = index[:, [0, 1, 3, 1, 2, 3]]
    triangles = np.where(use_a, split_a, split_b).reshape(-1, 3)
    return trimesh.Trimesh(vertices=position, faces=triangles, process=False), int(len(quad))


def _snap_to_source(
    vertices: np.ndarray,
    faces: np.ndarray,
    samples: np.ndarray,
    cell: float,
    reach: float,
    share: float = 0.4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Put the surface back where the model drew it, wherever that is safe.

    **The carve decides what is solid; it does not have to decide where the
    surface is.** The lattice settles the topology - which cells are inside,
    which sheets exist - and that part it is good at. The position it then
    hands out is a lattice position, and the model's own is better: measured
    2026-09-13 on the mechanical beetle, the mesh going into the carve sits
    0.23 of a cell from the decoder's surface and the mesh coming out sits
    0.72, with creases at 0.97. None of that is the decimation or the repair;
    every bit is added here.

    So each vertex is moved onto the nearest point of the surface that was
    rasterized, and the numbers come most of the way back: 0.72 to 0.38 on the
    flats and 0.97 to 0.49 at the creases, with 57% of the vertices moving.

    **Two limits keep it from closing a thin wall.** A vertex whose source is
    further than `reach` cells away is left alone, because there is nothing
    there to snap to and it is standing on a bridge the carve invented. And a
    vertex may cross at most `share` of the room between it and the surface
    facing back at it, since that surface is moving towards the same sheet -
    without this the two sides of a membrane meet and the wall is gone.
    """
    if reach <= 0 or len(samples) == 0:
        return vertices, {
            "snapped": 0.0,
            "moved_cells": 0.0,
            "moved_p99_cells": 0.0,
            "snap_held_back": 0,
        }

    normals = np.zeros_like(vertices)
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(normals, faces[:, i], face_n)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    # How much room each vertex has before it meets its own other side.
    own = cKDTree(vertices, balanced_tree=False, compact_nodes=False)
    gap, near = own.query(vertices, k=16, workers=-1)
    facing = (normals[near] * normals[:, None, :]).sum(axis=2) < -0.3
    room = np.where(facing & (gap > 0.0), gap, np.inf).min(axis=1)

    # **The tree is built unbalanced on purpose.** The samples are a surface,
    # so they are far from uniform in space, and scipy's balanced build turns
    # that into a tree whose queries thrash: measured 2026-09-13 on 29,332,568
    # of the beetle's samples, the balanced tree took 8.2 s to build and 107.5 s
    # to answer 1,028,371 queries, and the unbalanced one 3.0 s and 0.7 s.
    source = np.asarray(samples, dtype=np.float64)
    tree = cKDTree(source, balanced_tree=False, compact_nodes=False)
    distance, index = tree.query(vertices, workers=-1)
    step = source[index] - vertices
    limit = np.minimum(reach * cell, share * room)[:, None]
    length = np.linalg.norm(step, axis=1, keepdims=True)
    step = np.where(length > limit, step * (limit / np.maximum(length, 1e-12)), step)
    step[distance > reach * cell] = 0.0

    # **A vertex may not drag its own triangles through each other.** Two
    # corners of one triangle can have the same nearest point - on a ridge a
    # cell wide, or where the surface doubles back - and moving both onto it
    # leaves a triangle with no area, or one wound the other way. Measured
    # 2026-09-13 before this guard: 12,514 triangles of the beetle's 2,080,540
    # came out with zero area and 8,317 wound backwards, and the atlas built on
    # them asked for 44 GB. So the step is halved, for every vertex of every
    # triangle that would spoil, until none do.
    # **A sliver is as bad as a hole.** A triangle with area left but no width
    # still cannot be laid flat, and the atlas spends the whole bake on it:
    # measured 2026-09-13 on the dragon, 26 of them took the bake from 4.0 s to
    # 14.0 s. A hundredth of a cell's area leaves none of them, and costs
    # nothing measurable - the crease error moves 0.229 to 0.232 of a cell.
    area_floor = 1e-2 * cell * cell
    tri = vertices[faces]
    before = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    spoiled = 0
    for _ in range(8):
        moved_tri = (vertices + step)[faces]
        after = np.cross(moved_tri[:, 1] - moved_tri[:, 0], moved_tri[:, 2] - moved_tri[:, 0])
        bad = (np.linalg.norm(after, axis=1) < area_floor) | ((after * before).sum(axis=1) <= 0.0)
        spoiled = int(bad.sum())
        if spoiled == 0:
            break
        touched = np.zeros(len(vertices), dtype=bool)
        touched[faces[bad].ravel()] = True
        step[touched] *= 0.5
    if spoiled:
        # Whatever a halving could not rescue does not move at all.
        moved_tri = (vertices + step)[faces]
        after = np.cross(moved_tri[:, 1] - moved_tri[:, 0], moved_tri[:, 2] - moved_tri[:, 0])
        bad = (np.linalg.norm(after, axis=1) < area_floor) | ((after * before).sum(axis=1) <= 0.0)
        stuck = np.zeros(len(vertices), dtype=bool)
        stuck[faces[bad].ravel()] = True
        step[stuck] = 0.0

    moved = np.linalg.norm(step, axis=1)
    live = moved > 0.0
    return vertices + step, {
        "snapped": round(float(live.mean()), 4),
        "moved_cells": round(float(np.median(moved[live]) / cell) if live.any() else 0.0, 3),
        "moved_p99_cells": round(
            float(np.percentile(moved[live], 99) / cell) if live.any() else 0.0, 3
        ),
        "snap_held_back": int(spoiled),
    }


def _smooth_taubin(
    vertices: np.ndarray, faces: np.ndarray, rounds: int, lam: float = 0.5, mu: float = -0.53
) -> tuple[np.ndarray, float, float]:
    """Take the lattice's terraces off the surface without shrinking it.

    **A surface that crosses the lattice at a shallow angle comes out in
    steps.** Surface nets places one vertex per boundary cell and averages the
    crossings on its edges, which smooths a corner that spans several cells but
    has nothing to work with on a membrane one or two cells thick: the dragon's
    wings came out as flat terraces a cell apart, and no choice of distance
    field moves them (measured 2026-09-13: swapping the chamfer distance for the
    exact Euclidean one changed the roughness by 0.001 of a cell).

    Taubin's two steps are what fixes it. The positive pass is an ordinary
    Laplacian, which removes the steps and shrinks the model; the negative pass
    pushes back out at a slightly larger factor, and the pair leaves the volume
    where it was - measured 1.0017x on the dragon and 1.0007x on the beetle at
    five rounds, with vertices moving a median of 0.13 and 0.18 of a cell.
    **Creases survive it**: the beetle's panel lines and the recess in its head
    are still sharp at ten rounds, because a crease is a feature the whole
    neighbourhood agrees on and a terrace is not.

    Returns the new vertices and how far they moved, median and 99th centile,
    in the units the vertices are in.
    """
    if rounds <= 0:
        return vertices, 0.0, 0.0
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    who = edges[:, 0]
    count = np.bincount(who, minlength=len(vertices)).astype(np.float64)
    live = count > 0
    # **The neighbourhood is the same every round, so it is built once.** Doing
    # the averaging with `np.add.at` instead means walking twelve million edge
    # entries per pass through numpy's unbuffered scatter, which is its slow
    # path: measured 2026-09-15 on a 1.5 M-face solid, 9.15 s of the carve
    # against 0.92 s for one sparse matrix reused across the passes, agreeing
    # to 5e-16. A row of `average` holds a vertex's neighbours, each weighted
    # by one over how many there are, so a pass is a single matrix product.
    adjacency = sparse.csr_matrix(
        (np.ones(len(who)), (who, edges[:, 1])), shape=(len(vertices), len(vertices))
    )
    average = sparse.diags(1.0 / np.where(live, count, 1.0)) @ adjacency
    moving = vertices.copy()
    for _ in range(int(rounds)):
        for step in (lam, mu):
            delta = np.where(live[:, None], average @ moving - moving, 0.0)
            moving += step * delta
    moved = np.linalg.norm(moving - vertices, axis=1)
    return moving, float(np.median(moved)), float(np.percentile(moved, 99))


def _drop_islands(
    solid: np.ndarray, min_corners: int, structure: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Turn every connected piece of the solid smaller than `min_corners` into air.

    Returns the solid, the number of pieces dropped and the corners they held.
    The largest piece is always kept, whatever its size.
    """
    if min_corners <= 0:
        return solid, 0, 0
    labels, count = ndimage.label(solid, structure=structure)
    if count <= 1:
        return solid, 0, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = sizes >= min_corners
    keep[int(sizes.argmax())] = True
    dropped = int(count - keep[1:].sum())
    if dropped == 0:
        return solid, 0, 0
    corners = int(sizes[~keep].sum())
    return keep[labels], dropped, corners


def solidify(
    mesh: trimesh.Trimesh,
    grid: int = 512,
    mode: str = "carve",
    visibility: int = 4,
    thickness: float = 0.0375,
    fill_cavities: bool = True,
    device: str | None = None,
    island_corners: int = 64,
    smooth: int = 5,
    snap: float = 1.5,
) -> tuple[trimesh.Trimesh, ShellReport]:
    """Turn a surface into a closed solid.

    Args:
        mesh: Any triangle soup. Orientation, holes and self-intersections do
            not matter: only where the surface *is* matters.
        grid: Cells along the longest side. Memory is about `lattice^3` times
            a dozen bytes; 512 peaks near 2.5 GB.
        mode: `"carve"` keeps the outer surface where it is and fills what is
            behind it (see the module docstring); `"band"` makes every point
            within half a `thickness` of the surface solid.
        visibility: Carving only. A corner is air when it can see the outside
            in at least this many of 98 directions. **Measured 4**: deep
            concavities stay open up to 4 and start to fill at 6, while the
            specimen's interior is filled from 2 on.
        thickness: Band only. The wall, as a fraction of the longest side
            (0.0375 is 3 mm on an 80 mm print); the silhouette grows by half.
        fill_cavities: Band only. Fill every pocket the outside cannot reach.
            Carving always fills them.
        device: Where the rays run; `None` picks the GPU when there is one.
        snap: How far, in cells, a vertex may be moved back onto the surface it
            was carved from (see `_snap_to_source`). **Measured 1.5**
            (2026-09-13): the distance to the decoder's own surface falls from
            0.72 of a cell to 0.38 on the flats and 0.97 to 0.49 at the
            creases, for 57% of the vertices moving and 3.8% of the solid's
            volume. 0 leaves the surface where the lattice put it.
        smooth: Rounds of Taubin smoothing on the extracted surface, which is
            what takes the lattice's terraces off a thin wall (see
            `_smooth_taubin`). **Measured 5** (2026-09-13): three already
            removes the terracing on the dragon's wings, five is smoother
            still, and at ten the beetle's panel lines are beginning to soften.
            The volume moves by 0.07-0.17% and the median vertex by a fifth of
            a cell. 0 turns it off.
        island_corners: Carving only. A piece of solid not connected to the
            rest and smaller than this many corners is air. **Measured 64**
            (2026-09-12, 512 specimen): the strays are 1-4 cells across, and
            dropping everything under 8 or under 64 corners leaves the volume
            unchanged to five digits. 0 keeps every piece.

    Returns:
        A closed, consistently wound triangle mesh, and the report. Sheets
        that touch along an edge are separated by the caller
        (`split_manifold.make_manifold`).
    """
    if mode not in ("carve", "band"):
        raise ValueError(f"unknown shell mode {mode!r}: expected 'carve' or 'band'")
    report = ShellReport(mode=mode, grid=int(grid), thickness=float(thickness))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("cannot solidify a mesh with no faces")

    # --- rasterize the surface onto the corner lattice -------------------
    mark = time.perf_counter()
    referenced = vertices[np.unique(faces.ravel())]
    origin = referenced.min(axis=0)
    extent = float((referenced.max(axis=0) - origin).max())
    half = max(1, int(round(thickness * grid / 2))) if mode == "band" else 0
    pad = half + 2
    lattice = int(grid) + 2 * pad
    scale = (grid - 1) / max(extent, 1e-12)
    samples = _sample_surface(vertices, faces, max_edge=1.0 / scale)
    corner = np.rint((samples - origin) * scale).astype(np.int64) + pad
    corner = corner.clip(0, lattice - 1)
    thin = np.zeros((lattice,) * 3, dtype=bool)
    thin[corner[:, 0], corner[:, 1], corner[:, 2]] = True
    # **The samples are kept for the snap.** They are the surface as the carve
    # saw it, already spaced at half a cell, so the snap costs a tree rather
    # than a second pass over the mesh. float32 halves what that costs.
    source = samples.astype(np.float32) if snap > 0 and mode == "carve" else None
    del samples, corner
    report.half_cells = half
    report.lattice = lattice
    report.surface_corners = int(thin.sum())
    report.rasterize_sec = round(time.perf_counter() - mark, 2)

    # --- the solid -----------------------------------------------------
    mark = time.perf_counter()
    six = ndimage.generate_binary_structure(3, 1)
    if mode == "band":
        distance = ndimage.distance_transform_edt(~thin)
        band = distance <= half
        if fill_cavities:
            labels, count = ndimage.label(~band, structure=six)
            solid = labels != int(labels[0, 0, 0])
            del labels
            report.pockets_filled = max(int(count) - 1, 0)
        else:
            solid = band
        del band
        # Crossings sit on the band's edge, `half` cells from the surface.
        distance = np.abs(distance - half)
    else:
        # One corner of dilation seals the diagonal gaps a rasterized surface
        # has, so that a ray cannot slip between two corners of one triangle.
        barrier = ndimage.binary_dilation(thin, structure=six, iterations=1)
        report.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        count = visible_directions(barrier, report.device)
        exterior = count >= int(visibility)
        report.visibility = int(visibility)
        report.exterior_fraction = round(float(exterior.mean()), 4)
        # Air that does not connect to the lattice's edge is a pocket.
        labels, regions = ndimage.label(exterior, structure=six)
        exterior = labels == int(labels[0, 0, 0])
        del labels
        report.pockets_filled = max(int(regions) - 1, 0)
        # The barrier's outer layer was only there to stop rays; taking it
        # off puts the boundary back on the surface's own corners.
        solid = ndimage.binary_erosion(~exterior, structure=six, iterations=1) | thin
        del barrier, exterior, count
        # **Specks of "solid" hang in the air where the rays could not
        # reach.** A corner in the shadow of a strut sees the outside in
        # fewer directions than the threshold, and the erosion above strips
        # a layer off such a pocket, not its middle. Measured on the 512
        # specimen (2026-09-12): 254 connected pieces of solid, 213 of them
        # a single corner, 2-11 cells away from the body and none on the
        # sampled surface. Each becomes a part of its own, and parts are
        # what the manifold repair downstream pays for.
        solid, report.islands_dropped, report.island_corners_dropped = _drop_islands(
            solid, int(island_corners), six
        )
        # **Sheets one corner thick are counted, not thickened.** Where the
        # model drew a single skin with air on both sides, the solid is that
        # skin alone - a slab half a cell thick once extracted. Growing it
        # would not make it printable (three corners is 0.5 mm on an 80 mm
        # print) and would move a surface the operator asked to keep; the
        # count says how much of the model is like that.
        core = ndimage.binary_erosion(solid, structure=six, iterations=1)
        sheet = solid & ~ndimage.binary_dilation(core, structure=six, iterations=1)
        report.sheet_corners = int(sheet.sum())
        del core, sheet
        # A chamfer distance is enough here, at a tenth of the exact one's cost.
        # **Half a corner is added so that no crossing lands on a lattice
        # corner.** With the surface's own corners at distance 0, every edge
        # leaving one would cross at that corner, and two cells whose crossing
        # edges leave the same corners - the two sides of a convex edge, the
        # two sides of a thin sheet - would average to one point. That is a
        # zero-thickness slab, and welding vertices by position (forge's repair
        # does, first thing) turned it into non-manifold edges: measured
        # 2026-09-12, 6,527 coincident vertices and 15,693 such edges on the
        # 512 specimen. At 0.5 the crossing sits a quarter of the way along
        # the edge, unique to it, and the surface moves out by a quarter of a
        # cell - 0.04 mm on an 80 mm print.
        chamfer = ndimage.distance_transform_cdt(~thin, metric="taxicab")
        distance = chamfer.astype(np.float32) + 0.5
    del thin
    report.solid_corners = int(solid.sum())
    report.occupancy_sec = round(time.perf_counter() - mark, 2)

    # --- the surface -----------------------------------------------------
    mark = time.perf_counter()
    shell, crossings = _surface_nets(solid, distance, pad, scale, origin)
    del solid, distance
    report.crossing_edges = crossings
    report.extract_sec = round(time.perf_counter() - mark, 2)

    if smooth > 0:
        mark = time.perf_counter()
        cell = 1.0 / max(int(grid), 1) * max(extent, 1e-12)
        moved, median, p99 = _smooth_taubin(
            np.asarray(shell.vertices, dtype=np.float64),
            np.asarray(shell.faces, dtype=np.int64),
            int(smooth),
        )
        shell = trimesh.Trimesh(vertices=moved, faces=shell.faces, process=False)
        report.smooth_rounds = int(smooth)
        report.smooth_moved_cells = round(median / cell, 3)
        report.smooth_moved_p99_cells = round(p99 / cell, 3)
        report.smooth_sec = round(time.perf_counter() - mark, 2)

    if source is not None and snap > 0:
        mark = time.perf_counter()
        cell = 1.0 / max(int(grid), 1) * max(extent, 1e-12)
        moved, stats = _snap_to_source(
            np.asarray(shell.vertices, dtype=np.float64),
            np.asarray(shell.faces, dtype=np.int64),
            source,
            cell,
            float(snap),
        )
        shell = trimesh.Trimesh(vertices=moved, faces=shell.faces, process=False)
        report.snap_reach_cells = float(snap)
        report.snapped = stats["snapped"]
        report.snap_moved_cells = stats["moved_cells"]
        report.snap_moved_p99_cells = stats["moved_p99_cells"]
        report.snap_held_back = int(stats["snap_held_back"])
        report.snap_sec = round(time.perf_counter() - mark, 2)

    report.faces = int(len(shell.faces))
    report.vertices = int(len(shell.vertices))
    return shell, report


def thicken(
    mesh: trimesh.Trimesh,
    grid: int = 512,
    thickness: float = 0.0375,
    fill_cavities: bool = True,
) -> tuple[trimesh.Trimesh, ShellReport]:
    """The band construction, by its old name."""
    return solidify(mesh, grid=grid, mode="band", thickness=thickness, fill_cavities=fill_cavities)
