# SPDX-License-Identifier: MIT
"""Thicken the surface into a printable shell. **The model does not produce a solid.**

What the decoder emits is a thin, double-walled skin - an outer and an inner
surface two or three cells apart at 512 - with openings wider than 24 cells
where parts meet and pipes end (measured 2026-09-12 on the construction-mecha
specimen: flooding from outside reaches the interior however the openings are
closed, up to a closing radius of 12 cells). So there is no inside to fill,
and a manifold sewn out of that surface encloses a volume that is a tenth of
what the outer silhouette suggests, with a quarter of it wound inside-out.

**A print needs a solid, so one is made from the surface itself.** Every point
within half the wall thickness of the surface is solid, and so is every pocket
that cannot be reached from outside. That is what Blender's solidify and
OpenVDB's mesh-to-volume do with a surface soup, and it makes three promises
the sewn manifold could not: the result is closed by construction, its
orientation is decided by which side is solid rather than propagated from a
face that had none, and its walls are at least as thick as asked.

The price is detail narrower than the wall: a gap between two track links
closes when it is thinner than the wall is thick, because a wall cannot be
thinner than itself. `metrics.post.shell` reports the wall in cells and the
cavities it filled, so a caller can see what was traded.

**No dependency beyond numpy, scipy and trimesh**, and nothing here touches the
GPU. On the 512 specimen (1.3 M faces, a 536-cell lattice) the distance
transform is 12.6 s and everything else is a few seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

# The four cells around a lattice edge, listed so that the quad's normal points
# along the edge's axis (right-hand rule: for the edge axis a, the other two
# axes are taken in the cyclic order (a+1, a+2) mod 3, whose cross product is a).
_OTHER_AXES = ((1, 2), (2, 0), (0, 1))
_QUAD_OFFSETS = ((0, 0), (1, 0), (1, 1), (0, 1))


@dataclass
class ShellReport:
    """What thickening did, counted."""

    grid: int = 0
    half_cells: int = 0
    thickness: float = 0.0
    lattice: int = 0
    surface_corners: int = 0
    band_corners: int = 0
    cavities_filled: int = 0
    cavity_corners: int = 0
    crossing_edges: int = 0
    faces: int = 0
    vertices: int = 0
    rasterize_sec: float = 0.0
    distance_sec: float = 0.0
    flood_sec: float = 0.0
    extract_sec: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        cell = 1.0 / max(self.grid, 1)
        # Volumes as a fraction of the cube spanned by the longest side, the
        # same unit `volume` in `metrics.topology` is in after normalization.
        out["band_volume"] = round(self.band_corners * cell**3, 5)
        out["cavity_volume"] = round(self.cavity_corners * cell**3, 5)
        return out


def _sample_surface(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Thirteen points per triangle, so one a cell across marks every corner it touches."""
    tri = vertices[faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    m = tri.mean(axis=1)
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    return np.concatenate(
        [
            a,
            b,
            c,
            m,
            ab,
            bc,
            ca,
            (a + m) / 2,
            (b + m) / 2,
            (c + m) / 2,
            (ab + m) / 2,
            (bc + m) / 2,
            (ca + m) / 2,
        ]
    )


def thicken(
    mesh: trimesh.Trimesh,
    grid: int = 512,
    thickness: float = 0.0375,
    fill_cavities: bool = True,
) -> tuple[trimesh.Trimesh, ShellReport]:
    """Turn a surface into a closed solid shell of the given wall thickness.

    Args:
        mesh: Any triangle soup. Orientation, holes and self-intersections do
            not matter: only where the surface *is* matters.
        grid: Cells along the longest side. The lattice is this plus a margin
            for the wall, so memory is about `(grid + wall)^3` times 12 bytes.
        thickness: The wall, as a fraction of the longest side. **The runner
            does not know millimetres**; 0.0375 is 3 mm on an 80 mm print.
        fill_cavities: Fill every pocket the outside cannot reach. Off, the
            shell is hollow wherever the surface was.

    Returns:
        A closed, consistently wound triangle mesh (surface nets over the
        occupancy; sheets that touch along an edge are separated by the
        caller, `split_manifold.make_manifold`), and the report.
    """
    report = ShellReport(grid=int(grid), thickness=float(thickness))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("cannot thicken a mesh with no faces")

    # --- rasterize the surface onto the corner lattice -------------------
    mark = time.perf_counter()
    referenced = vertices[np.unique(faces.ravel())]
    origin = referenced.min(axis=0)
    extent = float((referenced.max(axis=0) - origin).max())
    half = max(1, int(round(thickness * grid / 2)))
    pad = half + 2
    lattice = int(grid) + 2 * pad
    scale = (grid - 1) / max(extent, 1e-12)
    samples = _sample_surface(vertices, faces)
    corner = np.rint((samples - origin) * scale).astype(np.int64) + pad
    corner = corner.clip(0, lattice - 1)
    thin = np.zeros((lattice,) * 3, dtype=bool)
    thin[corner[:, 0], corner[:, 1], corner[:, 2]] = True
    del samples, corner
    report.half_cells = half
    report.lattice = lattice
    report.surface_corners = int(thin.sum())
    report.rasterize_sec = round(time.perf_counter() - mark, 2)

    # --- the wall: everything within half a wall of the surface -----------
    mark = time.perf_counter()
    distance = ndimage.distance_transform_edt(~thin)
    del thin
    band = distance <= half
    report.band_corners = int(band.sum())
    report.distance_sec = round(time.perf_counter() - mark, 2)

    # --- fill what the outside cannot reach --------------------------------
    mark = time.perf_counter()
    if fill_cavities:
        # Six-connected, so that the band - always at least three corners
        # thick - cannot be crossed diagonally.
        labels, count = ndimage.label(~band, structure=ndimage.generate_binary_structure(3, 1))
        outside_label = int(labels[0, 0, 0])
        occupied = labels != outside_label
        del labels
        report.cavities_filled = max(int(count) - 1, 0)
        report.cavity_corners = int(occupied.sum()) - report.band_corners
    else:
        occupied = band
    del band
    report.flood_sec = round(time.perf_counter() - mark, 2)

    # --- surface nets over the occupancy -------------------------------
    mark = time.perf_counter()
    quads: list[np.ndarray] = []
    inside_low: list[np.ndarray] = []
    points: list[np.ndarray] = []
    for axis in range(3):
        low = [slice(None)] * 3
        low[axis] = slice(0, -1)
        high = [slice(None)] * 3
        high[axis] = slice(1, None)
        crossing = occupied[tuple(low)] != occupied[tuple(high)]
        start = np.argwhere(crossing)
        if len(start) == 0:
            continue
        end = start.copy()
        end[:, axis] += 1
        # The wall's surface sits at `distance == half`; between a corner
        # inside the band and one outside, that is where the edge crosses it.
        d_low = distance[start[:, 0], start[:, 1], start[:, 2]]
        d_high = distance[end[:, 0], end[:, 1], end[:, 2]]
        span = d_high - d_low
        t = np.where(
            np.abs(span) > 1e-9, (half - d_low) / np.where(np.abs(span) > 1e-9, span, 1.0), 0.5
        )
        t = t.clip(0.0, 1.0)
        point = start.astype(np.float64)
        point[:, axis] += t
        points.append(point)
        inside_low.append(occupied[start[:, 0], start[:, 1], start[:, 2]])
        # The four cells sharing this edge. Cell (i, j, k) spans corners
        # i..i+1, so the cells around the edge from corner `start` are one
        # step back along the other two axes and forward by the offsets.
        b, c = _OTHER_AXES[axis]
        quad = np.empty((len(start), 4, 3), dtype=np.int64)
        for slot, (ob, oc) in enumerate(_QUAD_OFFSETS):
            cell = start.copy()
            cell[:, b] += ob - 1
            cell[:, c] += oc - 1
            quad[:, slot] = cell
        quads.append(quad)
    del distance, occupied
    if not quads:
        raise ValueError("the shell has no surface: the mesh may be smaller than one cell")

    quad = np.concatenate(quads)
    point = np.concatenate(points)
    flip = ~np.concatenate(inside_low)
    del quads, points, inside_low
    report.crossing_edges = int(len(quad))

    cells = lattice - 1
    linear = (quad[:, :, 0] * cells + quad[:, :, 1]) * cells + quad[:, :, 2]
    unique_cells, index = np.unique(linear.ravel(), return_inverse=True)
    index = index.reshape(-1, 4)
    # A cell's vertex is the mean of the crossing points on its edges - the
    # surface nets rule, which is what keeps the offset surface smooth rather
    # than stepped.
    position = np.zeros((len(unique_cells), 3), dtype=np.float64)
    weight = np.zeros(len(unique_cells), dtype=np.float64)
    for slot in range(4):
        np.add.at(position, index[:, slot], point)
        np.add.at(weight, index[:, slot], 1.0)
    position /= weight[:, None]
    position = (position - pad) / scale + origin

    # Outward is from the solid side to the empty side.
    index[flip] = index[flip][:, ::-1]
    # Split each quad along its shorter diagonal.
    p = position[index]
    diagonal_a = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    diagonal_b = np.linalg.norm(p[:, 1] - p[:, 3], axis=1)
    use_a = (diagonal_a <= diagonal_b)[:, None]
    split_a = index[:, [0, 1, 2, 0, 2, 3]]
    split_b = index[:, [0, 1, 3, 1, 2, 3]]
    triangles = np.where(use_a, split_a, split_b).reshape(-1, 3)

    shell = trimesh.Trimesh(vertices=position, faces=triangles, process=False)
    report.faces = int(len(shell.faces))
    report.vertices = int(len(shell.vertices))
    report.extract_sec = round(time.perf_counter() - mark, 2)
    return shell, report
