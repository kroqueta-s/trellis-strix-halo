# SPDX-License-Identifier: MIT
"""UV unwrapping and texture baking. **A picture on the surface, not a colour per vertex.**

Vertex colours (`pipeline._apply_vertex_colors`) give one sample per vertex and
interpolate across each triangle, so their resolution *is* the mesh's. A texture
map separates the two: the colour lives in an image, and a mesh of any density
can carry detail finer than its own triangles - the warning labels and part
numbers on a model become readable rather than smeared.

**The price is that a UV atlas is bound to the vertices and faces it was built
for.** Decimation, hole closing and the manifold conversion all replace them, so
the bake has to happen at a fixed point in the chain rather than at the end
(see `pipeline._postprocess`). In particular it runs **before** `make_manifold`,
which adds patch worth 2.2-2.5x the input surface area
(`post.manifold.close.fan_area_fraction`, measured 2026-09-12): an atlas built
after it would spend about 69% of its texels on internal membrane nobody sees.

**No `nvdiffrast` and no `cumesh`.** Unwrapping is xatlas (MIT), and the bake
rasterizes in UV space, where the triangles are flat, axis-aligned and - this is
what an atlas guarantees - do not overlap. There is no depth test to do, so a
few lines of barycentric arithmetic replace a rasterizer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image

# Triangles rasterized at once. **This sets the bake's VRAM peak**: each one
# holds every texel in its bounding box while it is tested. Measured nothing
# here - it is a safety valve, not a tuned number.
TRIANGLE_CHUNK = 1 << 18


@dataclass
class BakeReport:
    """What the bake did, counted rather than claimed."""

    texture_size: int = 0
    vertices_before: int = 0
    vertices_after: int = 0
    texels_covered: int = 0
    texels_total: int = 0
    texels_unreached: int = 0
    dilate_rounds: int = 0
    unwrap_sec: float = 0.0
    bake_sec: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items()}
        out["coverage"] = round(self.texels_covered / max(self.texels_total, 1), 4)
        # **The vertex duplication is how shattered the atlas is.** xatlas
        # copies a vertex once per chart it touches, so 3x means the average
        # chart is a handful of triangles - and an atlas of confetti carries
        # less than its texel count suggests.
        out["vertex_growth"] = round(self.vertices_after / max(self.vertices_before, 1), 2)
        return out


def unwrap(mesh: trimesh.Trimesh, in_process: bool = False) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, BakeReport
]:
    """Cut the surface open and lay it flat with xatlas.

    **The vertex count grows**: a vertex on a chart boundary needs one copy per
    chart it belongs to, because it has a different place in the atlas in each.
    Measured on this model's output, that is 2.4x to 4.7x.

    **It runs in a child process** (`unwrap_worker`), because xatlas holds the
    GIL for its whole run and would otherwise silence the heartbeat - see that
    module. `in_process` skips the child; it is for the tests, which unwrap
    shapes small enough that nothing has time to notice.

    Returns the new vertices, faces and per-vertex UVs.
    """
    report = BakeReport(vertices_before=int(len(mesh.vertices)))
    positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    triangles = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

    mark = time.perf_counter()
    if in_process:
        from . import unwrap_worker

        vmapping, indices, uvs = unwrap_worker.unwrap(positions, triangles)
    else:
        import concurrent.futures
        import multiprocessing

        from . import unwrap_worker

        # **Spawn, not fork**: this is Windows, and the parent holds a CUDA
        # context that must not be inherited even where fork exists.
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=context) as pool:
            vmapping, indices, uvs = pool.submit(
                unwrap_worker.unwrap, positions, triangles
            ).result()
    report.unwrap_sec = round(time.perf_counter() - mark, 2)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)[vmapping]
    faces = np.asarray(indices, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)
    report.vertices_after = int(len(vertices))
    return vertices, faces, uvs, report


def _cover(
    uvs: torch.Tensor, faces: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Which texel belongs to which triangle, and where inside it.

    Walks the triangles' bounding boxes in texel space and keeps the texel
    centres that fall inside. **An atlas's triangles do not overlap**, so a
    texel claimed twice is a seam artefact rather than a depth question, and the
    later write simply wins.

    Returns the flat texel index, the triangle index and the barycentric weights.
    """
    corners = uvs[faces] * size  # [F, 3, 2]
    low = torch.floor(corners.amin(dim=1)).long().clamp(0, size - 1)
    high = torch.ceil(corners.amax(dim=1)).long().clamp(0, size)
    span = (high - low).clamp(min=0)
    counts = span[:, 0] * span[:, 1]

    keep = counts > 0
    if not bool(keep.any()):
        empty = torch.zeros(0, dtype=torch.long, device=uvs.device)
        return empty, empty, torch.zeros(0, 3, device=uvs.device)

    index = torch.nonzero(keep, as_tuple=False).squeeze(1)
    counts, low, span = counts[index], low[index], span[index]

    # One entry per (triangle, texel in its box), built without a python loop:
    # the offset inside each box is the running position minus where that box
    # started.
    total = int(counts.sum())
    triangle = torch.repeat_interleave(torch.arange(index.numel(), device=uvs.device), counts)
    starts = torch.cumsum(counts, dim=0) - counts
    offset = torch.arange(total, device=uvs.device) - starts[triangle]
    width = span[triangle, 0]
    x = low[triangle, 0] + offset % width
    y = low[triangle, 1] + offset // width

    # The texel's centre, in the same units as the corners.
    point = torch.stack([x.float() + 0.5, y.float() + 0.5], dim=1)
    tri = corners[index][triangle]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2, ep = v1 - v0, v2 - v0, point - v0
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    # A triangle with no area in the atlas cannot claim a texel.
    safe = torch.where(area.abs() < 1e-12, torch.ones_like(area), area)
    beta = (ep[:, 0] * e2[:, 1] - ep[:, 1] * e2[:, 0]) / safe
    gamma = (e1[:, 0] * ep[:, 1] - e1[:, 1] * ep[:, 0]) / safe
    alpha = 1.0 - beta - gamma
    inside = (
        (alpha >= 0) & (beta >= 0) & (gamma >= 0) & (area.abs() >= 1e-12)
    )

    texel = (y * size + x)[inside]
    return texel, index[triangle][inside], torch.stack([alpha, beta, gamma], dim=1)[inside]


def _dilate(colour: torch.Tensor, filled: torch.Tensor, rounds: int) -> tuple[torch.Tensor, int]:
    """Spread the edge colours outward, so filtering across a seam picks up the chart.

    A texel just outside a chart is sampled whenever the renderer filters near a
    seam. Left at zero it draws a black line along every cut. **This is not
    inpainting** - it copies the nearest filled neighbour and stops when nothing
    changed, which is all a seam needs.
    """
    size = int(colour.shape[0])
    done = 0
    for _ in range(rounds):
        if bool(filled.all()):
            break
        padded = torch.nn.functional.pad(
            colour.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
        )
        weight = torch.nn.functional.pad(
            filled.float().reshape(1, 1, size, size), (1, 1, 1, 1), mode="replicate"
        )
        total = torch.nn.functional.avg_pool2d(weight, 3, stride=1) * 9
        summed = torch.nn.functional.avg_pool2d(padded * weight, 3, stride=1) * 9
        neighbour = (summed / total.clamp_min(1e-6)).squeeze(0).permute(1, 2, 0)
        grew = (total.squeeze() > 0) & ~filled
        if not bool(grew.any()):
            break
        colour = torch.where(grew.unsqueeze(-1), neighbour, colour)
        filled = filled | grew
        done += 1
    return colour, done


def bake(
    mesh: trimesh.Trimesh,
    query: Any,
    size: int,
    dilate: int = 4,
    in_process: bool = False,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Unwrap `mesh` and bake the colours `query` returns into a texture.

    Args:
        mesh: The mesh to texture. **Its vertices are replaced** by the
            unwrapped ones, which is why the result is returned rather than the
            original being changed.
        query: Takes `[N, 3]` positions on the device and returns `[N, 3]` RGB
            in 0..1. `pipeline` passes the texture decoder's voxels.
        size: The texture is `size` x `size`.
        dilate: How many rounds to spread the colours past the chart edges.

    Returns:
        The unwrapped mesh carrying the texture, and what the bake counted.
    """
    vertices, faces, uvs, report = unwrap(mesh, in_process=in_process)
    report.texture_size = int(size)
    report.texels_total = int(size) * int(size)

    mark = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uv_t = torch.as_tensor(uvs, dtype=torch.float32, device=device)
    face_t = torch.as_tensor(faces, dtype=torch.long, device=device)
    vert_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)

    colour = torch.zeros(size * size, 3, dtype=torch.float32, device=device)
    filled = torch.zeros(size * size, dtype=torch.bool, device=device)
    for start in range(0, int(face_t.shape[0]), TRIANGLE_CHUNK):
        chunk = face_t[start : start + TRIANGLE_CHUNK]
        texel, triangle, weights = _cover(uv_t, chunk, size)
        if texel.numel() == 0:
            continue
        corners = vert_t[chunk[triangle]]  # [M, 3, 3]
        point = (corners * weights.unsqueeze(-1)).sum(dim=1)
        colour[texel] = query(point)
        filled[texel] = True

    report.texels_covered = int(filled.sum())
    colour = colour.view(size, size, 3)
    colour, rounds = _dilate(colour, filled.view(size, size), dilate)
    report.dilate_rounds = rounds
    report.texels_unreached = report.texels_total - int(filled.sum())

    image = Image.fromarray(
        (colour.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy(), mode="RGB"
    )
    # **The atlas's V axis points the other way from an image's rows**, so the
    # coordinate is flipped here rather than the picture being stored upside
    # down: a texture that looks wrong when opened is one nobody can check.
    flipped = np.column_stack([uvs[:, 0], 1.0 - uvs[:, 1]])
    textured = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=flipped, image=image),
        process=False,
    )
    report.bake_sec = round(time.perf_counter() - mark, 2)
    return textured, report.as_dict()
